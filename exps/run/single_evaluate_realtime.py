import time
import torch
import numpy as np
import argparse

from tqdm import tqdm
from config import config

from realtime import RealTimePrediction

from model import GCNext as Model
from datasets.h36m_eval import H36MEval
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from prediction_times import prediction_times
from visualize_motion import visualize_continuous_motion

import torch.nn.functional as F
import math

def hann_lowpass_1d(x, cutoff_frac, k_min=5):
    """
    x: [T, J, 3] (time=dim0). cutoff_frac in (0,1], relative to Nyquist (1.0 == Nyquist).
    Simple Hann-windowed low-pass via depthwise 1D conv along time.
    """
    T = x.shape[0]
    if T < 3:
        return x
    k = max(k_min, int(math.ceil(8 / max(1e-6, cutoff_frac)))) | 1
    t = torch.arange(k, device=x.device, dtype=x.dtype)
    n = t - (k - 1) / 2
    fc = 0.5 * cutoff_frac
    h = torch.sinc((2 * fc) * n)
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * (t / (k - 1)))
    h = (h * w).to(x.dtype)
    h = h / h.sum()
    pad = k // 2
    # [T, J, 3] -> [J*3, T]
    x_flat = x.reshape(T, -1).transpose(0, 1)
    h = h.view(1, 1, k).to(x.device, x.dtype)
    y = F.conv1d(F.pad(x_flat.unsqueeze(0), (pad, pad), mode='replicate'),
                 h.expand(x_flat.shape[0], 1, k),
                 groups=x_flat.shape[0])
    y = y.squeeze(0).transpose(0, 1).reshape(T, x.shape[1], x.shape[2])
    return y

def resample_sequence(sequence, downsample_rate, total_length, start_idx = 0, antialias=True):
    """
    Resample a sequence (downsample or upsample) to match the logic in single_evaluate_physmop.py.
    Args:
        sequence: torch.Tensor, shape [num_frames, num_joints, 3]
        downsample_rate: float
        total_length: int, number of frames to output
    Returns:
        torch.Tensor, shape [total_length, num_joints, 3]
    """
    device = sequence.device
    dtype = sequence.dtype
    if downsample_rate >= 1.0 or np.isclose(downsample_rate, 1.0):
        # Downsample: interpolate with anti-aliasing
        src_span = int(round(total_length * downsample_rate))
        src_end = min(start_idx + src_span, sequence.shape[0])
        src_seq = sequence[start_idx:src_end]
        # Anti-aliasing
        if antialias and downsample_rate > 1.0 and src_seq.shape[0] > 8:
            cutoff_frac = min(1.0 / downsample_rate, 0.9)
            src_seq = hann_lowpass_1d(src_seq, cutoff_frac)
        # Interpolate
        t_src = torch.arange(src_seq.shape[0], device=device, dtype=dtype)
        t_target = torch.linspace(src_seq.shape[0] - 1, 0, total_length, device=device, dtype=dtype)
        t_target = torch.clamp(t_target, 0, src_seq.shape[0] - 1)
        t0 = torch.floor(t_target).long()
        t1 = torch.clamp(t0 + 1, max=src_seq.shape[0] - 1)
        w = (t_target - t0.to(dtype)).view(-1, 1, 1)
        x0 = src_seq[t0]
        x1 = src_seq[t1]
        y = (1 - w) * x0 + w * x1
        # Reverse to chronological order
        y = torch.flip(y, dims=[0])
        return y
    else:
        # Upsample: interpolate to more frames
        upsample_factor = 1.0 / downsample_rate
        seq_slice = sequence[start_idx : start_idx + total_length]
        orig_time_steps = seq_slice.shape[0]
        new_time_steps = int(np.round(orig_time_steps * upsample_factor))
        seq_perm = seq_slice.permute(1, 2, 0)  # [num_joints, 3, time]
        seq_upsampled = F.interpolate(seq_perm, size=new_time_steps, mode='linear', align_corners=True)
        seq_upsampled = seq_upsampled.permute(2, 0, 1)  # [time, num_joints, 3]
        return seq_upsampled[:total_length]

class RealTimeGlobalPrediction(RealTimePrediction):
    # def add_global_translation(self, root_translation=None):
    #     """
    #     Add global translation to the predicted motion based on the root translation.

    #     :param root_translation: Optional; if provided, will use this translation instead of calculating it. tensor([num_frames, 32, 3])
    #     """
    #     global_observed_motion = []
    #     if root_translation is None:
    #         CONSTANT_VELOCITY = 0.03  # Adjust this value as needed
            
    #         # observed_motion: list of torch tensors, each [num_joints, 3]
    #         offsets = torch.arange(len(self.observed_motion), dtype=self.observed_motion[0].dtype, device=self.observed_motion[0].device) * CONSTANT_VELOCITY
            
    #         for i in range(len(self.observed_motion)):
    #             frame = self.observed_motion[i].clone()  # Clone to avoid modifying the original
    #             frame[:, 2] += offsets[i]                # Add offset to the z-coordinate
    #             global_observed_motion.append(frame)
                    
    #         timesteps = self.predicted_motion.shape[0] + len(self.observed_motion)

    #         offsets = np.arange(timesteps) * CONSTANT_VELOCITY

    #         offsets = offsets.reshape(-1, 1)  # Reshape to [timesteps, 1]
            

    #         for i in range(self.predicted_motion.shape[0]):
    #             self.predicted_motion[i, :, 2] += offsets[len(self.observed_motion) + i]

    #     else:
    #         root_translation_observed = root_translation[:len(self.observed_motion), :, :]  # shape: (num_frames, 32, 3)
    #         for i in range(len(self.observed_motion)):
    #             global_observed_motion.append(self.observed_motion[i] + root_translation_observed[i])  # Add root translation to each observed frame
            
    #         for i in range(self.predicted_motion.shape[0]):
    #             self.predicted_motion[i] += root_translation[len(self.observed_motion) + i].cpu().numpy()
            

    #     return global_observed_motion
    def add_global_translation(self):
        CONSTANT_VELOCITY = 0.03  # Adjust this value as needed
        
        # observed_motion: list of torch tensors, each [num_joints, 3]
        offsets = torch.arange(len(self.observed_motion), dtype=self.observed_motion[0].dtype, device=self.observed_motion[0].device) * CONSTANT_VELOCITY
        global_observed_motion = []
        for i in range(len(self.observed_motion)):
            frame = self.observed_motion[i].clone()  # Clone to avoid modifying the original
            frame[:, 2] += offsets[i]                # Add offset to the z-coordinate
            global_observed_motion.append(frame)
                
        timesteps = self.predicted_motion.shape[0] + len(self.observed_motion)

        offsets = np.arange(timesteps) * CONSTANT_VELOCITY

        offsets = offsets.reshape(-1, 1)  # Reshape to [timesteps, 1]
        

        for i in range(self.predicted_motion.shape[0]):
            self.predicted_motion[i, :, 2] += offsets[len(self.observed_motion) + i]

        return global_observed_motion

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="ckpt/baseline/model-iter-84000.pth", help='=encoder path')
parser.add_argument('--dyna', nargs='+', type=int, default=[0, 48], help='dynamic layer index')
args = parser.parse_args()

# Prepare model
model = Model(config, args.dyna)
state_dict = torch.load(args.model_pth, map_location = torch.device("cpu"))
model.load_state_dict(state_dict, strict=True)
model.to(torch.device("cpu"))  # Use CPU for inference


actions = ["walking", "eating", "smoking", "discussion", "directions",
                        "greeting", "phoning", "posing", "purchases", "sitting",
                        "sittingdown", "takingphoto", "waiting", "walkingdog",
                        "walkingtogether"]

action = "walking"  # Change this to the action you want to evaluate

config.motion.h36m_target_length = config.motion.h36m_target_length_eval
dataset = H36MEval(config, 'test')
mpjpe_data_all = []
total_number_of_predictions = []
for walking_sample, _ in dataset.get_full_sequences_for_action(action):
    # walking_sample, root_sample = dataset.get_full_sequences_for_action(action)[0]
# print("Walking sample shape: ", walking_sample.shape)
# print("Root sample shape: ", root_sample.shape)
# print("Root sample: ", root_sample[:, 0, :])
    mpjpe_data_per_sample = []

    realtime_predictor = RealTimeGlobalPrediction(model, config, tau=0.5)
    visualize = False
    debug = False


    all_observed_motion = []
    all_predicted_motion = []
    latency_times = []
    downsample_rates = np.arange(3, 0.1, -0.2)  # From 3 to 0.1, step -0.2

    input_len = config.motion.h36m_input_length
    output_len = config.motion.h36m_target_length_eval

    num_predictions_per_rate = []

    for downsample_rate in downsample_rates:
        mpjpe_data_per_downsample_rate = []
        print(f"Downsample rate: {downsample_rate:.2f}")
        total_length = input_len + output_len

        # Compute the maximum valid start index for the window
        if downsample_rate >= 1.0 or np.isclose(downsample_rate, 1.0):
            max_end_index = int(input_len * downsample_rate) + output_len
            max_start_idx = walking_sample.shape[0] - max_end_index
            max_start_idx = max(0, max_start_idx)
        else:
            max_start_idx = walking_sample.shape[0] - total_length
            max_start_idx = max(0, max_start_idx)

        stride = 1
        print(f"Max start idx: {max_start_idx}")
        for start_idx in tqdm(range(0, max_start_idx + 1, stride), desc="Processing"):
            # 1. Resample only the input part
            src_span = int(round(input_len * downsample_rate))
            src_end = min(start_idx + src_span, walking_sample.shape[0])
            input_resampled = resample_sequence(
                                walking_sample,
                                downsample_rate,
                                input_len,
                                start_idx=start_idx,
                                antialias=False
                            )

            output_start = src_end
            output_end = output_start + output_len
            # Make sure we don't go out of bounds
            if output_end > walking_sample.shape[0]:
                break
            output = walking_sample[output_start:output_end]

            # 4. Concatenate for evaluation
            walking_sample_resampled = torch.cat([input_resampled, output], dim=0)
            test_input_ = walking_sample_resampled[:config.motion.h36m_input_length]
            ground_truth = walking_sample_resampled[config.motion.h36m_input_length:]
            realtime_predictor.batch_predict(test_input_, ground_truth, visualize=False, debug=False)
            mpjpe_data = realtime_predictor.evaluate() # Shape (4, )
            # visualize_continuous_motion(walking_sample_resampled, skeleton_type='h36m',
            #                             save_gif_path='output_{}.gif'.format(downsample_rate))


            mpjpe_data_per_downsample_rate.append(mpjpe_data)
            # visualize_continuous_motion(walking_sample_resampled, skeleton_type='h36m',
            #                             save_gif_path='output_{}.gif'.format(downsample_rate))
            # break
        num_predictions_per_rate.append(len(mpjpe_data_per_downsample_rate))
        # break
        mpjpe_data_per_sample.append(np.mean(np.array(mpjpe_data_per_downsample_rate), axis=0))
    total_number_of_predictions.append(num_predictions_per_rate)
    mpjpe_data_all.append(np.array(mpjpe_data_per_sample))

total_number_of_predictions = np.sum(np.array(total_number_of_predictions), axis=0)
print("Total number of predictions per downsample rate: ", total_number_of_predictions)
mpjpe_data_all = np.array(mpjpe_data_all) # shape: (num_samples, num_downsample_rates, 4)
mpjpe_data_mean = np.mean(mpjpe_data_all, axis=0)  # shape: (num_downsample_rates, 4)
mpjpe_data_std = np.std(mpjpe_data_all, axis=0)  # shape: (num_downsample_rates, 4)
# Stack the mean and n_predictions as columns
mean_with_n = np.column_stack((mpjpe_data_mean, total_number_of_predictions))
header = ["80ms", "400ms", "560ms", "1000ms", "n_predictions"]

np.savetxt("resampled_gcn_consistent_output_gcn_mean.csv", mean_with_n, delimiter=",",  header=",".join(header))
np.savetxt("resampled_gcn_consistent_output_gcn_std.csv", mpjpe_data_std, delimiter=",")

plt.figure()
for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    plt.plot(downsample_rates, mpjpe_data_all[:, i], marker='o', label=label)
plt.xlabel('Downsample Rate')
plt.ylabel('Mean MPJPE')
plt.title("MPJPE for the GCNext model vs Downsample Rate")
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend(title='Timesteps into the future')
# plt.ylim(y_limits)
# plt.savefig(branch_filenames[branch])
plt.show()
# plt.close()