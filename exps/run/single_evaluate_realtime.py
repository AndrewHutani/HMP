import time
import torch
import numpy as np
import argparse
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

def resample_sequence(sequence, downsample_rate, total_length, start_idx = 0):
    """
    Resample a sequence (downsample or upsample) to match the logic in single_evaluate_physmop.py.

    Args:
        sequence: torch.Tensor, shape [num_frames, num_joints, 3]
        downsample_rate: float
        total_length: int, number of frames to output

    Returns:
        torch.Tensor, shape [total_length, num_joints, 3]
    """
    if downsample_rate >= 1.0 or np.isclose(downsample_rate, 1.0):
        # Downsample: select frames at intervals
        start_index = start_idx
        end_index = int(total_length * downsample_rate) + start_idx
        indices = np.round(np.linspace(start_index, end_index, total_length)).astype(int)
        indices = np.clip(indices, 0, sequence.shape[0] - 1)
        return sequence[indices]
    else:
        # Upsample: interpolate to more frames
        upsample_factor = 1.0 / downsample_rate
        orig_time_steps = sequence.shape[0]
        new_time_steps = int(np.round(orig_time_steps * upsample_factor))
        seq_perm = sequence.permute(1, 2, 0)  # [num_joints, 3, time]
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
walking_sample, root_sample = dataset.get_full_sequences_for_action(action)[0]
# print("Walking sample shape: ", walking_sample.shape)
# print("Root sample shape: ", root_sample.shape)
# print("Root sample: ", root_sample[:, 0, :])

realtime_predictor = RealTimeGlobalPrediction(model, config, tau=0.5)
visualize = False
debug = False


all_observed_motion = []
all_predicted_motion = []
latency_times = []
downsample_rates = np.arange(3, 0.1, -0.2)  # From 3 to 0.1, step -0.2


for downsample_rate in downsample_rates:
    print(f"Downsample rate: {downsample_rate:.2f}")
    total_length = config.motion.h36m_target_length + config.motion.h36m_target_length_eval  # or your desired length
    walking_sample_resampled = resample_sequence(walking_sample, downsample_rate, total_length, start_idx=100)
    print("Resampled walking sample shape: ", walking_sample_resampled.shape)
    visualize_continuous_motion(walking_sample_resampled, title="Ground Truth Motion", skeleton_type="h36m",
                                save_gif_path="output_ground_truth_{}.gif".format(downsample_rate))

# for i in range(walking_sample.shape[0] - config.motion.h36m_target_length):
# # for i in range(100):
#     test_input_ = walking_sample[i]
#     ground_truth = walking_sample[i:i+config.motion.h36m_target_length]
#     realtime_predictor.predict(test_input_, ground_truth, visualize, debug)
#     global_observed_motion = realtime_predictor.add_global_translation()  # Add global translation to the predicted motion

#     all_observed_motion.append(global_observed_motion[-config.motion.h36m_target_length:])
#     all_predicted_motion.append(realtime_predictor.predicted_motion)
