import torch
import numpy as np
import argparse
import os
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from models.gcnext.config import config

from models.gcnext.RealtimeGCNext import RealtimeGCNext

from models.gcnext.model import GCNext as Model
from dataset.gcnext.h36m_eval import H36MEval

from exps.C_temporal_resolution.utils_resample_eval import resample_sequence_gcnext as resample_sequence

save_directory = "exps/C_temporal_resolution/performance_logs/"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="ckpt/baseline/hist_length_50.pth", help='=encoder path')
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
    mpjpe_data_per_sample = []

    realtime_predictor = RealtimeGCNext(model, config, tau=0.5)
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
        num_predictions_per_rate.append(len(mpjpe_data_per_downsample_rate))
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

np.savetxt(save_directory + "resampled_gcn_consistent_output_gcn_mean.csv", mean_with_n, delimiter=",",  header=",".join(header))
np.savetxt(save_directory + "resampled_gcn_consistent_output_gcn_std.csv", mpjpe_data_std, delimiter=",")

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