import torch
import numpy as np
import argparse

from tqdm import tqdm
from config import config

from realtime import RealTimePrediction

from model import GCNext as Model
from datasets.h36m_eval import H36MEval


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="ckpt/baseline/model-iter-84000.pth", help='=encoder path')
parser.add_argument('--dyna', nargs='+', type=int, default=[0, 48], help='dynamic layer index')
args = parser.parse_args()

# Prepare model
model = Model(config, args.dyna)
state_dict = torch.load(args.model_pth)
model.load_state_dict(state_dict, strict=True)


actions = ["walking", "eating", 
           "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"
            ]

log_filename = "mpjpe_log.txt"
with open(log_filename, "w") as log_file:
    for action in actions:
        print(f"Evaluating action: {action}")
        config.motion.h36m_target_length = config.motion.h36m_target_length_eval
        dataset = H36MEval(config, 'test')
        action_indices = dataset.get_indices_for_action(action)

        realtime_predictor = RealTimePrediction(model, config, tau=0.5)
        visualize = False
        debug = False
        mpjpe_upper_all_samples = []
        mpjpe_lower_all_samples = []

        for i, idx in tqdm(enumerate(action_indices), desc=f"{action} Progress"):  # <-- Use tqdm here
            with torch.no_grad():
                test_input, test_output = dataset.__getitem__(idx)
                full_motion = torch.cat([test_input, test_output], dim=0)
                realtime_predictor = RealTimePrediction(model, config, tau=0.5)  # re-init to clear state
                mpjpe_upper_per_obs = []
                mpjpe_lower_per_obs = []
                for i in range(test_input.shape[0]):
                    test_input_ = test_input[:i+1]
                    ground_truth = full_motion[i+1:i+1+config.motion.h36m_target_length_eval, :, :]
                    motion_input = realtime_predictor.batch_predict(test_input_, ground_truth, visualize, debug)
                    mpjpe_upper, mpjpe_lower = realtime_predictor.evaluate_upper_and_lower_seperately()  # shape: (4,)
                    mpjpe_upper_per_obs.append(mpjpe_upper)
                    mpjpe_lower_per_obs.append(mpjpe_lower)
                    # if i == test_input.shape[0] - 1:
                    #     realtime_predictor.visualize_input_and_output(gif_path=f"realtime_{action}_input_output.gif")
                mpjpe_upper_all_samples.append(np.stack(mpjpe_upper_per_obs))  # shape: (obs_len, 4)
                mpjpe_lower_all_samples.append(np.stack(mpjpe_lower_per_obs))  # shape: (obs_len, 4)

        print(mpjpe_upper_all_samples)
        mpjpe_upper_all_samples = np.stack(mpjpe_upper_all_samples)  # shape: (num_samples, obs_len, 4)
        mpjpe_lower_all_samples = np.stack(mpjpe_lower_all_samples)  # shape: (num_samples, obs_len, 4)

        mpjpe_upper_mean = np.mean(mpjpe_upper_all_samples, axis=0)  # shape: (obs_len, 4)
        mpjpe_lower_mean = np.mean(mpjpe_lower_all_samples, axis=0)  # shape: (obs_len, 4)

        # Write to log file
        log_file.write(f"Averaged MPJPE (upper body) for each observation length and each selected timestep: {action}\n")
        for obs_len in range(mpjpe_upper_mean.shape[0]):
            log_file.write(f"Obs {obs_len+1}: {mpjpe_upper_mean[obs_len]}\n")
        log_file.write(f"Averaged MPJPE (lower body) for each observation length and each selected timestep: {action}\n")
        for obs_len in range(mpjpe_lower_mean.shape[0]):
            log_file.write(f"Obs {obs_len+1}: {mpjpe_lower_mean[obs_len]}\n")
        log_file.write("\n")

print(f"MPJPE logs saved to {log_filename}")