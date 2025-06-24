import torch
import numpy as np
import argparse
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


actions = ["walking", "eating", "smoking", "discussion", "directions",
                        "greeting", "phoning", "posing", "purchases", "sitting",
                        "sittingdown", "takingphoto", "waiting", "walkingdog",
                        "walkingtogether"]

# actions = ["walking", "eating"]

log_filename = "longer_term_mpjpe_log.txt"
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

        # Get full sequence for the action
        action_sequences = dataset.get_full_sequences_for_action(action)
        prediction_horizon = 50

        for idx, action_sequence in enumerate(action_sequences):
            print("Progress: {}/{} for action {}".format(idx+1, len(action_sequences), action))
            # test_input, test_output = dataset.__getitem__(idx)
            full_motion = action_sequence[0]
            realtime_predictor = RealTimePrediction(model, config, tau=0.5, total_prediction_horizon=prediction_horizon)  # re-init to clear state
            mpjpe_upper_per_obs = []
            mpjpe_lower_per_obs = []
            for i in range(full_motion.shape[0] - prediction_horizon - config.motion.h36m_input_length_dct):
            # for i in range(100):
                # print("i: {}".format(i))
                test_input_ = full_motion[i:config.motion.h36m_input_length_dct + i]
                ground_truth = full_motion[config.motion.h36m_input_length_dct + i:
                                           config.motion.h36m_input_length_dct + i + prediction_horizon, :, :]
                motion_input = realtime_predictor.batch_predict(test_input_, ground_truth, visualize, debug)
                mpjpe_upper, mpjpe_lower = realtime_predictor.evaluate_upper_and_lower_seperately()  # shape: (5,)
                mpjpe_upper_per_obs.append(mpjpe_upper)
                mpjpe_lower_per_obs.append(mpjpe_lower)
                # if i == test_input.shape[0] - 1:
                #     realtime_predictor.visualize_input_and_output(gif_path=f"realtime_{action}_input_output.gif")
            mpjpe_upper_all_samples.append(np.mean(np.array(mpjpe_upper_per_obs), axis=0))  # shape: (5,)
            mpjpe_lower_all_samples.append(np.mean(np.array(mpjpe_lower_per_obs), axis=0))  # shape: (5,)

        mpjpe_upper_all_samples = np.mean(np.array(mpjpe_upper_all_samples), axis=0) 
        mpjpe_lower_all_samples = np.mean(np.array(mpjpe_lower_all_samples), axis=0)  # shape: (num_samples, N, 5)

        # mpjpe_upper_mean = np.mean(mpjpe_upper_all_samples, axis=0)  # shape: (obs_len, 5)
        # mpjpe_lower_mean = np.mean(mpjpe_lower_all_samples, axis=0)  # shape: (obs_len, 5)

        # mpjpe_upper_mean = np.mean(mpjpe_upper_mean, axis=2)  # shape: (5,)
        # mpjpe_lower_mean = np.mean(mpjpe_lower_mean, axis=2)  # shape: (5,)

        # Write to log file
        log_file.write(f"Averaged MPJPE (upper body) for {action}: {mpjpe_upper_all_samples}\n")
        log_file.write(f"Averaged MPJPE (lower body) for {action}: {mpjpe_lower_all_samples}\n")
        log_file.write("\n")

print(f"MPJPE logs saved to {log_filename}")