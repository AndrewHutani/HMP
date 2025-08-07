import torch
import numpy as np
import argparse
import os
import glob

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



log_filename = "gcnext_on_amass.txt"
with open(log_filename, "w") as log_file:
    with torch.no_grad():
        directory = "data/data_processed/physmop_to_gcn"

        realtime_predictor = RealTimePrediction(model, config, tau=0.5)
        visualize = False
        debug = False
        mpjpe_upper_all_samples = []
        mpjpe_lower_all_samples = []

        for txt_file in tqdm(glob.glob(os.path.join(directory, "*.txt")), desc="Evaluating samples"):

            amass_sample = np.loadtxt(txt_file, delimiter=',')
            amass_sample = amass_sample.reshape(-1, 22, 3)  # Reshape to [num_frames, num_joints, 3]
            # the xyz axes are also not in the same order so reorder them
            amass_sample = np.stack((amass_sample[:, :, 1], 
                                        amass_sample[:, :, 2],
                                        amass_sample[:, :, 0]), axis=2)  # [num_frames, num_joints, 3]
            mpjpe_upper_per_obs = []
            mpjpe_lower_per_obs = []
            for i in range(amass_sample.shape[0]-config.motion.h36m_target_length_eval):
                test_input_ = torch.tensor(amass_sample[i], dtype=torch.float32)
                ground_truth = torch.tensor(amass_sample[i:i+config.motion.h36m_target_length_eval, :, :], dtype=torch.float32)
                motion_input = realtime_predictor.predict(test_input_, ground_truth, visualize, debug)
                mpjpe_upper, mpjpe_lower = realtime_predictor.evaluate_upper_and_lower_seperately()  # shape: (4,)
                mpjpe_upper_per_obs.append(mpjpe_upper)
                mpjpe_lower_per_obs.append(mpjpe_lower)
            mpjpe_upper_all_samples.append(np.stack(mpjpe_upper_per_obs))  # shape: (obs_len, 4)
            mpjpe_lower_all_samples.append(np.stack(mpjpe_lower_per_obs))  # shape: (obs_len, 4)
        
        mpjpe_upper_all_samples = np.stack(mpjpe_upper_all_samples)  # shape: (num_samples, obs_len, 4)
        mpjpe_lower_all_samples = np.stack(mpjpe_lower_all_samples)  # shape: (num_samples, obs_len, 4)

        mpjpe_upper_mean = np.mean(mpjpe_upper_all_samples, axis=0)  # shape: (obs_len, 4)
        mpjpe_lower_mean = np.mean(mpjpe_lower_all_samples, axis=0)  # shape: (obs_len, 4)

        # Write to log file
        log_file.write(f"Averaged MPJPE (upper body) for each observation length and each selected timestep:\n")
        for obs_len in range(mpjpe_upper_mean.shape[0]):
            log_file.write(f"Obs {obs_len+1}: {mpjpe_upper_mean[obs_len]}\n")
        log_file.write(f"Averaged MPJPE (lower body) for each observation length and each selected timestep:\n")
        for obs_len in range(mpjpe_lower_mean.shape[0]):
            log_file.write(f"Obs {obs_len+1}: {mpjpe_lower_mean[obs_len]}\n")
        log_file.write("\n")

print(f"MPJPE logs saved to {log_filename}")