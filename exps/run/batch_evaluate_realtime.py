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
state_dict = torch.load(args.model_pth, map_location=torch.device('cpu'))
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

            # Divide the sample into smaller samples of size config.motion.h36m_input_length + config.motion.h36m_target_length_eval
            sample_length = config.motion.h36m_input_length + config.motion.h36m_target_length_eval
            num_subsamples = amass_sample.shape[0] // sample_length

            subsamples = []
            for j in range(num_subsamples):
                start_idx = j * sample_length
                end_idx = start_idx + sample_length
                subsample = amass_sample[start_idx:end_idx]
                subsamples.append(subsample)
            if not subsamples:
                print(f"No valid subsamples found in {txt_file}")
                continue

            mpjpe_upper_per_sample = []
            mpjpe_lower_per_sample = []
            for subsample in subsamples:
                mpjpe_upper_per_subsample = []
                mpjpe_lower_per_subsample = []
                for i in range(subsample.shape[0]-config.motion.h36m_target_length_eval):
                    test_input_ = torch.tensor(subsample[i], dtype=torch.float32)
                    ground_truth = torch.tensor(subsample[i:i+config.motion.h36m_target_length_eval, :, :], dtype=torch.float32)
                    motion_input = realtime_predictor.predict(test_input_, ground_truth, visualize, debug)
                    mpjpe_upper, mpjpe_lower = realtime_predictor.evaluate_upper_and_lower_seperately()  # shape: (4,)
                    mpjpe_upper_per_subsample.append(mpjpe_upper)
                    mpjpe_lower_per_subsample.append(mpjpe_lower)
                mpjpe_upper_per_sample.append(np.array(mpjpe_upper_per_subsample))  # shape: (4, 50)
                mpjpe_lower_per_sample.append(np.array(mpjpe_lower_per_subsample))  # shape: (4, 50)
            mpjpe_upper_all_samples.append(np.mean(mpjpe_upper_per_sample, axis=0))  # shape: (4, 50)
            mpjpe_lower_all_samples.append(np.mean(mpjpe_lower_per_sample, axis=0))  # shape: (4, 50)

        mpjpe_upper_all_samples = np.array(mpjpe_upper_all_samples)  # shape: (num_samples, obs_len, 4)
        mpjpe_lower_all_samples = np.array(mpjpe_lower_all_samples)  # shape: (num_samples, obs_len, 4)

        mpjpe_upper_mean = np.mean(mpjpe_upper_all_samples, axis=0)  # shape: (obs_len, 4)
        mpjpe_upper_std = np.std(mpjpe_upper_all_samples, axis=0)  # shape: (obs_len, 4)
        mpjpe_lower_mean = np.mean(mpjpe_lower_all_samples, axis=0)  # shape: (obs_len, 4)

        mpjpe_upper_std = np.std(mpjpe_upper_all_samples, axis=0)
        mpjpe_lower_std = np.std(mpjpe_lower_all_samples, axis=0)
        percentile_25_upper = np.percentile(mpjpe_upper_all_samples, 25, axis=0)
        percentile_75_upper = np.percentile(mpjpe_upper_all_samples, 75, axis=0)
        percentile_25_lower = np.percentile(mpjpe_lower_all_samples, 25, axis=0)
        percentile_75_lower = np.percentile(mpjpe_lower_all_samples, 75, axis=0)
        

        # Write to log file
        log_file.write(f"Averaged MPJPE (upper body) for each observation length and each selected timestep:\n")
        for obs_len in range(mpjpe_upper_mean.shape[0]):
            log_file.write(f"Obs {obs_len+1}: {mpjpe_upper_mean[obs_len]}\n")
            log_file.write(f"Std {obs_len+1}: {mpjpe_upper_std[obs_len]}\n")
        log_file.write(f"Averaged MPJPE (lower body) for each observation length and each selected timestep:\n")
        for obs_len in range(mpjpe_lower_mean.shape[0]):
            log_file.write(f"Obs {obs_len+1}: {mpjpe_lower_mean[obs_len]}\n")
            log_file.write(f"Std {obs_len+1}: {mpjpe_lower_std[obs_len]}\n")

        log_file.write("25th percentile MPJPE (upper body) for each observation length and each selected timestep:\n")
        for obs_len in range(percentile_25_upper.shape[0]):
            log_file.write(f"Obs {obs_len+1}: {percentile_25_upper[obs_len]}\n")
        log_file.write("75th percentile MPJPE (upper body) for each observation length and each selected timestep:\n")
        for obs_len in range(percentile_75_upper.shape[0]):
            log_file.write(f"Obs {obs_len+1}: {percentile_75_upper[obs_len]}\n")

        log_file.write("25th percentile MPJPE (lower body) for each observation length and each selected timestep:\n")
        for obs_len in range(percentile_25_lower.shape[0]):
            log_file.write(f"Obs {obs_len+1}: {percentile_25_lower[obs_len]}\n")

        log_file.write("75th percentile MPJPE (lower body) for each observation length and each selected timestep:\n")
        for obs_len in range(percentile_75_lower.shape[0]):
            log_file.write(f"Obs {obs_len+1}: {percentile_75_lower[obs_len]}\n")

        log_file.write("\n")

print(f"MPJPE logs saved to {log_filename}")