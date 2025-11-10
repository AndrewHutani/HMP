from models.physmop.RealtimePhysMop import RealtimePhysMop
from torch.utils.data import DataLoader
from exps.visualize_motion import visualize_motion_with_ground_truth, visualize_continuous_motion

from dataset.physmop.full_sequence_dataset_test import BaseDataset_test

import models.physmop.utils.config as config
import numpy as np
import torch
from tqdm import tqdm


def get_specific_batch(data_loader, batch_index):
    """Get a specific batch by index without loading all batches into memory"""
    for i, batch in enumerate(data_loader):
        if i == batch_index:
            return batch
    return None

actions = [
    "walking", "eating", 
    "smoking", "discussion", "directions",
    "greeting", "phoning", "posing", "purchases", "sitting",
    "sittingdown", "takingphoto", "waiting", "walkingdog",
    "walkingtogether"
]
time_idx = [1, 3, 7, 9, 13, 17, 21, 24] # Corresponds to (idx+1)*40 ms in the future
selected_indices = [t + config.hist_length - 1 for t in time_idx]

ds = "AMASS" 
if __name__ == "__main__":
    realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')
    
    save_directory = "exps/B_input_length/performance_logs/"
    log_files = [save_directory + "physmop_data_mpjpe_log.txt", 
                 save_directory + "physmop_physics_mpjpe_log.txt", 
                 save_directory + "physmop_fusion_mpjpe_log.txt"]
    log_file_handles = [open(f, "w") for f in log_files]  # Open all files once in write mode
    data_loader = DataLoader(dataset=BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length),
                                batch_size=1,
                                shuffle=False,
                                num_workers=8)
    mpjpe_data_upper = []
    mpjpe_physics_gt_upper = []
    mpjpe_fusion_upper = []
    mpjpe_data_lower = []
    mpjpe_physics_gt_lower = []
    mpjpe_fusion_lower = []


    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            T = batch['q'].shape[1]
            mpjpe_data_per_obs_upper = []
            mpjpe_physics_gt_per_obs_upper = []
            mpjpe_fusion_per_obs_upper = []
            mpjpe_data_per_obs_lower = []
            mpjpe_physics_gt_per_obs_lower = []
            mpjpe_fusion_per_obs_lower = []

            for j in range(config.hist_length):
                # Trim input to last obs_len frames
                batch_trimmed = {k: v[:, :j+1 + config.pred_length, ...] if (isinstance(v, torch.Tensor) and v.shape[1] >= j+1 + config.pred_length)
                                else v
                                for k, v in batch.items()
                            }
                model_output, batch_info = realtime_model.predict(batch_trimmed)
                gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(model_output, batch_info, mode='test')
                # Compute MPJPE for this observation length (example: using pred_J_fusion)
                eval_results = realtime_model.evaluation_metrics(gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion)

                mpjpe_data_per_obs_upper.append(eval_results['error_test_data_upper'][0, selected_indices]) 
                mpjpe_physics_gt_per_obs_upper.append(eval_results['error_test_physics_gt_upper'][0, selected_indices])
                mpjpe_fusion_per_obs_upper.append(eval_results['error_test_fusion_upper'][0, selected_indices])
                mpjpe_data_per_obs_lower.append(eval_results['error_test_data_lower'][0, selected_indices])
                mpjpe_physics_gt_per_obs_lower.append(eval_results['error_test_physics_gt_lower'][0, selected_indices])
                mpjpe_fusion_per_obs_lower.append(eval_results['error_test_fusion_lower'][0, selected_indices])


            mpjpe_data_upper.append(mpjpe_data_per_obs_upper)  # shape: (obs_len, 8)
            mpjpe_physics_gt_upper.append(mpjpe_physics_gt_per_obs_upper)  # shape: (obs_len, 8)
            mpjpe_fusion_upper.append(mpjpe_fusion_per_obs_upper)  # shape: (obs_len, 8)
            mpjpe_data_lower.append(mpjpe_data_per_obs_lower)  # shape: (obs_len, 8)
            mpjpe_physics_gt_lower.append(mpjpe_physics_gt_per_obs_lower)  # shape: (obs_len, 8)
            mpjpe_fusion_lower.append(mpjpe_fusion_per_obs_lower)  # shape: (obs_len, 8)

    mpjpe_data_upper = np.array(mpjpe_data_upper)  # shape: (num_samples, obs_len, 8)
    mpjpe_physics_gt_upper = np.array(mpjpe_physics_gt_upper)  # shape: (num_samples, obs_len, 8)
    mpjpe_fusion_upper = np.array(mpjpe_fusion_upper)  # shape: (num_samples, obs_len, 8)
    mpjpe_data_lower = np.array(mpjpe_data_lower)  # shape: (num_samples, obs_len, 8)
    mpjpe_physics_gt_lower = np.array(mpjpe_physics_gt_lower)  # shape: (num_samples, obs_len, 8)
    mpjpe_fusion_lower = np.array(mpjpe_fusion_lower)  # shape: (num_samples, obs_len, 8)

    log_files = ["physmop_data_mpjpe_log.txt", "physmop_physics_mpjpe_log.txt", "physmop_fusion_mpjpe_log.txt"]
    for i, log_file in enumerate(log_file_handles):
        if i == 0:
            mpjpe_mean_upper = np.mean(mpjpe_data_upper, axis=0)
            mpjpe_std_upper = np.std(mpjpe_data_upper, axis=0)
            mpjpe_mean_lower = np.mean(mpjpe_data_lower, axis=0)
            mpjpe_std_lower = np.std(mpjpe_data_lower, axis=0)
            percentile_25_upper = np.percentile(mpjpe_data_upper, 25, axis=0)
            percentile_75_upper = np.percentile(mpjpe_data_upper, 75, axis=0)
            percentile_25_lower = np.percentile(mpjpe_data_lower, 25, axis=0)
            percentile_75_lower = np.percentile(mpjpe_data_lower, 75, axis=0)
        elif i == 1:
            mpjpe_mean_upper = np.mean(mpjpe_physics_gt_upper, axis=0)
            mpjpe_std_upper = np.std(mpjpe_physics_gt_upper, axis=0)
            mpjpe_mean_lower = np.mean(mpjpe_physics_gt_lower, axis=0)
            mpjpe_std_lower = np.std(mpjpe_physics_gt_lower, axis=0)
            percentile_25_upper = np.percentile(mpjpe_physics_gt_upper, 25, axis=0)
            percentile_75_upper = np.percentile(mpjpe_physics_gt_upper, 75, axis=0)
            percentile_25_lower = np.percentile(mpjpe_physics_gt_lower, 25, axis=0)
            percentile_75_lower = np.percentile(mpjpe_physics_gt_lower, 75, axis=0)

        else:
            mpjpe_mean_upper = np.mean(mpjpe_fusion_upper, axis=0)
            mpjpe_std_upper = np.std(mpjpe_fusion_upper, axis=0)
            mpjpe_mean_lower = np.mean(mpjpe_fusion_lower, axis=0)
            mpjpe_std_lower = np.std(mpjpe_fusion_lower, axis=0)
            percentile_25_upper = np.percentile(mpjpe_fusion_upper, 25, axis=0)
            percentile_75_upper = np.percentile(mpjpe_fusion_upper, 75, axis=0)
            percentile_25_lower = np.percentile(mpjpe_fusion_lower, 25, axis=0)
            percentile_75_lower = np.percentile(mpjpe_fusion_lower, 75, axis=0)

        # Write to log file
        log_file.write(f"Averaged MPJPE (upper body) for each observation length and each selected timestep:\n")
        for obs_len in range(mpjpe_mean_upper.shape[0]):
            log_file.write(f"Obs {obs_len+1}: [" + " ".join([f"{v:.6f}" for v in mpjpe_mean_upper[obs_len]]) + "]\n")
            log_file.write(f"Std {obs_len+1}: [" + " ".join([f"{v:.6f}" for v in mpjpe_std_upper[obs_len]]) + "]\n")
        log_file.write("\n")
        log_file.write(f"Averaged MPJPE (lower body) for each observation length and each selected timestep:\n")
        for obs_len in range(mpjpe_mean_lower.shape[0]):
            log_file.write(f"Obs {obs_len+1}: [" + " ".join([f"{v:.6f}" for v in mpjpe_mean_lower[obs_len]]) + "]\n")
            log_file.write(f"Std {obs_len+1}: [" + " ".join([f"{v:.6f}" for v in mpjpe_std_lower[obs_len]]) + "]\n")
        log_file.write("\n")

        log_file.write(f"25th percentile (upper body) for each observation length and each selected timestep:\n")
        for obs_len in range(percentile_25_upper.shape[0]):
            log_file.write(f"Obs {obs_len+1}: [" + " ".join([f"{v:.6f}" for v in percentile_25_upper[obs_len]]) + "]\n")
        log_file.write(f"75th percentile (upper body) for each observation length and each selected timestep:\n")
        for obs_len in range(percentile_75_upper.shape[0]):
            log_file.write(f"Obs {obs_len+1}: [" + " ".join([f"{v:.6f}" for v in percentile_75_upper[obs_len]]) + "]\n")
        
        log_file.write(f"25th percentile MPJPE (lower body) for each observation length and each selected timestep:\n")
        for obs_len in range(percentile_25_lower.shape[0]):
            log_file.write(f"Obs {obs_len+1}: [" + " ".join([f"{v:.6f}" for v in percentile_25_lower[obs_len]]) + "]\n")
        log_file.write(f"75th percentile MPJPE (lower body) for each observation length and each selected timestep:\n")
        for obs_len in range(percentile_75_lower.shape[0]):
            log_file.write(f"Obs {obs_len+1}: [" + " ".join([f"{v:.6f}" for v in percentile_75_lower[obs_len]]) + "]\n")

    # Close all log files at the end
    for log_file in log_file_handles:
        log_file.close()
