from torch.utils.data import DataLoader, Subset
from dataset.base_dataset_test import BaseDataset_test
import time
import numpy as np
from tqdm import tqdm
from RealtimePhysMop import RealtimePhysMop
import utils.config as config
import torch
from dataset.action_aware_dataset import ActionAwareDataset

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
time_idx = [1, 3, 7, 9, 13, 17, 21, 24] # Corresponds to idx*40 ms in the future
selected_indices = [t + config.hist_length - 1 for t in time_idx]

ds = "H36M" 
if __name__ == "__main__":
    realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')

    log_files = ["physmop_data_mpjpe_log.txt", "physmop_physics_mpjpe_log.txt", "physmop_fusion_mpjpe_log.txt"]
    log_file_handles = [open(f, "w") for f in log_files]  # Open all files once in write mode

    for action in actions:
        print(f"Evaluating action: {action}")
        dataset = ActionAwareDataset(
            'data/data_processed/h36m_test_50.pkl',
            specific_action=action
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        mpjpe_data_all = []
        mpjpe_physics_gt_all = []
        mpjpe_fusion_all = []
        accel_data_all = []
        accel_physics_gt_all = []
        accel_fusion_all = []

        for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f"{action} samples"):
            with torch.no_grad():
                # Assume batch['q'] shape: (1, T, 3), T = total frames
                T = batch['q'].shape[1]
                mpjpe_data_per_obs = []
                mpjpe_physics_gt_per_obs = []
                mpjpe_fusion_per_obs = []
                accel_data_per_obs = []
                accel_physics_gt_per_obs = []
                accel_fusion_per_obs = []
                for i in range(config.hist_length):
                    start_idx = 24 - i
                    # Trim input to last obs_len frames
                    batch_trimmed = {k: v[:, start_idx:config.total_length, ...] if isinstance(v, torch.Tensor) and v.shape[1] == config.total_length else v for k, v in batch.items()}
                    model_output, batch_info = realtime_model.predict(batch_trimmed)
                    gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(model_output, batch_info, mode='test')
                    # Compute MPJPE for this observation length (example: using pred_J_fusion)
                    error_test_data, error_test_physics_gt, error_test_fusion, accel_data, accel_physics_gt, accel_fusion = realtime_model.evaluation_metrics(gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion)

                    mpjpe_data_per_obs.append(error_test_data[0, selected_indices]) 
                    mpjpe_physics_gt_per_obs.append(error_test_physics_gt[0, selected_indices])
                    mpjpe_fusion_per_obs.append(error_test_fusion[0, selected_indices])
                    # accel_data_per_obs.append(accel_data[0, selected_indices])
                    # accel_physics_gt_per_obs.append(accel_physics_gt[0, selected_indices])
                    # accel_fusion_per_obs.append(accel_fusion[0, selected_indices])

                mpjpe_data_all.append(mpjpe_data_per_obs) 
                mpjpe_physics_gt_all.append(mpjpe_physics_gt_per_obs)
                mpjpe_fusion_all.append(mpjpe_fusion_per_obs)
                # accel_data_all.append(accel_data_per_obs)
                # accel_physics_gt_all.append(accel_physics_gt_per_obs)
                # accel_fusion_all.append(accel_fusion_per_obs)

        mpjpe_data_all = np.array(mpjpe_data_all)  # shape: (num_samples, obs_len, 8)
        mpjpe_physics_gt_all = np.array(mpjpe_physics_gt_all)  # shape: (num_samples, obs_len, 8)
        mpjpe_fusion_all = np.array(mpjpe_fusion_all)  # shape: (num_samples, obs_len, 8)

#         mpjpe_mean = np.mean(mpjpe_all_samples, axis=0)  # shape: (obs_len,)
        log_files = ["physmop_data_mpjpe_log.txt", "physmop_physics_mpjpe_log.txt", "physmop_fusion_mpjpe_log.txt"]
        for i, log_file in enumerate(log_file_handles):
            if i == 0:
                mjpe_mean = np.mean(mpjpe_data_all, axis=0) # shape: (obs_len, 8)
            elif i == 1:
                mjpe_mean = np.mean(mpjpe_physics_gt_all, axis=0)  # shape: (obs_len, 8)
            else:
                mjpe_mean = np.mean(mpjpe_fusion_all, axis=0)  # shape: (obs_len, 8)

            # Write to log file

            log_file.write(f"Averaged MPJPE for each observation length and each selected timestep: {action}\n")
            for obs_len in range(mjpe_mean.shape[0]):
                log_file.write(f"Obs {obs_len+1}: [" + " ".join([f"{v:.6f}" for v in mjpe_mean[obs_len]]) + "]\n")
            log_file.write("\n")

    # Close all log files at the end
    for log_file in log_file_handles:
        log_file.close()
