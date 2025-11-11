from models.physmop.RealtimePhysMop import RealtimePhysMop
from torch.utils.data import DataLoader
from exps.visualize_motion import visualize_motion_with_ground_truth, visualize_continuous_motion

from dataset.physmop.full_sequence_dataset_test import BaseDataset_test

import models.physmop.utils.config as config
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn.functional as F
import os
import math

from exps.C_temporal_resolution.utils_resample_eval import resample_with_history_boundary_anchored_physmop as resample_with_history_boundary_anchored


save_directory = "exps/C_temporal_resolution/performance_logs/"
figure_directory = "exps/C_temporal_resolution/figures/"


time_idx = [2, 10, 14, 25]
selected_indices = [t + config.hist_length - 1 for t in time_idx]

ds = "AMASS" 
if __name__ == "__main__":

    mpjpe_data_all = None
    mpjpe_physics_gt_all = None
    mpjpe_fusion_all = None
    downsample_rates = np.arange(3, 0.1, -0.2)

    
    num_rates = len(downsample_rates)
    selected_indices = [t + config.hist_length - 1 for t in [2, 10, 14, 25]]

    # Aggregate MPJPEs for each rate (across all samples/windows)
    mpjpe_data_by_rate = [[] for _ in range(num_rates)]
    mpjpe_physics_by_rate = [[] for _ in range(num_rates)]
    mpjpe_fusion_by_rate = [[] for _ in range(num_rates)]
    preds_count_by_rate = [0 for _ in range(num_rates)]

    with torch.no_grad():
        realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')
        data_loader = DataLoader(
            dataset=BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length, filter_str="treadmill_norm"),
            batch_size=1,
            shuffle=False,
            num_workers=8
        )

        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing samples")):
            del batch['file_paths']

            for rate_idx, downsample_rate in enumerate(downsample_rates):
                input_len = config.hist_length
                total_len = config.total_length

                # Compute max_start_idx for sliding window
                if downsample_rate >= 1.0 or np.isclose(downsample_rate, 1.0):
                    max_end_index = int(round(total_len * downsample_rate))
                    max_start_idx = batch['q'].shape[1] - max_end_index
                    max_start_idx = max(0, max_start_idx)
                else:
                    max_start_idx = batch['q'].shape[1] - total_len
                    max_start_idx = max(0, max_start_idx)

                stride = 20
                for start_idx in range(0, max_start_idx + 1, stride):
                    processed_batch = {}
                    if downsample_rate >= 1.0 or np.isclose(downsample_rate, 1.0):
                        src_span = int(round(total_len * downsample_rate))
                        src_start = start_idx
                        src_end = src_start + src_span
                        for k, v in batch.items():
                            if not isinstance(v, torch.Tensor):
                                processed_batch[k] = v
                                continue
                            src_seq = v[:, src_start:src_end, ...]
                            processed_batch[k] = resample_with_history_boundary_anchored(
                                src_seq, total_len, float(downsample_rate), antialias=False
                            )
                    else:
                        for k, v in batch.items():
                            if not isinstance(v, torch.Tensor):
                                processed_batch[k] = v
                                continue
                            src_seq = v[:, start_idx : start_idx + total_len, ...]
                            upsample_factor = 1.0 / downsample_rate
                            b, t = src_seq.shape[0], src_seq.shape[1]
                            rest_shape = src_seq.shape[2:]
                            feat = int(np.prod(rest_shape))
                            src_perm = src_seq.reshape(b, t, feat).permute(0, 2, 1)
                            new_time = int(np.round(t * upsample_factor))
                            up = F.interpolate(src_perm, size=new_time, mode='linear', align_corners=True)
                            up_perm = up.permute(0, 2, 1).reshape(1, new_time, *rest_shape)
                            up_seq = up_perm[:, -total_len:, ...]
                            processed_batch[k] = up_seq

                    # Model prediction and metric aggregation
                    model_output, batch_info = realtime_model.predict(processed_batch)
                    gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(
                        model_output, batch_info, mode='test'
                    )
                    eval_results = realtime_model.evaluation_metrics(
                        gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion
                    )
                    mpjpe_data = np.mean([eval_results['error_test_data_upper'][0], eval_results['error_test_data_lower'][0]], axis=0)
                    mpjpe_physics = np.mean([eval_results['error_test_physics_gt_upper'][0], eval_results['error_test_physics_gt_lower'][0]], axis=0)
                    mpjpe_fusion = np.mean([eval_results['error_test_fusion_upper'][0], eval_results['error_test_fusion_lower'][0]], axis=0)

                    # Aggregate all selected indices for this rate
                    mpjpe_data_by_rate[rate_idx].append(mpjpe_data[selected_indices])
                    mpjpe_physics_by_rate[rate_idx].append(mpjpe_physics[selected_indices])
                    mpjpe_fusion_by_rate[rate_idx].append(mpjpe_fusion[selected_indices])
                    preds_count_by_rate[rate_idx] += 1

    # Compute mean and std for each rate
    mpjpe_data_mean = np.array([np.mean(mpjpe_data_by_rate[i], axis=0) for i in range(num_rates)])
    mpjpe_data_std  = np.array([np.std(mpjpe_data_by_rate[i], axis=0)  for i in range(num_rates)])
    mpjpe_physics_mean = np.array([np.mean(mpjpe_physics_by_rate[i], axis=0) for i in range(num_rates)])
    mpjpe_physics_std  = np.array([np.std(mpjpe_physics_by_rate[i], axis=0)  for i in range(num_rates)])
    mpjpe_fusion_mean = np.array([np.mean(mpjpe_fusion_by_rate[i], axis=0) for i in range(num_rates)])
    mpjpe_fusion_std  = np.array([np.std(mpjpe_fusion_by_rate[i], axis=0)  for i in range(num_rates)])

    # Save prediction counts per downsample rate
    mpjpe_counts_all = np.array(preds_count_by_rate)
    print("Predictions per downsample rate:", mpjpe_counts_all)

    # Save performance logs
    mpjpe_data_mean = np.column_stack((mpjpe_data_mean, mpjpe_counts_all))
    mpjpe_physics_mean = np.column_stack((mpjpe_physics_mean, mpjpe_counts_all))
    mpjpe_fusion_mean = np.column_stack((mpjpe_fusion_mean, mpjpe_counts_all))
    header = ["80ms", "400ms", "560ms", "1000ms", "n_predictions"]
    np.savetxt(save_directory+"resampled_physmop_data_matching_output_mean.csv", mpjpe_data_mean, delimiter=",", header=",".join(header))
    np.savetxt(save_directory+"resampled_physmop_physics_matching_output_mean.csv", mpjpe_physics_mean, delimiter=",", header=",".join(header))
    np.savetxt(save_directory+"resampled_physmop_fusion_matching_output_mean.csv", mpjpe_fusion_mean, delimiter=",", header=",".join(header))
    np.savetxt(save_directory+"resampled_physmop_data_matching_output_std.csv", mpjpe_data_std, delimiter=",")
    np.savetxt(save_directory+"resampled_physmop_physics_matching_output_std.csv", mpjpe_physics_std, delimiter=",")
    np.savetxt(save_directory+"resampled_physmop_fusion_matching_output_std.csv", mpjpe_fusion_std, delimiter=",")


