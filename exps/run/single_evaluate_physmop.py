from RealtimePhysMop import RealtimePhysMop
from dataset.action_aware_dataset import ActionAwareDataset
from torch.utils.data import DataLoader
from visualize_motion import visualize_motion_with_ground_truth, visualize_continuous_motion

from dataset.full_sequence_dataset_test import BaseDataset_test

import utils.config as config
from prediction_times import prediction_times
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn.functional as F
import os
import math

def _hann_lowpass_1d(x, cutoff_frac, k_min=5):
    """
    x: [B, T, F...] (time=dim1). cutoff_frac in (0,1], relative to Nyquist (1.0 == Nyquist).
    Simple Hann-windowed low-pass via depthwise 1D conv along time.
    """
    B, T = x.shape[0], x.shape[1]
    if T < 3:
        return x

    # Kernel length ~ 8/cutoff; keep odd
    k = max(k_min, int(math.ceil(8 / max(1e-6, cutoff_frac)))) | 1
    t = torch.arange(k, device=x.device, dtype=x.dtype)
    n = t - (k - 1) / 2

    # cutoff_frac is relative to Nyquist=0.5 cycles/sample -> normalized cutoff fc (cycles/sample)
    fc = 0.5 * cutoff_frac  # so cutoff_frac=1.0 means fc=0.5 (Nyquist)
    # sinc kernel (torch.sinc uses π-normalization: sinc(x)=sin(πx)/(πx))
    h = torch.sinc((2 * fc) * n)
    # Hann window
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * (t / (k - 1)))
    h = (h * w).to(x.dtype)
    h = h / h.sum()  # DC gain = 1

    pad = k // 2
    x_flat = x.reshape(B, T, -1).transpose(1, 2)  # [B, F, T]
    h = h.view(1, 1, k).to(x.device, x.dtype)
    y = F.conv1d(F.pad(x_flat, (pad, pad), mode='replicate'),
                 h.expand(x_flat.shape[1], 1, k),
                 groups=x_flat.shape[1])
    return y.transpose(1, 2).reshape_as(x)

def resample_with_history_boundary_anchored(x_hist, N, alpha, antialias = True):
    """
    Resample a sequence x_hist of length L to a new length N, keeping the last frame fixed.
    The resampling is anchored at the history boundary (last frame of x_hist).
    Args:
        x_hist: Input sequence of shape (L, D)
        N: Desired output length
        alpha: Resampling factor (N / L)
        antialias: Whether to apply anti-aliasing filter when downsampling
    """
    B, T_h = x_hist.shape[0], x_hist.shape[1]

    # Anti-alias if downsampling
    if antialias and alpha > 1.0 and T_h > 8:
        # cutoff relative to Nyquist: 1/alpha (<=1). Use a small safety margin.
        cutoff_frac = min(1.0 / alpha, 0.9)
        x_hist = _hann_lowpass_1d(x_hist, cutoff_frac)
    
    # Build target times anchored at the last frame
    t_last = T_h - 1
    t = torch.arange(N, device=x_hist.device, dtype=x_hist.dtype)
    t_target = t_last - t * float(alpha)                # newest->oldest
    # print("Resampling with alpha =", alpha)
    # print("t_target before clamp:", t_target)
    t_target = torch.clamp(t_target, 0, T_h - 1)

    t0 = torch.floor(t_target).long()
    t1 = torch.clamp(t0 + 1, max=T_h - 1)
    w = (t_target - t0.to(t_target.dtype)).view(1, N, *([1] * (x_hist.dim() - 2)))

    x0 = x_hist[:, t0, ...]
    x1 = x_hist[:, t1, ...]
    y  = (1 - w) * x0 + w * x1
    # reverse to chronological order (oldest -> newest)
    return torch.flip(y, dims=[1])


time_idx = [2, 10, 14, 25]
selected_indices = [t + config.hist_length - 1 for t in time_idx]

ds = "AMASS" 
if __name__ == "__main__":
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
            if batch_idx >= 2:
                break
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

                stride = 1
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
    np.savetxt("resampled_physmop_data_consistent_output_mean.csv", mpjpe_data_mean, delimiter=",", header=",".join(header))
    np.savetxt("resampled_physmop_physics_consistent_output_mean.csv", mpjpe_physics_mean, delimiter=",", header=",".join(header))
    np.savetxt("resampled_physmop_fusion_consistent_output_mean.csv", mpjpe_fusion_mean, delimiter=",", header=",".join(header))
    np.savetxt("resampled_physmop_data_consistent_output_std.csv", mpjpe_data_std, delimiter=",")
    np.savetxt("resampled_physmop_physics_consistent_output_std.csv", mpjpe_physics_std, delimiter=",")
    np.savetxt("resampled_physmop_fusion_consistent_output_std.csv", mpjpe_fusion_std, delimiter=",")


    mjpe_gcnext_all = np.loadtxt("mpjpe_gcnext.txt", delimiter=",")

    branch_results = {
        "data": mpjpe_data_all,
        "physics_gt": mpjpe_physics_gt_all,
        "fusion": mpjpe_fusion_all,
        "gcnext": mjpe_gcnext_all
    }
    branch_titles = {
        "data": "MPJPE for the data branch vs Downsample Rate",
        "physics_gt": "MPJPE for the physics branch vs Downsample Rate",
        "fusion": "MPJPE for the fusion branch vs Downsample Rate",
        "gcnext": "MPJPE for the GCNext model vs Downsample Rate"
    }
    branch_filenames = {
        "data": "mpjpe_vs_downsample_rate_data.png",
        "physics_gt": "mpjpe_vs_downsample_rate_physics.png",
        "fusion": "mpjpe_vs_downsample_rate_fusion.png",
        "gcnext": "mpjpe_vs_downsample_rate_gcnext.png"
    }

    all_arrays = [
    mpjpe_data_all,
    mpjpe_physics_gt_all,
    mpjpe_fusion_all,
    mjpe_gcnext_all
    ]

    all_data = np.concatenate([arr.flatten() for arr in all_arrays if arr is not None])
    y_min = np.min(all_data[all_data > 0])  # Avoid zero for log scale
    y_max = np.percentile(all_data, 100)
    y_limits = (y_min, y_max)

    for branch, results in branch_results.items():
        plt.figure()
        for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
            plt.plot(downsample_rates, results[:, i], marker='o', label=label)
        plt.xlabel(r'Resample rate $\alpha$')
        plt.ylabel(r'Mean MPJPE (mm)')
        plt.title(branch_titles[branch])
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.legend(title='Prediction horizon')
        plt.ylim(y_limits)
        plt.savefig(branch_filenames[branch])
        plt.close()
