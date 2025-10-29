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
import math

def front_to_back_frame_indices(positions, joint_idx=8, axis=0):
    # axis=0 assumes forward is along x; change if needed
    foot_traj = positions[:, joint_idx, axis]  # shape: (num_frames,)
    
    # Find zero-crossings (from positive to negative or vice versa)
    sign_changes = np.where(np.diff(np.sign(foot_traj)))[0]
    
    # Frame indices of transitions
    transition_frames = sign_changes
    
    # Intervals between transitions (in frames)
    intervals = np.diff(transition_frames)
    
    return transition_frames, intervals

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
    realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')
    data_loader = DataLoader(dataset=BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length, filter_str="treadmill_slow"),
                                batch_size=1,
                                shuffle=False,
                                num_workers=8)
    
    mpjpe_data_all = []
    mpjpe_physics_gt_all = []
    mpjpe_fusion_all = []
    # Process first walking sample
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing samples")):
        # print(f"Action: {batch['action'][0]}")
        # print(f"File: {batch['file_path'][0]}")
        # print("Batch keys:", batch.keys())
        # print("Root joint:", batch['q'][:, :25, :3])
        # print(f"Batch shape: {batch['q'].shape}")
        # print(f"Batch file paths: {batch['file_paths']}")
        num_files = len(batch['file_paths'])
        
        del batch['file_paths']

    #     # # Assume that the model always predicts the same number of frames, at the same frequency
    #     for i in range(num_files):

    #         normal_batch = {k: v[:, i*config.total_length:(i+1)*config.total_length, ...] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    #         # print("Normal batch shape:", normal_batch['q'].shape)
    #         model_output, batch_info = realtime_model.predict(normal_batch)
    #         gt_J, _, _, _ = realtime_model.model_output_to_3D_joints(model_output, batch_info, mode='test')

    #         # visualize_continuous_motion(gt_J, title="Ground Truth Motion")
    #         transition_frames, intervals = front_to_back_frame_indices(gt_J.numpy())
    #         # visualize_continuous_motion(gt_J.numpy(), title="Ground Truth Motion", skeleton_type="amass", save_gif_path="fast_output_gt.gif")
    #         # print("Transition frames:", transition_frames)
    #         # print("Intervals between transitions (in frames):", intervals)
    #         all_intervals.extend(intervals)
    #     # break
    # # Compute the average interval (in frames) across all samples
    # if all_intervals:
    #     avg_interval = np.mean(all_intervals)
    #     print(f"Average interval between transitions (frames): {avg_interval:.2f}")
    # else:
    #     print("No transitions found in any sample.")


        # downsample_rates = np.arange(3, 0.1, -0.2)  # From 3 to 0.1, step -0.2
        interval_fast = 15.59
        interval_slow = 20.91
        interval_normal = 17.16
        downsample_rate = interval_normal / interval_slow  # ≈ 1.10
        downsample_rates = [1]
        mpjpe_data_results = []
        mpjpe_physics_results = []
        mpjpe_fusion_results = []

        for downsample_rate in downsample_rates:
            processed_batch = {}
            # print("downsample_rate:", downsample_rate)
            input_len = config.hist_length
            total_len = config.total_length
            output_len = total_len - input_len

            if downsample_rate >= 1.0 or np.isclose(downsample_rate, 1.0):
                # Downsample only the input part, then take the next output_len frames at original rate
                for k, v in batch.items():
                    if not isinstance(v, torch.Tensor):
                        processed_batch[k] = v
                        continue

                    # 1) Take the original input history support on the native grid
                    src_input = v[:, :int(round(total_len * downsample_rate)), ...]  # [B, input_len, ...] chronological

                    # 2) Resample the input and output uniformly at factor alpha, anchored at the boundary
                    #    (no jitter, no duplicates; with anti-alias for alpha>1)
                    processed_batch[k] = resample_with_history_boundary_anchored(src_input, total_len, float(downsample_rate), antialias=True)

            
            elif downsample_rate < 1.0:
                # Upsample: interpolate to more frames
                upsample_factor = 1.0 / downsample_rate
                processed_batch = {}
                for k, v in batch.items():
                    if k == 'q' and isinstance(v, torch.Tensor):
                        v_spliced = v[:, :, ...]
                        orig_time_steps = v_spliced.shape[1]
                        new_time_steps = int(np.round(orig_time_steps * upsample_factor))
                        v_perm = v_spliced.permute(0, 2, 1)  # [batch, features, time]
                        v_upsampled = F.interpolate(v_perm, size=new_time_steps, mode='linear', align_corners=True)
                        v_upsampled = v_upsampled.permute(0, 2, 1)  # [batch, time, features]
                        processed_batch[k] = v_upsampled[:, :config.total_length, ...]
                    else:
                        processed_batch[k] = v[:, :config.total_length, ...]

            model_output, batch_info = realtime_model.predict(processed_batch)
            gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(model_output, batch_info, mode='test')

            # visualize_continuous_motion(gt_J, title="Ground Truth Motion", skeleton_type="amass", save_path="fast_output_gt.mp4")
            # visualize_motion_with_ground_truth(pred_J_data.cpu().detach().numpy(), gt_J.cpu().detach().numpy(), title="Predicted vs Ground Truth Motion (Data)",
            #                                     skeleton_type="amass", save_path="output_fast_data.mp4")
            # visualize_motion_with_ground_truth(pred_J_physics_gt.cpu().detach().numpy(), gt_J.cpu().detach().numpy(), title="Predicted vs Ground Truth Motion (Physics)",
            #                                     skeleton_type="amass", save_path="output_fast_physics.mp4")
            eval_results = realtime_model.evaluation_metrics(gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion)


            mpjpe_data = np.mean([eval_results['error_test_data_upper'][0], eval_results['error_test_data_lower'][0]], axis=0)
            mpjpe_data_results.append(mpjpe_data[selected_indices])  # shape: (4,)
            mpjpe_physics_gt = np.mean([eval_results['error_test_physics_gt_upper'][0], eval_results['error_test_physics_gt_lower'][0]], axis=0)
            mpjpe_physics_results.append(mpjpe_physics_gt[selected_indices])  # shape: (4,)
            mpjpe_fusion = np.mean([eval_results['error_test_fusion_upper'][0], eval_results['error_test_fusion_lower'][0]], axis=0)
            mpjpe_fusion_results.append(mpjpe_fusion[selected_indices])  # shape: (4,)
        mpjpe_data_all.append(np.array(mpjpe_data_results))  # shape: (num_downsample_rates, 4)
        mpjpe_physics_gt_all.append(np.array(mpjpe_physics_results))  # shape: (num_downsample_rates, 4)
        mpjpe_fusion_all.append(np.array(mpjpe_fusion_results))  # shape: (num_downsample_rates, 4)

    mpjpe_data_all = np.array(mpjpe_data_all)  # shape: (num_samples, num_downsample_rates, 4)
    mpjpe_data_all = np.mean(mpjpe_data_all, axis=0)  # shape: (num_downsample_rates, 4)
    mpjpe_physics_gt_all = np.array(mpjpe_physics_gt_all)  # shape: (num_samples, num_downsample_rates, 4)
    mpjpe_physics_gt_all = np.mean(mpjpe_physics_gt_all, axis=0)  # shape: (num_downsample_rates, 4)
    mpjpe_fusion_all = np.array(mpjpe_fusion_all)  # shape: (num_samples, num_downsample_rates, 4)
    mpjpe_fusion_all = np.mean(mpjpe_fusion_all, axis=0)  # shape: (num_downsample_rates, 4)

    print("MPJPE Data Branch:\n", mpjpe_data_all)
    print("MPJPE Physics Branch:\n", mpjpe_physics_gt_all)
    print("MPJPE Fusion Branch:\n", mpjpe_fusion_all)

    # mjpe_gcnext_all = np.loadtxt("mpjpe_data_fast.txt", delimiter=",")

    # branch_results = {
    #     "data": mpjpe_data_all,
    #     "physics_gt": mpjpe_physics_gt_all,
    #     "fusion": mpjpe_fusion_all,
    #     "gcnext": mjpe_gcnext_all
    # }
    # branch_titles = {
    #     "data": "MPJPE for the data branch vs Downsample Rate",
    #     "physics_gt": "MPJPE for the physics branch vs Downsample Rate",
    #     "fusion": "MPJPE for the fusion branch vs Downsample Rate",
    #     "gcnext": "MPJPE for the GCNext model vs Downsample Rate"
    # }
    # branch_filenames = {
    #     "data": "mpjpe_vs_downsample_rate_data.png",
    #     "physics_gt": "mpjpe_vs_downsample_rate_physics.png",
    #     "fusion": "mpjpe_vs_downsample_rate_fusion.png",
    #     "gcnext": "mpjpe_vs_downsample_rate_gcnext.png"
    # }

    # all_arrays = [
    # mpjpe_data_all,
    # mpjpe_physics_gt_all,
    # mpjpe_fusion_all,
    # mjpe_gcnext_all
    # ]

    # all_data = np.concatenate([arr.flatten() for arr in all_arrays if arr is not None])
    # y_min = np.min(all_data[all_data > 0])  # Avoid zero for log scale
    # y_max = np.percentile(all_data, 100)
    # y_limits = (y_min, y_max)

    # for branch, results in branch_results.items():
    #     plt.figure()
    #     for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    #         plt.plot(downsample_rates, results[:, i], marker='o', label=label)
    #     plt.xlabel('Downsample Rate')
    #     plt.ylabel('Mean MPJPE')
    #     plt.title(branch_titles[branch])
    #     plt.gca().invert_xaxis()
    #     plt.grid(True)
    #     plt.legend(title='Timesteps into the future')
    #     plt.ylim(y_limits)
    #     plt.savefig(branch_filenames[branch])
    #     plt.close()
