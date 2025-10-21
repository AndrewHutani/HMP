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


time_idx = [2, 10, 14, 25]
selected_indices = [t + config.hist_length - 1 for t in time_idx]

ds = "AMASS" 
if __name__ == "__main__":
    # Load in performance logs mpjpe_physmop_data.txt, mpjpe_physmop_physics.txt, mpjpe_physmop_fusion.txt if they exist and store in array

    mpjpe_data_all = None
    mpjpe_physics_gt_all = None
    mpjpe_fusion_all = None

    if os.path.exists("mpjpe_physmop_data.txt"):
        mpjpe_data_all = np.loadtxt("mpjpe_physmop_data.txt", delimiter=",")
    if os.path.exists("mpjpe_physmop_physics.txt"):
        mpjpe_physics_gt_all = np.loadtxt("mpjpe_physmop_physics.txt", delimiter=",")
    if os.path.exists("mpjpe_physmop_fusion.txt"):
        mpjpe_fusion_all = np.loadtxt("mpjpe_physmop_fusion.txt", delimiter=",")
    
    # Check if all three arrays were loaded
    if mpjpe_data_all is not None and mpjpe_physics_gt_all is not None and mpjpe_fusion_all is not None:
        print("Performance logs found.")
        downsample_rates = np.arange(3, 0.1, -0.2)
    
    else:
        print("Performance logs not found. Running evaluation...")

        realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')
        # Option 2: Load only walking data
        # print("\n=== Loading walking data only ===")
        # walking_dataset = ActionAwareDataset(
        #     'data/data_processed/h36m_test_50.pkl',
        #     specific_action='walking'
        # )
        data_loader = DataLoader(dataset=BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length, filter_str="treadmill_norm"),
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
            
            del batch['file_paths']

            # Assume that the model always predicts the same number of frames, at the same frequency
            normal_batch = {k: v[:, -config.total_length:, ...] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


            downsample_rates = np.arange(3, 0.1, -0.2)  # From 3 to 0.1, step -0.2
            mpjpe_data_results = []
            mpjpe_physics_results = []
            mpjpe_fusion_results = []

            for downsample_rate in downsample_rates:
                processed_batch = {}

                input_len = config.hist_length
                total_len = config.total_length
                output_len = total_len - input_len

                if downsample_rate >= 1.0 or np.isclose(downsample_rate, 1.0):
                    # Downsample only the input part, then take the next output_len frames at original rate
                    # Compute how many source frames we need for the input
                    src_input_span = int(np.round(input_len * downsample_rate))

                    # choose a start index in the original batch time axis (here 0 -> start of sequence)
                    start_index = 0
                    input_src_end = start_index + src_input_span - 1

                    # build input indices (evenly spaced over the source span)
                    input_indices = np.round(np.linspace(start_index, input_src_end, input_len)).astype(int)

                    for k, v in batch.items():
                        if not isinstance(v, torch.Tensor):
                            processed_batch[k] = v
                            continue

                        # time axis is axis=1: [batch, time, ...]
                        max_t = v.shape[1] - 1
                        input_indices_clipped = np.clip(input_indices, 0, max_t)

                        # last input index (in source)
                        last_input_idx = int(input_indices_clipped[-1])
                        # output indices are the next output_len frames at the original sampling rate
                        output_indices = np.arange(last_input_idx + 1, last_input_idx + 1 + output_len)
                        output_indices = np.clip(output_indices, 0, max_t)

                        # gather and concatenate along time axis
                        # use torch.index_select for tensors
                        in_sel = torch.index_select(v, 1, torch.from_numpy(input_indices_clipped).to(v.device))
                        out_sel = torch.index_select(v, 1, torch.from_numpy(output_indices).to(v.device))
                        processed_batch[k] = torch.cat([in_sel, out_sel], dim=1)

                else:
                    # Upsample input: interpolate the input portion, then take following output frames
                    upsample_factor = 1.0 / downsample_rate
                    for k, v in batch.items():
                        if not isinstance(v, torch.Tensor):
                            processed_batch[k] = v
                            continue

                        # take the original input slice (first input_len frames of the available sequence)
                        src_input = v[:, :input_len, ...]  # [B, input_len, ...]
                        # prepare for interpolation: [B, features, time]
                        b, t = src_input.shape[0], src_input.shape[1]
                        rest_shape = src_input.shape[2:]
                        feat = int(np.prod(rest_shape))
                        src_perm = src_input.reshape(b, t, feat).permute(0, 2, 1)  # [B, feat, time]
                        new_time = int(np.round(t * upsample_factor))
                        # interpolate per-batch by looping (F.interpolate expects 3D or 4D; do each batch separately)
                        upsampled = []
                        for bi in range(b):
                            up = F.interpolate(src_perm[bi:bi+1], size=new_time, mode='linear', align_corners=True)
                            upsampled.append(up)
                        upsampled = torch.cat(upsampled, dim=0)  # [B, feat, new_time]
                        up_perm = upsampled.permute(0, 2, 1).reshape(b, new_time, *rest_shape)
                        # take last input_len frames of upsampled (or pad/truncate)
                        up_input = up_perm[:, -input_len:, ...]

                        # keep original output frames after the original input_end
                        original_last_input_idx = input_len - 1
                        output_indices = np.arange(original_last_input_idx + 1, original_last_input_idx + 1 + output_len)
                        output_indices = np.clip(output_indices, 0, v.shape[1] - 1)
                        out_sel = torch.index_select(v, 1, torch.from_numpy(output_indices).to(v.device))

                        processed_batch[k] = torch.cat([up_input, out_sel], dim=1)

                # now processed_batch contains input (resampled) + original-rate output for feeding to model
                model_output, batch_info = realtime_model.predict(processed_batch)
                gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(model_output, batch_info, mode='test')

                # visualize_continuous_motion(gt_J, title="Ground Truth Motion", skeleton_type="amass", save_gif_path="output_data_{}.gif".format(downsample_rate))
                # visualize_motion_with_ground_truth(pred_J_data.cpu().detach().numpy(), gt_J.cpu().detach().numpy(), title="Predicted vs Ground Truth Motion (Data)",
                #                                     skeleton_type="amass", save_gif_path="output_data_{}.gif".format(downsample_rate))
                # visualize_motion_with_ground_truth(pred_J_physics_gt.cpu().detach().numpy(), gt_J.cpu().detach().numpy(), title="Predicted vs Ground Truth Motion (Physics)",
                #                                     skeleton_type="amass", save_gif_path="output_physics_{}.gif".format(downsample_rate))
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
            # break  # Process only the first sample for this evaluation

        mpjpe_data_all = np.array(mpjpe_data_all)  # shape: (num_samples, num_downsample_rates, 4)
        mpjpe_data_all = np.mean(mpjpe_data_all, axis=0)  # shape: (num_downsample_rates, 4)
        mpjpe_physics_gt_all = np.array(mpjpe_physics_gt_all)  # shape: (num_samples, num_downsample_rates, 4)
        mpjpe_physics_gt_all = np.mean(mpjpe_physics_gt_all, axis=0)  # shape: (num_downsample_rates, 4)
        mpjpe_fusion_all = np.array(mpjpe_fusion_all)  # shape: (num_samples, num_downsample_rates, 4)
        mpjpe_fusion_all = np.mean(mpjpe_fusion_all, axis=0)  # shape: (num_downsample_rates, 4)

        # Save performance logs
        np.savetxt("mpjpe_physmop_data.txt", mpjpe_data_all, delimiter=",")
        np.savetxt("mpjpe_physmop_physics.txt", mpjpe_physics_gt_all, delimiter=",")
        np.savetxt("mpjpe_physmop_fusion.txt", mpjpe_fusion_all, delimiter=",")

    mjpe_gcnext_all = np.loadtxt("mpjpe_data_all.txt", delimiter=",")

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
        plt.xlabel('Resample Rate')
        plt.ylabel('Mean MPJPE')
        plt.title(branch_titles[branch])
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.legend(title='Prediction horizon')
        plt.ylim(y_limits)
        plt.savefig(branch_filenames[branch])
        plt.close()
