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

import torch.nn.functional as F


time_idx = [1, 3, 13, 24] # Corresponds to idx*40 ms in the future
selected_indices = [t + config.hist_length - 1 for t in time_idx]

ds = "AMASS" 
if __name__ == "__main__":
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
    for batch_idx, batch in enumerate(data_loader):
        print(f"Processing sample {batch_idx}")
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

            if downsample_rate >= 1.0 or np.isclose(downsample_rate, 1.0):
                # Downsample: select frames at intervals
                start_index = config.pred_length + int(config.hist_length * downsample_rate)
                # Generate indices for downsampling
                num_hist_frames = config.hist_length
                hist_indices = np.round(np.arange(-start_index, -config.pred_length, downsample_rate)).astype(int)
                processed_batch = {k: v[:, hist_indices, ...] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            elif downsample_rate < 1.0:
                # Upsample: interpolate to more frames
                upsample_factor = 1.0 / downsample_rate
                processed_batch = {}
                for k, v in batch.items():
                    if k == 'q' and isinstance(v, torch.Tensor):
                        v_spliced = v[:, :-config.pred_length, ...]
                        orig_time_steps = v_spliced.shape[1]
                        new_time_steps = int(np.round(orig_time_steps * upsample_factor))
                        v_perm = v_spliced.permute(0, 2, 1)  # [batch, features, time]
                        v_upsampled = F.interpolate(v_perm, size=new_time_steps, mode='linear', align_corners=True)
                        v_upsampled = v_upsampled.permute(0, 2, 1)  # [batch, time, features]
                        processed_batch[k] = v_upsampled[:, -config.hist_length:, ...]
                    else:
                        processed_batch[k] = v[:, -config.hist_length:, ...]

            combined_batch = {}
            predicted_batch = {k: v[:, -config.pred_length:, ...] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            for k in processed_batch.keys():
                if isinstance(processed_batch[k], torch.Tensor) and isinstance(predicted_batch[k], torch.Tensor):
                    # Concatenate along the frame/time dimension (usually dim=1)
                    combined_batch[k] = torch.cat([processed_batch[k], predicted_batch[k]], dim=1)

            # assert that the last config.pred_length frames are the same between combined and normal batch
            assert torch.all(combined_batch['q'][:, -config.pred_length:, :] == normal_batch['q'][:, -config.pred_length:, :]), "Last frames do not match between combined and normal batch"

            model_output, batch_info = realtime_model.predict(combined_batch)
            gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(model_output, batch_info, mode='test')

            # visualize_continuous_motion(gt_J, title="Ground Truth Motion", skeleton_type="amass")
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

    mpjpe_data_all = np.array(mpjpe_data_all)  # shape: (num_samples, num_downsample_rates, 4)
    mpjpe_data_all = np.mean(mpjpe_data_all, axis=0)  # shape: (num_downsample_rates, 4)
    mpjpe_physics_gt_all = np.array(mpjpe_physics_gt_all)  # shape: (num_samples, num_downsample_rates, 4)
    mpjpe_physics_gt_all = np.mean(mpjpe_physics_gt_all, axis=0)  # shape: (num_downsample_rates, 4)
    mpjpe_fusion_all = np.array(mpjpe_fusion_all)  # shape: (num_samples, num_downsample_rates, 4)
    mpjpe_fusion_all = np.mean(mpjpe_fusion_all, axis=0)  # shape: (num_downsample_rates, 4)

    branch_results = {
        "data": mpjpe_data_all,
        "physics_gt": mpjpe_physics_gt_all,
        "fusion": mpjpe_fusion_all
    }
    branch_titles = {
        "data": "MPJPE for the data branch vs Downsample Rate",
        "physics_gt": "MPJPE for the physics branch vs Downsample Rate",
        "fusion": "MPJPE for the fusion branch vs Downsample Rate"
    }
    branch_filenames = {
        "data": "mpjpe_vs_downsample_rate_data.png",
        "physics_gt": "mpjpe_vs_downsample_rate_physics.png",
        "fusion": "mpjpe_vs_downsample_rate_fusion.png"
    }

    all_arrays = [
    mpjpe_data_all,
    mpjpe_physics_gt_all,
    mpjpe_fusion_all
    ]

    all_data = np.concatenate([arr.flatten() for arr in all_arrays if arr is not None])
    y_min = np.min(all_data[all_data > 0])  # Avoid zero for log scale
    y_max = np.percentile(all_data, 100)
    y_limits = (y_min, y_max)

    for branch, results in branch_results.items():
        plt.figure()
        for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
            plt.plot(downsample_rates, results[:, i], marker='o', label=label)
        plt.xlabel('Downsample Rate')
        plt.ylabel('Mean MPJPE')
        plt.title(branch_titles[branch])
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.legend(title='Timesteps into the future')
        plt.ylim(y_limits)
        plt.savefig(branch_filenames[branch])
        plt.close()
