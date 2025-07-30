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

        print(f"Normal batch shape: {normal_batch['q'].shape}")

        downsample_rates = np.arange(3, 0.9, -0.2)  # From 3 to 1, step -0.2
        mpjpe_data_results = []
        mpjpe_physics_results = []
        mpjpe_fusion_results = []

        for downsample_rate in downsample_rates:
            start_index = config.pred_length + int(config.hist_length * downsample_rate)
            # Generate indices for downsampling
            num_hist_frames = config.hist_length
            hist_indices = np.round(np.arange(-start_index, -config.pred_length, downsample_rate)).astype(int)

            predicted_batch = {k: v[:, -config.pred_length:, ...] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            downsampled_input = {k: v[:, hist_indices, ...] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            print(f"Downsampled input shape: {downsampled_input['q'].shape}")
            combined_batch = {}
            for k in downsampled_input.keys():
                if isinstance(downsampled_input[k], torch.Tensor) and isinstance(predicted_batch[k], torch.Tensor):
                    # Concatenate along the frame/time dimension (usually dim=1)
                    combined_batch[k] = torch.cat([downsampled_input[k], predicted_batch[k]], dim=1)

            # assert that the last config.pred_length frames are the same between combined and normal batch
            assert torch.all(combined_batch['q'][:, -config.pred_length:, :] == normal_batch['q'][:, -config.pred_length:, :]), "Last frames do not match between combined and normal batch"

            model_output, batch_info = realtime_model.predict(combined_batch)
            gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(model_output, batch_info, mode='test')

            # visualize_continuous_motion(gt_J, title="Ground Truth Motion", skeleton_type="amass")
            # visualize_motion_with_ground_truth(pred_J_fusion.cpu().detach().numpy(), gt_J.cpu().detach().numpy(), title="Predicted vs Ground Truth Motion (Data)",
            #                                     skeleton_type="amass", save_gif_path="output.gif")
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

        break
    mpjpe_data_all = np.array(mpjpe_data_all)  # shape: (num_samples, num_downsample_rates, 4)
    mpjpe_data_all = np.mean(mpjpe_data_all, axis=0)  # shape: (num_downsample_rates, 4)
    mpjpe_physics_gt_all = np.array(mpjpe_physics_gt_all)  # shape: (num_samples, num_downsample_rates, 4)
    mpjpe_physics_gt_all = np.mean(mpjpe_physics_gt_all, axis=0)  # shape: (num_downsample_rates, 4)
    mpjpe_fusion_all = np.array(mpjpe_fusion_all)  # shape: (num_samples, num_downsample_rates, 4)
    mpjpe_fusion_all = np.mean(mpjpe_fusion_all, axis=0)  # shape: (num_downsample_rates, 4)

    plt.figure()
    for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
        plt.plot(downsample_rates, mpjpe_fusion_all[:, i], marker='o', label=label)
    plt.xlabel('Downsample Rate')
    plt.ylabel('Mean MPJPE')
    plt.title('MPJPE for the fusion branch vs Downsample Rate')
    plt.gca().invert_xaxis()  # Optional: show 3->1 left to right
    plt.grid(True)
    plt.legend(title='Timesteps into the future')
    plt.savefig('mpjpe_vs_downsample_rate_fusion.png')
    plt.show()
