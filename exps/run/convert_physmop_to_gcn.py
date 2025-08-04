from RealtimePhysMop import RealtimePhysMop
from dataset.action_aware_dataset import ActionAwareDataset
from torch.utils.data import DataLoader
from visualize_motion import visualize_motion_with_ground_truth, visualize_continuous_motion

from dataset.full_sequence_dataset_test import BaseDataset_test

import utils.config as config
from torchviz import make_dot
from prediction_times import prediction_times
import time
import numpy as np

physmop_to_gcn = [2, 5, 8, 8, 1, 4, 7, 7, 3, 6, 9, 12, 13, 15, 17, 17, 17, 14, 16, 18, 18, 18]

ds = "AMASS" 
if __name__ == "__main__":
    all_gcn_gt_J_flat = []
    realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')
    # Option 2: Load only walking data
    # print("\n=== Loading walking data only ===")
    # walking_dataset = ActionAwareDataset(
    #     'data/data_processed/h36m_test_50.pkl',
    #     specific_action='walking'
    # )
    data_loader = DataLoader(dataset=BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length, filter_str="treadmill"),
                                batch_size=1,
                                shuffle=False,
                                num_workers=8)
    
    # walking_loader = DataLoader(walking_dataset, batch_size=1, shuffle=False)
    # print(f"Total batches in data_loader: {len(walking_loader)}")
    latency_times = []

    # Process first walking sample
    for batch_idx, batch in enumerate(data_loader):

        print(f"Processing sample {batch_idx}")
        # print(f"Action: {batch['action'][0]}")
        # print(f"File: {batch['file_path'][0]}")
        # print("Batch keys:", batch.keys())
        # print("Root joint:", batch['q'][:, :25, :3])
        print(f"Batch shape: {batch['q'].shape}")
        print(f"Batch file paths: {batch['file_paths']}")
        num_of_samples = len(batch['file_paths'])
        
        del batch['file_paths']
        # # Use a sliding window of some sort to feed the model the correct amount of data
        for i in range(num_of_samples):
            input_batch = {key: value[:,i*config.total_length:i*config.total_length+config.total_length] for key, value in batch.items()}
            print(f"Input batch shape: {input_batch['q'].shape}")

            model_output, batch_info = realtime_model.predict(input_batch)
            gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(
                model_output, batch_info, mode='test'
            )
            print(f"GT Joints shape: {gt_J.shape}")
            # visualize_continuous_motion(gt_J.detach().numpy(), "Ground Truth")
            gt_J = np.reshape(gt_J.detach().numpy(), (gt_J.shape[0], -1, 3))
            # the 0th index is the root joint, which is not used in GCN, and should be subtracted from all joints
            gt_J = gt_J - gt_J[:, 0, :][:, np.newaxis, :]       
            gcn_gt_J = gt_J[:, physmop_to_gcn, :]
            print(f"GCN GT Joints shape: {gcn_gt_J.shape}")
            # Flatten to 2D if needed (e.g., [frames, joints*3])
            frames = gcn_gt_J.shape[0]
            joints = gcn_gt_J.shape[1]
            gcn_gt_J_flat = gcn_gt_J.reshape(frames, joints * 3)  # assuming batch size 1
            all_gcn_gt_J_flat.append(gcn_gt_J_flat)

        # Save all GCN ground truth joints to a file
        all_gcn_gt_J_flat = np.concatenate(all_gcn_gt_J_flat, axis=0)
        np.savetxt("ordered_gt_J.txt", all_gcn_gt_J_flat, fmt="%.6f", delimiter=',')
        break
