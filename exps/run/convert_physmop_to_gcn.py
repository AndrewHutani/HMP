import os
import re

import torch
from tqdm import tqdm
from RealtimePhysMop import RealtimePhysMop
from dataset.action_aware_dataset import ActionAwareDataset
from torch.utils.data import DataLoader
from visualize_motion import visualize_motion_with_ground_truth, visualize_continuous_motion

from dataset.full_sequence_dataset_test import BaseDataset_test

import utils.config as config
from prediction_times import prediction_times
import time
import numpy as np

physmop_to_gcn = [
    1, # LKNEE 0
    4, # LANKLE 1
    7, # LFOOT 2
    7, # LTOE 3
    2, # RKNEE 4
    5, # RANKLE 5
    8, # RFOOT 6
    8, # RTOE 7
    3, # Spine 8
    9, # Neck 9
    12, # Nose 10
    12, # Head 11
    14, # RShoulder 12
    16, # RElbow 13
    18, # Rwrist 14
    18, # RHand 15
    18, # RThumb 16
    13, # LShoulder 17
    15, # LElbow 18
    17, # Lwrist 19
    17, # LHand 20
    17, # LThumb 21
]

ds = "AMASS" 
if __name__ == "__main__":
    with torch.no_grad():
        realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')
        # Option 2: Load only walking data
        # print("\n=== Loading walking data only ===")
        # walking_dataset = ActionAwareDataset(
        #     'data/data_processed/h36m_test_50.pkl',
        #     specific_action='walking'
        # )
        data_loader = DataLoader(dataset=BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length, 
                                                        #   filter_str="treadmill_norm"
                                                        ),
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=8)
        latency_times = []

        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Batches"):
            try:
                all_gcn_gt_J_flat = []
                # print(f"Processing sample {batch_idx}")
                # print(f"Action: {batch['action'][0]}")
                # print(f"File: {batch['file_path'][0]}")
                # print("Batch keys:", batch.keys())
                # print("Root joint:", batch['q'][:, :25, :3])
                num_of_samples = len(batch['file_paths'])
                
                
                file_path = batch['file_paths'][0][0]
                file_path = os.path.basename(file_path)
                file_path = os.path.splitext(file_path)[0]
                file_path = file_path.rsplit('_', 1)[0]
                del batch['file_paths']
                # # Use a sliding window of some sort to feed the model the correct amount of data
                for i in range(num_of_samples):
                    input_batch = {key: value[:,i*config.total_length:i*config.total_length+config.total_length] for key, value in batch.items()}

                    model_output, batch_info = realtime_model.predict(input_batch)
                    gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(
                        model_output, batch_info, mode='test'
                    )
                    # visualize_continuous_motion(gt_J.detach().numpy(), "Ground Truth")
                    gt_J = np.reshape(gt_J.detach().numpy(), (gt_J.shape[0], -1, 3))
                    # the 0th index is the root joint, which is not used in GCN, and should be subtracted from all joints
                    gt_J = gt_J - gt_J[:, 0, :][:, np.newaxis, :]       
                    gcn_gt_J = gt_J[:, physmop_to_gcn, :]
                    # Flatten to 2D if needed (e.g., [frames, joints*3])
                    frames = gcn_gt_J.shape[0]
                    joints = gcn_gt_J.shape[1]
                    gcn_gt_J_flat = gcn_gt_J.reshape(frames, joints * 3)  # assuming batch size 1
                    # print(f"GCN Ground Truth Joints shape: {gcn_gt_J.shape}")
                    # visualize_continuous_motion(gcn_gt_J, "GCN Ground Truth Joints")
                    all_gcn_gt_J_flat.append(gcn_gt_J_flat)

                # Save all GCN ground truth joints to a file
                all_gcn_gt_J_flat = np.concatenate(all_gcn_gt_J_flat, axis=0)
                directory = "data/data_processed/physmop_to_gcn"

                np.savetxt(os.path.join(directory, file_path + ".txt"), all_gcn_gt_J_flat, fmt="%.6f", delimiter=',')
            except Exception as e:
                print(f"Error at batch {batch_idx}: {e}")
                break
    print("Processing complete.")