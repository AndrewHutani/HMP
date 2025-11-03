from RealtimePhysMop import RealtimePhysMop
from dataset.action_aware_dataset import ActionAwareDataset
from torch.utils.data import DataLoader
from visualize_motion import visualize_input_and_output_physmop

from dataset.full_sequence_dataset_test import BaseDataset_test

import utils.config as config
from prediction_times import prediction_times
import time
import numpy as np

import os
import random


ds = "AMASS" 
actions = ["treadmill", "throwing", "circle", "sitting", "jumping", "kicking", "knocking"]
if __name__ == "__main__":
    realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')
    for action in actions:
        data_loader = DataLoader(dataset=BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length, filter_str=action),
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=8)
        os.makedirs(f"motion_examples/physmop/{action}", exist_ok=True)
        os.makedirs(f"motion_examples/physmop/{action}/physics_branch", exist_ok=True)
        os.makedirs(f"motion_examples/physmop/{action}/data_branch", exist_ok=True)

        print(f"Visualizing action: {action}")
        # Get random sample
        save_intervals = [0, 1, 2, 3, 4, 9, 14, 19, 24]

        # Iterate over the DataLoader and break after the first batch
        for batch_idx, batch in enumerate(data_loader):
            num_of_samples = len(batch['file_paths'])
            del batch['file_paths']

            random_sub_sample = random.randint(0, num_of_samples - 1)
            subsample = {key: value[:, random_sub_sample * config.total_length : (random_sub_sample + 1) * config.total_length] 
                        for key, value in batch.items()}

            save_intervals = [0, 1, 2, 3, 4, 9, 14, 19, 24]
            for i in save_intervals:
                input_batch = {key: value[:, :i + config.pred_length + 1] for key, value in subsample.items()}
                model_output, batch_info = realtime_model.predict(input_batch)
                gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(
                    model_output, batch_info, mode='test'
                )
                visualize_input_and_output_physmop(gt_J, pred_J_physics_gt, 
                                                   num_input_frames=i + 1,
                                                   num_predicted_frames=config.pred_length,
                                                   skeleton_type='amass',
                                                   save_mp4_path=f"motion_examples/physmop/{action}/physics_branch/{i+1}_input_frames.mp4")
                visualize_input_and_output_physmop(gt_J, pred_J_data, 
                                                   num_input_frames=i + 1,
                                                   num_predicted_frames=config.pred_length,
                                                   skeleton_type='amass',
                                                   save_mp4_path=f"motion_examples/physmop/{action}/data_branch/{i+1}_input_frames.mp4")
            break  # Only process one batch per action
        # break