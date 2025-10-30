import os
from RealtimePhysMop import RealtimePhysMop
from dataset.action_aware_dataset import ActionAwareDataset
from torch.utils.data import DataLoader
from visualize_motion import visualize_motion_with_ground_truth, visualize_continuous_motion

from dataset.full_sequence_dataset_test import BaseDataset_test

import utils.config as config
from prediction_times import prediction_times
import time
import numpy as np

from tqdm import tqdm



ds = "AMASS" 
output_dir = "exported_gt_joints"
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')
    # Option 2: Load only walking data
    # print("\n=== Loading walking data only ===")
    # walking_dataset = ActionAwareDataset(
    #     'data/data_processed/h36m_test_50.pkl',
    #     specific_action='walking'
    # )
    data_loader = DataLoader(dataset=BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length),
                                batch_size=1,
                                shuffle=False,
                                num_workers=8)
    
    # walking_loader = DataLoader(walking_dataset, batch_size=1, shuffle=False)
    # print(f"Total batches in data_loader: {len(walking_loader)}")
    latency_times = []

    # Process first walking sample
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
        full_sequence_gt = []
        # print(f"Processing sample {batch_idx}")
        # print(f"Action: {batch['action'][0]}")
        # print(f"File: {batch['file_path'][0]}")
        # print("Batch keys:", batch.keys())
        # print("Root joint:", batch['q'][:, :25, :3])
        # print(f"Batch shape: {batch['q'].shape}")
        # print(f"Batch file paths: {batch['file_paths'][0]}")
        num_of_samples = len(batch['file_paths'])
        key = os.path.basename(batch['file_paths'][0][0]).rsplit('_', 1)[0] + '.txt'
        # print(key)
        del batch['file_paths']
        # # Use a sliding window of some sort to feed the model the correct amount of data
        for i in range(num_of_samples):
            input_batch = {key: value[:,i*config.total_length:i*config.total_length+config.total_length] for key, value in batch.items()}
            # print(f"Input batch shape: {input_batch['q'].shape}")

            # t0 = time.perf_counter()
            model_output, batch_info = realtime_model.predict(input_batch)
            # t1 = time.perf_counter()
            # latency_times.append(t1 - t0)
            gt_J, _, _, _ = realtime_model.model_output_to_3D_joints(
                model_output, batch_info, mode='test'
            )
            # visualize_continuous_motion(gt_J)
            full_sequence_gt.append(gt_J)
        full_sequence_gt = np.concatenate(full_sequence_gt, axis=0)
        # print(f"Full sequence GT shape: {full_sequence_gt.shape}")
        np.savetxt(os.path.join(output_dir, key), full_sequence_gt.reshape(full_sequence_gt.shape[0], -1), delimiter=',')
        # break
        

        # avg_latency = sum(latency_times) / len(latency_times)
        # print(f"Average latency for processing {batch_idx + 1} samples: {avg_latency:.4f} seconds")
        # jitter = np.std(latency_times)
        # print(f"Jitter in latency: {jitter:.6f} seconds")
        # if prediction_times:
        #     prediction_times = np.array(prediction_times)
        #     avg_prediction_times = np.mean(prediction_times, axis=0)
        #     print(f"Average prediction times: {avg_prediction_times}")
        # break
