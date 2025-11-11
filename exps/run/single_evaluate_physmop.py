from exps.physmop_model.RealtimePhysMop import RealtimePhysMop
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
if __name__ == "__main__":
    realtime_model = RealtimePhysMop('ckpt/PhysMoP/hist_length_8.pt', device='cpu')
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

    latency_times = []
    all_prediction_times = []

    # Loop over all treadmill actions
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing treadmill actions")):
        if batch_idx >= 2:  # Limit to first batch for testing
            break
        # Remove file_paths for sliding window
        num_of_frames = batch['q'].shape[1]  # assuming 'q' is [batch, frames, ...]
        del batch['file_paths']
        window_size = config.total_length
        # Sliding window over the whole batch
        for start_idx in range(0, num_of_frames - window_size + 1):
            input_batch = {key: value[:, start_idx:start_idx+window_size] for key, value in batch.items()}

            t0 = time.perf_counter()
            model_output, batch_info = realtime_model.predict(input_batch)
            t1 = time.perf_counter()
            latency_times.append(t1 - t0)
            gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(
                model_output, batch_info, mode='test'
            )

        # Collect prediction times if available
        if prediction_times:
            all_prediction_times.extend(prediction_times)
    # Report average latency and prediction time
    avg_latency = sum(latency_times) / len(latency_times) * 1000 if latency_times else 0
    jitter = np.std(latency_times) * 1000 if latency_times else 0
    print(f"Average latency for all treadmill actions: {avg_latency:.2f} ms")
    print(f"Jitter in latency: {jitter:.2f} ms")
    if all_prediction_times:
        all_prediction_times = np.array(all_prediction_times)
        avg_prediction_time = np.mean(all_prediction_times) * 1000
        print(f"Average prediction time: {avg_prediction_time:.2f} ms")
        
