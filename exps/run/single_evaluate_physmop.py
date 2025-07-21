from RealtimePhysMop import RealtimePhysMop
from dataset.action_aware_dataset import ActionAwareDataset
from torch.utils.data import DataLoader
from visualize_motion import visualize_motion_with_ground_truth, visualize_continuous_motion

from dataset.base_dataset_test import BaseDataset_test

import utils.config as config
from torchviz import make_dot


ds = "AMASS" 
if __name__ == "__main__":
    realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cuda')
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

    # Process first walking sample
    for i, batch in enumerate(data_loader):
        print(f"Processing walking sample {i}")
        # print(f"Action: {batch['action'][0]}")
        # print(f"File: {batch['file_path'][0]}")
        print("Batch keys:", batch.keys())
        print("Root joint:", batch['q'][:, :25, :3])
        
        model_output, batch_info = realtime_model.predict(batch)
        gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(
            model_output, batch_info, mode='test'
        )
        # print(gt_J[0])
        
        print(f"Prediction shape: {pred_J_data.shape}")
        visualize_continuous_motion(gt_J.cpu().detach(), title=f"Je moeder")
        
        # visualize_motion_with_ground_truth(pred_J_fusion.cpu().detach(), gt_J.cpu().detach())
        
        # Only process first sample for demo
        break
