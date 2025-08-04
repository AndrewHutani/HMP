from RealtimePhysMop import RealtimePhysMop
from dataset.action_aware_dataset import ActionAwareDataset
from torch.utils.data import DataLoader
from visualize_motion import visualize_motion_with_ground_truth, visualize_continuous_motion

from dataset.base_dataset_test import BaseDataset_test

import utils.config as config
import numpy as np
from scipy.interpolate import interp1d
import torch

def rescale_joint_limits(q, joint_indices, new_min, new_max):
    # Extract the joint trajectory
    joint_vals = q[:, :, joint_indices]  # shape: (batch, frames, 3)
    # Find original min/max for normalization
    orig_min = joint_vals.min(dim=1, keepdim=True)[0]
    orig_max = joint_vals.max(dim=1, keepdim=True)[0]
    # Normalize to [0, 1]
    joint_norm = (joint_vals - orig_min) / (orig_max - orig_min + 1e-8)
    # Scale to new limits
    joint_scaled = joint_norm * (new_max - new_min) + new_min
    # Put back into q
    q[:, :, joint_indices] = joint_scaled
    return q

def repeat_joint_trajectory(q, target_length):
    """
    q: Tensor of shape (batch, frames, 63)
    target_length: int, desired number of frames
    Returns: Tensor of shape (batch, target_length, 63)
    """
    batch_size, seq_length, num_joints = q.shape
    num_repeats = int(np.ceil(target_length / seq_length))
    # Repeat along the time axis
    q_repeated = q.repeat(1, num_repeats, 1)  # shape: (batch, seq_length * num_repeats, 63)
    # Truncate to target_length
    q_repeated = q_repeated[:, :target_length, :]
    return q_repeated

def augment_motion(data):
    """
    data: dict
            Contains the following keys:
            - 'q': Tensor of shape (Batch size, Frames, 63 (# of joints * 3)) representing the joint angles
            - 'shape': Tensor of shape (Batch size, Frames, 10) representing SMPL shape parameters (?)
            - gender_id: Tensor of shape (Batch size, Frames) representing which gender to use
    """
    q = data['q'].clone()
    shape = data['shape']
    gender_id = data['gender_id']

    seq_length = q.shape[1]

    # Current limits of the angle between hip and knee joints (indices 6 and 9)
    current_min = q[:, :, [6, 9]].min(dim=1, keepdim=True)[0]
    current_max = q[:, :, [6, 9]].max(dim=1, keepdim=True)[0]
    print(f"Current limits for hip joints: min={current_min}, max={current_max}")

    # Try increasing the amplitude of angle between hip and knee joints along the first axis
    q = rescale_joint_limits(q, joint_indices=[6,9], new_min=-0.26, new_max=-0.04)


    # Speed augmentation, say we want to speed up the motion by 50%
    speed_factor = 1

    # Time warping (speed up or slow down)
    t_old = np.linspace(0, 1, seq_length)
    t_new = np.linspace(0, 1, int(seq_length / speed_factor))
    q_warped = np.zeros((1, len(t_new), 63))
    for i in range(63):
        interp_func = interp1d(t_old, q[:,:,i], kind='cubic', bounds_error=False, fill_value='extrapolate')
        q_warped[:,: ,i] = interp_func(t_new)

    # Convert the arrays to torch tensors
    q_warped = torch.from_numpy(q_warped)

    # Repeat the joint trajectories until the desired length
    q_warped = repeat_joint_trajectory(q_warped, target_length=config.total_length)

    # Shape parameters and gender id should be constant across the new sequence
    shape_new = np.repeat(shape[:, 0:1, :], q_warped.shape[1], axis=1)
    gender_id_new = np.repeat(gender_id[:, 0:1], q_warped.shape[1], axis=1)

    augmented_data = {
        'q': q_warped,
        'shape': shape_new,
        'gender_id': gender_id_new
    }

    return augmented_data


ds = "AMASS" 
time_idx = [1, 3, 7, 9, 13, 17, 21, 24] # Corresponds to idx*40 ms in the future
selected_indices = [t + config.hist_length - 1 for t in time_idx]
if __name__ == "__main__":
    realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')
    data_loader = DataLoader(dataset=BaseDataset_test(ds, config.DATASET_FOLDERS_TEST, config.hist_length, filter_str="treadmill_slow"),
                                batch_size=1,
                                shuffle=False,
                                num_workers=8)

    # Process first walking sample
    for batch_idx, batch in enumerate(data_loader):

        del batch['file_paths']
        raw_gait_cycle_data = {key: value[:, :50] for key, value in batch.items()}
        model_output, batch_info = realtime_model.predict(raw_gait_cycle_data)
        # unaugmented_J, _, _, _ = realtime_model.model_output_to_3D_joints(
        #         model_output, batch_info, mode='test'
        #     )
        augmented_data = augment_motion(raw_gait_cycle_data)
        model_output, batch_info = realtime_model.predict(augmented_data)
        augmented_J, pred_J_data, pred_J_physics, pred_J_fusion = realtime_model.model_output_to_3D_joints(
                model_output, batch_info, mode='test'
            )
        eval_results = realtime_model.evaluation_metrics(augmented_J, pred_J_data, pred_J_physics, pred_J_fusion)

        print(augmented_J.shape)
        visualize_continuous_motion(
            augmented_J.detach().cpu().numpy(), 
            title=f"Augmented Motion Visualization - Sample {batch_idx+1}",
        )

        # # Only use the error keys and not the accel
        # for key, value in eval_results.items():
        #     if 'error' in key:
        #         print(f"{key}: {value[0, selected_indices]}, shape: {value.shape}")
        # visualize_motion_with_ground_truth(
        #     pred_J_fusion.detach().cpu().numpy(), augmented_J.detach().cpu().numpy(), 
        # )

        break
