<<<<<<< Updated upstream
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

from config import config
from datasets.h36m_eval import H36MEval
import torch
from utils.misc import expmap2rotmat_torch, find_indices_256, find_indices_srnn, rotmat2xyz_torch

def visualize_continuous_motion(motion_sequence, title="Continuous Motion Visualization"):
    """
    Visualize a continuous motion sequence in 3D.

    :param motion_sequence: Numpy array of shape [num_frames, num_joints, 3] (motion sequence).
    :param title: Title of the plot.
    """
    axes_limit = 2
    # Define the connections between joints (skeleton structure)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (11, 12), (12, 13), (13, 14), (14, 15),
        (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31)
    ]

    # connections = [
    #     (0 + 1, 1 + 1), (1 + 1, 2 + 1), (2 + 1, 3 + 1), (3 + 1, 4 + 1), (4 + 1, 5 + 1),
    #     (0 + 1, 6 + 1), (6 + 1, 7 + 1), (7 + 1, 8 + 1), (8 + 1, 9 + 1), (9 + 1, 10 + 1),
    #     (11 + 1, 12 + 1), (12 + 1, 13 + 1), (13 + 1, 14 + 1), (14 + 1, 15 + 1),
    #     (16 + 1, 17 + 1), (17 + 1, 18 + 1), (18 + 1, 19 + 1), (19 + 1, 20 + 1), 
    #     (20 + 1, 21 + 1), (21 + 1, 22 + 1), (22 + 1, 23 + 1),
    #     (24 + 1, 25 + 1), (25 + 1, 26 + 1), (26 + 1, 27 + 1), (27 + 1, 28 + 1), 
    #     (28 + 1, 29 + 1), (29 + 1, 30 + 1), (30 + 1, 31 + 1)
    # ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame_idx in range(motion_sequence.shape[0]):
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title(f"Frame {frame_idx}: {motion_sequence[frame_idx, 0]}")

        joints = motion_sequence[frame_idx]
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c='r', marker='o')

         # Add joint indices as text annotations
        for joint_idx, (x, z, y) in enumerate(joints):
            ax.text(x, y, z, str(joint_idx), color='blue', fontsize=8)
        # for connection in connections:
        #     joint1, joint2 = connection
        #     ax.plot([joints[joint1, 0], joints[joint2, 0]],
        #             [joints[joint1, 2], joints[joint2, 2]],
        #             [joints[joint1, 1], joints[joint2, 1]], 'r', alpha=0.5)

        plt.pause(0.05)  # Adjust the pause duration for smoother animation

    plt.show()

def preprocess(filename):
    info = open(filename, 'r').readlines()
    pose_info = []
    for line in info:
        line = line.strip().split(',')
        if len(line) > 0:
            pose_info.append(np.array([float(x) for x in line]))
    pose_info = np.array(pose_info)
    pose_info = pose_info.reshape(-1, 33, 3)  # [num_frames, num_joints, 3]
    
    # Convert exponential maps to rotation matrices
    pose_info_flat = pose_info.reshape(-1, 3)  # Flatten for conversion
    rotation_matrices = expmap2rotmat_torch(torch.tensor(pose_info_flat).float())
    
    rotation_matrices = rotation_matrices.reshape(-1, 33, 3, 3)
    root_rotation = rotation_matrices[:, 0, :, :].clone().detach()
    print("Root rotation shape: ", root_rotation.shape)

    joint_positions = rotmat2xyz_torch(rotation_matrices[:, 1:, :, :])  # Convert to XYZ format
    print("Joint positions shape: ", joint_positions.shape)

    root_translation = pose_info[:, 0, :]


    rotated_joint_positions = torch.matmul(root_rotation.unsqueeze(1), joint_positions.unsqueeze(-1))
    rotated_joint_positions = rotated_joint_positions.squeeze(-1)  # Remove the last dimension
    print("Root position unsqueeze shape: ", torch.tensor(root_translation).float().unsqueeze(1).shape)
    rotated_joint_positions += torch.tensor(root_translation).float().unsqueeze(1)

    return rotated_joint_positions
        
# filename = '{0}/{1}/{2}_{3}.txt'.format(config.h36m_anno_dir, "S1", "walking", 1)
# motion_sequence = preprocess(filename)
=======
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

from config import config
from datasets.h36m_eval import H36MEval
import torch
from utils.misc import expmap2rotmat_torch, find_indices_256, find_indices_srnn, rotmat2xyz_torch

def visualize_continuous_motion(motion_sequence, title="Continuous Motion Visualization"):
    """
    Visualize a continuous motion sequence in 3D.

    :param motion_sequence: Numpy array of shape [num_frames, num_joints, 3] (motion sequence).
    :param title: Title of the plot.
    """
    axes_limit = 2
    # Define the connections between joints (skeleton structure)
    connections = [
        # Spine/Torso
        (0, 7),   # Hip to Spine
        (7, 8),   # Spine to Thorax  
        (8, 9),   # Thorax to Neck/Nose
        (9, 10),  # Neck to Head
        
        # Right leg
        (0, 1),   # Hip to RHip
        (1, 2),   # RHip to RKnee
        (2, 3),   # RKnee to RFoot
        
        # Left leg  
        (0, 4),   # Hip to LHip
        (4, 5),   # LHip to LKnee
        (5, 6),   # LKnee to LFoot
        
        # Right arm
        (8, 14),  # Thorax to RShoulder
        (14, 15), # RShoulder to RElbow
        (15, 16), # RElbow to RWrist
        
        # Left arm
        (8, 11),  # Thorax to LShoulder
        (11, 12), # LShoulder to LElbow
        (12, 13), # LElbow to LWrist
    ]

    # connections = [
    #     (0 + 1, 1 + 1), (1 + 1, 2 + 1), (2 + 1, 3 + 1), (3 + 1, 4 + 1), (4 + 1, 5 + 1),
    #     (0 + 1, 6 + 1), (6 + 1, 7 + 1), (7 + 1, 8 + 1), (8 + 1, 9 + 1), (9 + 1, 10 + 1),
    #     (11 + 1, 12 + 1), (12 + 1, 13 + 1), (13 + 1, 14 + 1), (14 + 1, 15 + 1),
    #     (16 + 1, 17 + 1), (17 + 1, 18 + 1), (18 + 1, 19 + 1), (19 + 1, 20 + 1), 
    #     (20 + 1, 21 + 1), (21 + 1, 22 + 1), (22 + 1, 23 + 1),
    #     (24 + 1, 25 + 1), (25 + 1, 26 + 1), (26 + 1, 27 + 1), (27 + 1, 28 + 1), 
    #     (28 + 1, 29 + 1), (29 + 1, 30 + 1), (30 + 1, 31 + 1)
    # ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame_idx in range(motion_sequence.shape[0]):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title(f"Frame {frame_idx}: {motion_sequence[frame_idx, 0]}")

        joints = motion_sequence[frame_idx]
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c='r', marker='o')

         # Add joint indices as text annotations
        for joint_idx, (x, z, y) in enumerate(joints):
            ax.text(x, y, z, str(joint_idx), color='blue', fontsize=8)
        for connection in connections:
            joint1, joint2 = connection
            ax.plot([joints[joint1, 0], joints[joint2, 0]],
                    [joints[joint1, 2], joints[joint2, 2]],
                    [joints[joint1, 1], joints[joint2, 1]], 'r', alpha=0.5)

        plt.pause(0.5)  # Adjust the pause duration for smoother animation

    plt.show()

def preprocess(filename):
    info = open(filename, 'r').readlines()
    pose_info = []
    for line in info:
        line = line.strip().split(',')
        if len(line) > 0:
            pose_info.append(np.array([float(x) for x in line]))
    pose_info = np.array(pose_info)
    pose_info = pose_info.reshape(-1, 33, 3)  # [num_frames, num_joints, 3]
    
    # Convert exponential maps to rotation matrices
    pose_info_flat = pose_info.reshape(-1, 3)  # Flatten for conversion
    rotation_matrices = expmap2rotmat_torch(torch.tensor(pose_info_flat).float())
    
    rotation_matrices = rotation_matrices.reshape(-1, 33, 3, 3)
    root_rotation = rotation_matrices[:, 0, :, :].clone().detach()
    print("Root rotation shape: ", root_rotation.shape)

    joint_positions = rotmat2xyz_torch(rotation_matrices[:, 1:, :, :])  # Convert to XYZ format
    print("Joint positions shape: ", joint_positions.shape)

    root_translation = pose_info[:, 0, :]


    rotated_joint_positions = torch.matmul(root_rotation.unsqueeze(1), joint_positions.unsqueeze(-1))
    rotated_joint_positions = rotated_joint_positions.squeeze(-1)  # Remove the last dimension
    print("Root position unsqueeze shape: ", torch.tensor(root_translation).float().unsqueeze(1).shape)
    rotated_joint_positions += torch.tensor(root_translation).float().unsqueeze(1)

    return rotated_joint_positions

def visualize_motion_with_ground_truth(predicted_positions, ground_truth_positions, title="Predicted vs Ground Truth Motion"):
    """
    Visualize the predicted motion and ground truth in 3D for specific time steps, with skeleton connections.

    :param predicted_positions: Tensor of shape [num_frames, num_joints, 3] (predicted motion)
    :param ground_truth_positions: Tensor of shape [num_frames, num_joints, 3] (ground truth motion)
    :param time_steps: List of time steps to visualize (e.g., [2, 10, 14, 25])
    :param title: Title of the plot
    """
    # Define the connections between joints
    connections = [
        # Spine/Torso
        (0, 7),   # Hip to Spine
        (7, 8),   # Spine to Thorax  
        (8, 9),   # Thorax to Neck/Nose
        (9, 10),  # Neck to Head
        
        # Right leg
        (0, 1),   # Hip to RHip
        (1, 2),   # RHip to RKnee
        (2, 3),   # RKnee to RFoot
        
        # Left leg  
        (0, 4),   # Hip to LHip
        (4, 5),   # LHip to LKnee
        (5, 6),   # LKnee to LFoot
        
        # Right arm
        (8, 14),  # Thorax to RShoulder
        (14, 15), # RShoulder to RElbow
        (15, 16), # RElbow to RWrist
        
        # Left arm
        (8, 11),  # Thorax to LShoulder
        (11, 12), # LShoulder to LElbow
        (12, 13), # LElbow to LWrist
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the default viewing angle
    ax.view_init(elev=20, azim=55)  # Adjust elevation and azimuth as needed
    ax.set_title(title)

    for frame_idx in range(len(predicted_positions)):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Update the title to include the current time step
        ax.set_title(f"{title} - Time Step #{frame_idx}")

        # Plot predicted joints for the specific frame
        predicted_joints = predicted_positions[frame_idx - 1]  # Subtract 1 because time_steps are 1-based
        ax.scatter(predicted_joints[:, 0], predicted_joints[:, 2], predicted_joints[:, 1], c='r', marker='o', label='Predicted')

        # Plot ground truth joints for the specific frame
        ground_truth_joints = ground_truth_positions[frame_idx - 1]
        ax.scatter(ground_truth_joints[:, 0], ground_truth_joints[:, 2], ground_truth_joints[:, 1], c='b', marker='^', label='Ground Truth')

        ax.legend()
        # Draw skeleton connections for predicted motion
        for connection in connections:
            joint1, joint2 = connection
            ax.plot([predicted_joints[joint1, 0], predicted_joints[joint2, 0]],
                    [predicted_joints[joint1, 2], predicted_joints[joint2, 2]],
                    [predicted_joints[joint1, 1], predicted_joints[joint2, 1]], 'r', alpha=0.5)
            ax.plot(
                [ground_truth_joints[joint1, 0], ground_truth_joints[joint2, 0]],
                [ground_truth_joints[joint1, 2], ground_truth_joints[joint2, 2]],
                [ground_truth_joints[joint1, 1], ground_truth_joints[joint2, 1]],
                c='b'
            )  

        plt.pause(0.5)  # Pause to display each frame

    plt.show()
        
# filename = '{0}/{1}/{2}_{3}.txt'.format(config.h36m_anno_dir, "S1", "walking", 1)
# motion_sequence = preprocess(filename)
>>>>>>> Stashed changes
# visualize_continuous_motion(motion_sequence/1000., title="Continuous Motion Visualization")