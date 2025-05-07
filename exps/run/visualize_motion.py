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
        ax.set_zlim([0, axes_limit])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title(f"Frame {frame_idx}: {motion_sequence[frame_idx, 0]}")

        joints = motion_sequence[frame_idx]
        # ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', marker='o')

        #  # Add joint indices as text annotations
        # for joint_idx, (x, y, z) in enumerate(joints):
        #     ax.text(x, y, z, str(joint_idx), color='blue', fontsize=8)
        for connection in connections:
            joint1, joint2 = connection
            ax.plot([joints[joint1, 0], joints[joint2, 0]],
                    [joints[joint1, 2], joints[joint2, 2]],
                    [joints[joint1, 1], joints[joint2, 1]], 'r', alpha=0.5)

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
        
filename = '{0}/{1}/{2}_{3}.txt'.format(config.h36m_anno_dir, "S1", "walking", 1)
motion_sequence = preprocess(filename)
visualize_continuous_motion(motion_sequence/1000., title="Continuous Motion Visualization")