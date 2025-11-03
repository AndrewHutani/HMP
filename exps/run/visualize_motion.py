import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

from config import config
from datasets.h36m_eval import H36MEval
import torch
from utils.misc import expmap2rotmat_torch, find_indices_256, find_indices_srnn, rotmat2xyz_torch
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter 

# Define the connections between joints (skeleton structure)
amass_connections = [
    # Spine/Torso/Head
    (0, 3), (3,6), (6, 9), (9, 12),
    #Left leg
    (0, 1), (1, 4), (4, 7),
    # Right leg
    (0, 2), (2, 5), (5, 8),
    # Left arm
    (6, 10), (10, 13), (13, 15), (15, 17),
    # Right arm
    (6, 11), (11, 14), (14, 16), (16, 18)
]

h36m_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (11, 12), (12, 13), (13, 14), (14, 15),
    (16, 17), (17, 18), (18, 19), (19, 20), 
    (20, 21), (21, 22), (22, 23),
    (24, 25), (25, 26), (26, 27), (27, 28), 
    (28, 29), (29, 30), (30, 31)
]

incomplete_h36m_connections = [
    (-1, 0), (-1, 4), (-1, 8),
    (0, 1), (1, 2), (2, 3), 
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (14, 15),
    (10, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    (10, 17), (17, 18), (18, 19), (19, 20), (20, 21)
]


def visualize_continuous_motion(motion_sequence, title="Continuous Motion Visualization",
                                skeleton_type = None,
                                save_mp4_path = None,
                                show_axes = True,
                                skeleton_color = 'b'):
    """
    Save a continuous motion sequence in 3D.

    :param motion_sequence: Numpy array of shape [num_frames, num_joints, 3] (motion sequence).
    :param title: Title of the plot.
    """
    axes_limit = 2

    if skeleton_type == 'h36m':
        connections = h36m_connections
        motion_sequence = np.stack((motion_sequence[:, :, 2],
                                       motion_sequence[:, :, 0],
                                       motion_sequence[:, :, 1]), axis=2)
    elif skeleton_type == 'amass':
        connections = amass_connections
    elif skeleton_type == 'incomplete_h36m':
        connections = incomplete_h36m_connections
        motion_sequence = np.stack((motion_sequence[:, :, 2],
                                       motion_sequence[:, :, 0],
                                       motion_sequence[:, :, 1]), axis=2)   

    fig = plt.figure(figsize=(19, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=20, azim=55)  # Adjust elevation and azimuth as needed

    def update(frame_idx):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Frame {frame_idx}: {motion_sequence[frame_idx, 0]}")
        ax.grid(False)

        # Optional: Remove axis panes (background planes)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Optional: Make pane edges transparent
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        # Optional: Remove axis lines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        joints = motion_sequence[frame_idx]

        # Add artificial origin as the last joint
        joints_with_origin = np.vstack([joints, np.array([0, 0, 0])])

        if skeleton_type is not None:
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints_with_origin[joint1, 0], joints_with_origin[joint2, 0]],
                        [joints_with_origin[joint1, 1], joints_with_origin[joint2, 1]],
                        [joints_with_origin[joint1, 2], joints_with_origin[joint2, 2]], skeleton_color, alpha=1)
        else:
            for joint_idx, (x, y, z) in enumerate(joints):
                ax.text(x, y, z, str(joint_idx), color='blue', fontsize=8)

    ani = animation.FuncAnimation(fig, update, frames=motion_sequence.shape[0], interval=100)

    if save_mp4_path:
        ani.save(save_mp4_path, writer=FFMpegWriter(fps=25))
    else:
        while True:
            plt.show()


def visualize_motion_with_ground_truth(predicted_positions, ground_truth_positions, 
                                       title="Predicted vs Ground Truth Motion",
                                       skeleton_type=None,
                                       save_mp4_path=None,
                                       show_axes = True):
    """
    Visualize the predicted motion and ground truth in 3D for specific time steps, with skeleton connections.

    :param predicted_positions: Tensor of shape [num_frames, num_joints, 3] (predicted motion)
    :param ground_truth_positions: Tensor of shape [num_frames, num_joints, 3] (ground truth motion)
    :param time_steps: List of time steps to visualize (e.g., [2, 10, 14, 25])
    :param title: Title of the plot
    """

    if skeleton_type == 'h36m':
        connections = h36m_connections
        predicted_positions = np.stack((predicted_positions[:, :, 2],
                                       predicted_positions[:, :, 0],
                                       predicted_positions[:, :, 1]), axis=2)
        ground_truth_positions = np.stack((ground_truth_positions[:, :, 2],
                                           ground_truth_positions[:, :, 0],
                                           ground_truth_positions[:, :, 1]), axis=2)
    elif skeleton_type == 'amass':
        connections = amass_connections
    elif skeleton_type == 'incomplete_h36m':
        connections = incomplete_h36m_connections
        predicted_positions = np.stack((predicted_positions[:, :, 2],
                                       predicted_positions[:, :, 0],
                                       predicted_positions[:, :, 1]), axis=2)
        ground_truth_positions = np.stack((ground_truth_positions[:, :, 2],
                                           ground_truth_positions[:, :, 0],
                                           ground_truth_positions[:, :, 1]), axis=2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the default viewing angle
    ax.view_init(elev=20, azim=55)  # Adjust elevation and azimuth as needed
    ax.set_title(title)

    def update(frame_idx):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Frame {frame_idx}")

        if show_axes:
            ax.grid(True)
            ax.set_xticks(np.linspace(-1, 1, 5))
            ax.set_yticks(np.linspace(-1, 1, 5))
            ax.set_zticks(np.linspace(-1, 1, 5))
        else:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Remove axis panes (background planes)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # Optional: Make pane edges transparent
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')
        
        predicted_joints = predicted_positions[frame_idx]
        # Add artificial origin as the last joint
        predicted_joints_with_origin = np.vstack([predicted_joints, np.array([0, 0, 0])])
        ax.scatter(predicted_joints_with_origin[:, 0], predicted_joints_with_origin[:, 1], predicted_joints_with_origin[:, 2], c='r', marker='o', label='Predicted')

        # Plot ground truth joints for the specific frame
        ground_truth_joints = ground_truth_positions[frame_idx]
        # Add artificial origin as the last joint
        ground_truth_joints_with_origin = np.vstack([ground_truth_joints, np.array([0, 0, 0])])
        ax.scatter(ground_truth_joints_with_origin[:, 0], ground_truth_joints_with_origin[:, 1], ground_truth_joints_with_origin[:, 2], c='b', marker='^', label='Ground Truth')

        if skeleton_type is not None:
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([predicted_joints_with_origin[joint1, 0], predicted_joints_with_origin[joint2, 0]],
                        [predicted_joints_with_origin[joint1, 1], predicted_joints_with_origin[joint2, 1]],
                        [predicted_joints_with_origin[joint1, 2], predicted_joints_with_origin[joint2, 2]], 'r', alpha=0.5)
                ax.plot(
                    [ground_truth_joints_with_origin[joint1, 0], ground_truth_joints_with_origin[joint2, 0]],
                    [ground_truth_joints_with_origin[joint1, 1], ground_truth_joints_with_origin[joint2, 1]],
                    [ground_truth_joints_with_origin[joint1, 2], ground_truth_joints_with_origin[joint2, 2]],
                    c='b'
                )  
        
        ax.legend(loc='upper right')

    ani = animation.FuncAnimation(fig, update, frames=predicted_positions.shape[0], interval=100)

    if save_mp4_path:
        ani.save(save_mp4_path, writer=FFMpegWriter(fps=25))
    else:
        while True:
            plt.show()
        
def visualize_input_and_output_gcnext(input_positions, predicted_positions, ground_truth_positions,
                                skeleton_type=None,
                                save_mp4_path=None):
    
    if skeleton_type == 'h36m':
        connections = h36m_connections
        input_positions = np.stack((input_positions[:, :, 2],
                                    input_positions[:, :, 0],
                                    input_positions[:, :, 1]), axis=2)
        predicted_positions = np.stack((predicted_positions[:, :, 2],
                                       predicted_positions[:, :, 0],
                                       predicted_positions[:, :, 1]), axis=2)
        ground_truth_positions = np.stack((ground_truth_positions[:, :, 2],
                                           ground_truth_positions[:, :, 0],
                                           ground_truth_positions[:, :, 1]), axis=2)
    elif skeleton_type == 'amass':
        connections = amass_connections
    elif skeleton_type == 'incomplete_h36m':
        connections = incomplete_h36m_connections
        input_positions = np.stack((input_positions[:, :, 2],
                                    input_positions[:, :, 0],
                                    input_positions[:, :, 1]), axis=2)
        predicted_positions = np.stack((predicted_positions[:, :, 2],
                                       predicted_positions[:, :, 0],
                                       predicted_positions[:, :, 1]), axis=2)
        ground_truth_positions = np.stack((ground_truth_positions[:, :, 2],
                                           ground_truth_positions[:, :, 0],
                                           ground_truth_positions[:, :, 1]), axis=2)
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=55)

    total_frames = len(input_positions) + len(predicted_positions)

    def update(frame_idx):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if frame_idx < len(input_positions):
            ax.set_title(f"Input Motion - Frame: {frame_idx}")
            joints = input_positions[frame_idx]
            # Add artificial origin as the last joint
            joints = np.vstack([joints, np.array([0, 0, 0])])
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints[joint1, 0], joints[joint2, 0]],
                        [joints[joint1, 1], joints[joint2, 1]],
                        [joints[joint1, 2], joints[joint2, 2]], 'g', alpha=0.7)
        else:
            out_idx = frame_idx - len(input_positions)
            ax.set_title(f"Predicted vs Ground Truth - Frame: {out_idx}")
            joints_pred = predicted_positions[out_idx]
            # Add artificial origin as the last joint
            joints_pred = np.vstack([joints_pred, np.array([0, 0, 0])])
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints_pred[joint1, 0], joints_pred[joint2, 0]],
                        [joints_pred[joint1, 1], joints_pred[joint2, 1]],
                        [joints_pred[joint1, 2], joints_pred[joint2, 2]], 'b', alpha=0.7, label='Prediction' if connection == connections[0] else "")
            # Plot ground truth
            joints_gt = ground_truth_positions[out_idx]
            # Add artificial origin as the last joint
            joints_gt = np.vstack([joints_gt, np.array([0, 0, 0])])
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints_gt[joint1, 0], joints_gt[joint2, 0]],
                        [joints_gt[joint1, 1], joints_gt[joint2, 1]],
                        [joints_gt[joint1, 2], joints_gt[joint2, 2]], 'r', alpha=0.7, label='Ground Truth' if connection == connections[0] else "")
            
            ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100)

    if save_mp4_path:
        ani.save(save_mp4_path, writer=FFMpegWriter(fps=25))
    else:
        while True:
            plt.show()
    plt.close()

def visualize_input_and_output_physmop(ground_truth_positions, predicted_positions, num_input_frames,
                                skeleton_type=None,
                                save_mp4_path=None):
    """
    Biggest difference is that the the physmop model concatenates the input and output together
    """

    
    if skeleton_type == 'amass':
        connections = amass_connections

        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=55)

    total_frames = predicted_positions

    def update(frame_idx):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if frame_idx < len(input_positions):
            ax.set_title(f"Input Motion - Frame: {frame_idx}")
            joints = input_positions[frame_idx]
            # Add artificial origin as the last joint
            joints = np.vstack([joints, np.array([0, 0, 0])])
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints[joint1, 0], joints[joint2, 0]],
                        [joints[joint1, 1], joints[joint2, 1]],
                        [joints[joint1, 2], joints[joint2, 2]], 'g', alpha=0.7)
        else:
            out_idx = frame_idx - len(input_positions)
            ax.set_title(f"Predicted vs Ground Truth - Frame: {out_idx}")
            joints_pred = predicted_positions[out_idx]
            # Add artificial origin as the last joint
            joints_pred = np.vstack([joints_pred, np.array([0, 0, 0])])
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints_pred[joint1, 0], joints_pred[joint2, 0]],
                        [joints_pred[joint1, 1], joints_pred[joint2, 1]],
                        [joints_pred[joint1, 2], joints_pred[joint2, 2]], 'b', alpha=0.7, label='Prediction' if connection == connections[0] else "")
            # Plot ground truth
            joints_gt = ground_truth_positions[out_idx]
            # Add artificial origin as the last joint
            joints_gt = np.vstack([joints_gt, np.array([0, 0, 0])])
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints_gt[joint1, 0], joints_gt[joint2, 0]],
                        [joints_gt[joint1, 1], joints_gt[joint2, 1]],
                        [joints_gt[joint1, 2], joints_gt[joint2, 2]], 'r', alpha=0.7, label='Ground Truth' if connection == connections[0] else "")
            
            ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100)

    if save_mp4_path:
        ani.save(save_mp4_path, writer=FFMpegWriter(fps=25))
    else:
        while True:
            plt.show()
    plt.close()

