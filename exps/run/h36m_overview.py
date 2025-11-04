
from config import config
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from datasets.h36m_eval import H36MEval
# from datasets.h36m import H36MDataset

from visualize_motion import visualize_continuous_motion

config.motion.h36m_target_length = config.motion.h36m_target_length_eval
dataset = H36MEval(config, 'test')

incomplete_h36m_connections = [
        (-1, 0), (-1, 4), (-1, 8),
        (0, 1), (1, 2), (2, 3), 
        (4, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11), (14, 15),
        (9, 12), (12, 13), (13, 14), (14, 15), (15, 16),
        (9, 17), (17, 18), (18, 19), (19, 20), (20, 21)
    ]

all_full_sequences = []
for action in dataset._actions:
    seqs = dataset.get_full_sequences_for_action(action)
    for motion, root in seqs:
        all_full_sequences.append(motion)  # motion shape: [num_frames, 32, 3]
all_full_sequences = np.concatenate(all_full_sequences, axis=0)  # shape: [num_frames_total, 32, 3]
print("Shape of all full sequences:", all_full_sequences.shape)
used_joint_indexes = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)

# Only select the used joints
all_full_sequences = all_full_sequences[:, used_joint_indexes, :]  # shape: [num_frames_total, 22, 3]

# swap y and z axes
all_full_sequences = all_full_sequences[:, :, [0, 2, 1]]
# Compute mean positions for skeleton plot
mean_positions = np.mean(all_full_sequences, axis=0)  # shape: [22, 3]

relative_positions = all_full_sequences - mean_positions[np.newaxis, :, :]  # shape: [num_frames_total, 22, 3]
relative_min = np.min(relative_positions)
relative_max = np.max(relative_positions)
dist_x_lim = [-max(relative_min, relative_max), max(relative_min, relative_max)]
dist_x_lim = [-0.5, 0.5]



joint_names = ["LKnee", "LAnkle", "LFoot", "LToe",
               "RKnee", "RAnkle", "RFoot", "RToe",
               "Spine", "Neck", "Head", "Nose",
               "RShoulder", "RElbow", "RWrist", "RHand", "RThumb",
               "LShoulder", "LElbow", "LWrist", "LHand", "LThumb"]
fontsize = 14

global_ymax = 0
for joint_idx in range(all_full_sequences.shape[1]):
    joint_data = all_full_sequences[:, joint_idx, :]
    joint_data_centered = joint_data - np.mean(joint_data, axis=0)
    for i in range(3):  # x, y, z
        kde = gaussian_kde(joint_data_centered[:, i])
        x_vals = np.linspace(dist_x_lim[0], dist_x_lim[1], 200)
        density = kde(x_vals)
        global_ymax = max(global_ymax, np.max(density))

for joint_idx, joint_name in enumerate(joint_names):
    # fig = plt.figure(figsize=(14.4, 5.2)) # 3 per page in latex
    fig = plt.figure(figsize=(14.4, 3.75)) # 4 per page in latex

    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    
    # --- Distribution plot (KDE) ---
    ax_dist = fig.add_subplot(gs[0])
    joint_data = all_full_sequences[:, joint_idx, :]  # shape: [num_frames, 3]
    joint_data_centered = joint_data - np.mean(joint_data, axis=0)
    colors = ['green', 'blue', 'red']
    for i, axis in enumerate(['x', 'y', 'z']):
        kde = gaussian_kde(joint_data_centered[:, i])
        x_vals = np.linspace(joint_data_centered[:, i].min(), joint_data_centered[:, i].max(), 200)
        ax_dist.plot(x_vals, kde(x_vals), color=colors[i], label=axis)
    ax_dist.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax_dist.set_xlim(dist_x_lim)
    ax_dist.set_ylim([0, global_ymax * 1.05])
    mean_coords = mean_positions[joint_idx]
    mean_str = f"[{mean_coords[0]:.3f}, {mean_coords[1]:.3f}, {mean_coords[2]:.3f}]"
    ax_dist.set_title(f"{joint_name} relative data distribution \nMean: {mean_str}", fontsize=fontsize)
    ax_dist.legend()
    ax_dist.set_xlabel("Joint Displacement from Mean (m)", fontsize=fontsize)
    ax_dist.set_ylabel("Empirical Density", fontsize=fontsize)

    # --- Skeleton plot ---
    ax_skel = fig.add_subplot(gs[1], projection='3d')
    ax_skel.set_xlim([-0.5, 0.5])
    ax_skel.set_ylim([-0.5, 0.5])
    ax_skel.set_zlim([-1, 1])
    ax_skel.set_box_aspect([1, 1, 2])  # z is twice as large as x/y
    ax_skel.set_xlabel("x")
    ax_skel.set_ylabel("y")
    ax_skel.set_zlabel("z")
    # Reduce number of ticks
    ax_skel.set_xticks(np.linspace(-0.5, 0.5, 3))  
    ax_skel.set_yticks(np.linspace(-0.5, 0.5, 3)) 

    mean_positions = np.vstack([mean_positions, np.array([0, 0, 0])])
    # Plot skeleton
    for (i, j) in incomplete_h36m_connections:
        ax_skel.plot(
            [mean_positions[i, 0], mean_positions[j, 0]],
            [mean_positions[i, 1], mean_positions[j, 1]],
            [mean_positions[i, 2], mean_positions[j, 2]],
            color='gray', linewidth=1.5
        )
    # Plot all joints
    # ax_skel.scatter(mean_positions[:, 0], mean_positions[:, 2], mean_positions[:, 1], color='blue')

    # Highlight active joint
    # Plot XYZ axes at active joint
    joint_pos = mean_positions[joint_idx]
    axis_length = 0.1  # Adjust as needed
    # Plot XYZ axes at active joint
    ax_skel.quiver(joint_pos[0], joint_pos[1], joint_pos[2], axis_length, 0, 0, color='green', linewidth=2, label='x')
    ax_skel.quiver(joint_pos[0], joint_pos[1], joint_pos[2], 0, axis_length, 0, color='blue', linewidth=2, label='y')
    ax_skel.quiver(joint_pos[0], joint_pos[1], joint_pos[2], 0, 0, axis_length, color='red', linewidth=2, label='z')

    # Add mean coordinates as text in the plot (top left corner)
    mean_text = f"Mean: [{joint_pos[0]:.3f}, \n{joint_pos[1]:.3f}, \n{joint_pos[2]:.3f}]"
    # ax_skel.text2D(0.05, 0.95, mean_text, transform=ax_skel.transAxes, fontsize=fontsize, verticalalignment='top')

    ax_skel.view_init(elev=23.5, azim=40)

    ax_skel.set_title("Mean Skeleton (active joint highlighted)", fontsize=fontsize)
    ax_skel.legend( loc='upper left', 
                    bbox_to_anchor=(0.8, 1), 
                    fontsize=12)

    def print_view(event):
        azim = ax_skel.azim
        elev = ax_skel.elev
        fig_size = fig.get_size_inches()
        print(f"Current view: elev={elev}, azim={azim}, fig_size={fig_size}")

    fig.canvas.mpl_connect('button_release_event', print_view)
    plt.tight_layout()
    plt.savefig('dataset_overview/h36m_joint_distribution_{}.png'.format(joint_name), dpi=300)
    plt.close(fig)