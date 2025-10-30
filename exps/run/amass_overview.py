import os

import numpy as np
import glob

import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from tqdm import tqdm


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
joint_names = [
    "Spine1",      # 3
    "LKnee",       # 4
    "RKnee",       # 5
    "Spine2",      # 6
    "LAnkle",      # 7
    "RAnkle",      # 8
    "Spine3",      # 9
    "LFoot",       # 10
    "RFoot",       # 11
    "Neck",        # 12
    "LCollar",     # 13
    "RCollar",     # 14
    "Head",        # 15
    "LShoulder",   # 16
    "RShoulder",   # 17
    "LElbow",      # 18
    "RElbow",      # 19
    "LWrist",      # 20
    "RWrist"       # 21
]

fontsize = 14

data_dir = "exported_gt_joints"
txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

all_sequences = []

for file in tqdm(txt_files, desc="Loading joint files"):
    arr = np.loadtxt(file, delimiter=',')
    arr = arr.reshape(arr.shape[0], -1, 3)
    all_sequences.append(arr)
all_sequences = np.concatenate(all_sequences, axis=0)  # shape: [num_frames_total, num_joints, 3]
print("Shape of all sequences:", all_sequences.shape)

assert all_sequences.shape[1] == len(joint_names), f"The number of joints does not match the joint names list. \n All sequences shape: {all_sequences.shape}, len(joint_names): {len(joint_names)}"
# Compute mean positions for skeleton plot
mean_positions = np.mean(all_sequences, axis=0)  # shape: [num_joints, 3]

relative_positions = all_sequences - mean_positions[np.newaxis, :, :]  # shape: [num_frames_total, num_joints, 3]
flat_rel = relative_positions.reshape(-1)
low, high = np.percentile(flat_rel, [1, 99])
# dist_x_lim = [low, high]
dist_x_lim = [-0.5, 0.5]

num_hist_bins = 200  # You can adjust this

global_ymax = 0
for joint_idx in range(all_sequences.shape[1]):
    joint_data = all_sequences[:, joint_idx, :]
    joint_data_centered = joint_data - np.mean(joint_data, axis=0)
    for i in range(3):  # x, y, z
        hist, bins = np.histogram(joint_data_centered[:, i], bins=num_hist_bins, range=dist_x_lim, density=True)
        global_ymax = max(global_ymax, np.max(hist))

for joint_idx, joint_name in enumerate(joint_names):
    # fig = plt.figure(figsize=(14.4, 5.2)) # 3 per page in latex
    fig = plt.figure(figsize=(14.4, 3.75)) # 4 per page in latex

    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    
    # --- Distribution plot (KDE) ---
    ax_dist = fig.add_subplot(gs[0])
    joint_data = all_sequences[:, joint_idx, :]  # shape: [num_frames, 3]
    joint_data_centered = joint_data - np.mean(joint_data, axis=0)
    colors = ['green', 'blue', 'red']
    for i, axis in enumerate(['x', 'y', 'z']):
        data = joint_data_centered[:, i]
        hist, bins = np.histogram(data, bins=num_hist_bins, range=dist_x_lim, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax_dist.plot(bin_centers, hist, color=colors[i], label=axis)
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

    # mean_positions = np.vstack([mean_positions, np.array([0, 0, 0])])
    # Plot skeleton
    for (i, j) in amass_connections:
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

    ax_skel.view_init(elev=5.6, azim=-40)

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
    plt.savefig('dataset_overview/AMASS/amass_joint_distribution_{}.png'.format(joint_name), dpi=300)
    # plt.show()
    plt.close(fig)
