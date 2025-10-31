from config import config
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from tqdm import tqdm

# from datasets.h36m import H36MDataset

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

num_bins = 100        # Number of bins for joint value axis (x, y, or z)
num_time_bins = 50    # Number of bins for gait cycle time axis (in seconds)

data_dir = "exported_gt_joints"
txt_files = glob.glob(os.path.join(data_dir, "*treadmill_norm*.txt"))

all_sequences = []

for file in tqdm(txt_files, desc="Loading joint files"):
    arr = np.loadtxt(file, delimiter=',')
    arr = arr.reshape(arr.shape[0], -1, 3)
    all_sequences.append(arr)
# all_sequences = np.concatenate(all_sequences, axis=0)  # shape: [num_frames_total, num_joints, 3]
# print("Shape of all sequences:", all_sequences.shape)


# motions = np.array(motions)
frame_rate = 25.0  # Hz

# --- Detect gait cycles using RToe y-axis ---
toe_joint_idx = joint_names.index("RAnkle")
toe_axis_idx = 0  # x-axis

all_cycles = []  # List of (cycle_start, cycle_end) tuples for all motions

for sequence in all_sequences:
    toe_traj = sequence[:, toe_joint_idx, toe_axis_idx]
    frame_count = toe_traj.shape[0]
    sign_changes = np.where(np.diff(np.sign(toe_traj)))[0]
    transition_frames = np.append(sign_changes, frame_count)
    min_cycle_len = int(frame_rate * 0.3)

    # Check direction of first cycle
    if len(transition_frames) < 3:
        continue
    start0, end0 = transition_frames[0], transition_frames[2]
    cycle0 = toe_traj[start0:end0]
    direction0 = cycle0[5] - cycle0[0] if len(cycle0) > 5 else cycle0[-1] - cycle0[0]

    # If direction is not as desired, shift indices by one
    if direction0 < 0:  # If moving backward, shift
        transition_frames = transition_frames[1:]

    for i in range(0, len(transition_frames)-2, 2):
        start, end = transition_frames[i], transition_frames[i+2]
        if end - start < min_cycle_len:
            continue
        all_cycles.append((sequence, start, end))

print(f"Detected {len(all_cycles)} gait cycles across {len(all_sequences)} motions.")

# --- Calculate global min/max displacement ---
all_displacements = []
for joint_idx in range(len(joint_names)):
    for axis_idx in range(3):
        for motion, start, end in all_cycles:
            mean_val = motion[:, joint_idx, axis_idx].mean()
            cycle_vals = motion[start:end, joint_idx, axis_idx] - mean_val
            all_displacements.append(cycle_vals)
all_displacements_flat = np.concatenate(all_displacements)
global_min = all_displacements_flat.min()
global_max = all_displacements_flat.max()

# --- For each joint and axis, collect values and times from all cycles ---
for joint_idx, joint_name in enumerate(joint_names):
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        all_times = []
        all_vals = []
        cycle_durations = []

        for motion, start, end in all_cycles:
            mean_val = motion[:, joint_idx, axis_idx].mean()
            cycle_vals = motion[start:end, joint_idx, axis_idx] - mean_val
            cycle_len = end - start
            if cycle_len < 2:
                continue
            cycle_time = np.arange(cycle_len) / frame_rate  # time in seconds since cycle start
            all_times.append(cycle_time)
            all_vals.append(cycle_vals)
            cycle_durations.append(cycle_len / frame_rate)

        if not all_times or not all_vals:
            continue

        # Flatten arrays
        all_times_flat = np.concatenate(all_times)
        all_vals_flat = np.concatenate(all_vals)
        avg_cycle_time = np.mean(cycle_durations)

        # Bin by normalized time and value
        num_time_bins = int(np.ceil(np.mean(avg_cycle_time) * 25))
        val_bins = np.linspace(min(all_vals_flat), max(all_vals_flat), num_bins)
        time_bins = np.linspace(0, avg_cycle_time, num_time_bins + 1)
        density_matrix = np.zeros((num_time_bins, num_bins-1))

        for i in range(num_time_bins):
            time_mask = (all_times_flat >= time_bins[i]) & (all_times_flat < time_bins[i+1])
            data = all_vals_flat[time_mask]
            if len(data) == 0:
                density_matrix[i, :] = 0
            else:
                hist, _ = np.histogram(data, bins=val_bins, density=True)
                density_matrix[i, :] = hist

        # Plot
        Y = (time_bins[:-1] + time_bins[1:]) / 2
        X = (val_bins[:-1] + val_bins[1:]) / 2
        X, Y = np.meshgrid(X, Y)
        Z = density_matrix

        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

        ax_surface = fig.add_subplot(gs[0], projection='3d')
        surf = ax_surface.plot_surface(X, Y, Z, cmap='viridis')
        fig.colorbar(surf, ax=ax_surface, shrink=0.5, aspect=10, label='Density', pad=0.08)
        ax_surface.view_init(elev=45, azim=-50)
        ax_surface.set_xlabel(f'{joint_name} {axis_name} Value (m)', fontsize=fontsize)
        ax_surface.set_ylabel('Gait Cycle Time (s)', fontsize=fontsize)
        ax_surface.set_zlabel('Density', fontsize=fontsize)

        mean_positions = np.mean(motion, axis=0)
        mean_txt = f"Mean: [{mean_positions[joint_idx, 0]:.3f}, {mean_positions[joint_idx, 1]:.3f}, {mean_positions[joint_idx, 2]:.3f}]"
        ax_surface.set_title(f'{joint_name} relative {axis_name} data distribution over Gait Cycles\n {mean_txt}', fontsize=fontsize)
        
        # --- Skeleton Plot ---
        mean_positions = np.mean(motion, axis=0)
        mean_positions = np.vstack([mean_positions, np.array([0, 0, 0])])
        ax_skel = fig.add_subplot(gs[1], projection='3d')
        ax_skel.set_xlim([-0.5, 0.5])
        ax_skel.set_ylim([-0.5, 0.5])
        ax_skel.set_zlim([-1, 1])
        ax_skel.set_box_aspect([1, 1, 2])
        ax_skel.set_xlabel("x")
        ax_skel.set_ylabel("y")
        ax_skel.set_zlabel("z", labelpad=15)
        # ax_skel.tick_params(axis='x', which='major', pad=-5)
        # ax_skel.tick_params(axis='y', which='major', pad=-5)
        ax_skel.tick_params(axis='z', which='major', pad=15)
        ax_skel.set_xticks(np.linspace(-0.5, 0.5, 3))
        ax_skel.set_yticks(np.linspace(-0.5, 0.5, 3))

        for (i, j) in amass_connections:
            ax_skel.plot(
                [mean_positions[i, 0], mean_positions[j, 0]],
                [mean_positions[i, 1], mean_positions[j, 1]],
                [mean_positions[i, 2], mean_positions[j, 2]],
                color='gray', linewidth=1.5
            )

        joint_pos = mean_positions[joint_idx]
        axis_length = 0.1
        ax_skel.quiver(joint_pos[0], joint_pos[1], joint_pos[2], axis_length, 0, 0, color='green', linewidth=2, label='x')
        ax_skel.quiver(joint_pos[0], joint_pos[1], joint_pos[2], 0, axis_length, 0, color='blue', linewidth=2, label='y')
        ax_skel.quiver(joint_pos[0], joint_pos[1], joint_pos[2], 0, 0, axis_length, color='red', linewidth=2, label='z')

        ax_skel.view_init(elev=8.5, azim=62)
        ax_skel.set_title("Mean Skeleton (active joint highlighted)", fontsize=fontsize)
        ax_skel.legend(loc='upper left', bbox_to_anchor=(0.8, 1), fontsize=12)


        def print_view(event):
            azim = ax_surface.azim
            elev = ax_surface.elev
            fig_size = fig.get_size_inches()
            print(f"Current view: elev={elev}, azim={azim}, fig_size={fig_size}")

        fig.canvas.mpl_connect('button_release_event', print_view)
        
        plt.subplots_adjust(top=0.88)  # Lower values move plots closer to the top edge
        # plt.tight_layout()
        plt.savefig(f'walking_dataset_overview/amass/amass_joint_gaitcycle_surface_{joint_name}_{axis_name}.png', dpi=300)
        # plt.show()
        plt.close(fig)
