from config import config
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

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
joint_names = ["LKnee", "LAnkle", "LFoot", "LToe",
               "RKnee", "RAnkle", "RFoot", "RToe",
               "Spine", "Neck", "Head", "Nose",
               "RShoulder", "RElbow", "RWrist", "RHand", "RThumb",
               "LShoulder", "LElbow", "LWrist", "LHand", "LThumb"]
fontsize = 14

num_bins = 100        # Number of bins for joint value axis (x, y, or z)
num_time_bins = 50    # Number of bins for gait cycle time axis (in seconds)

motions = []
seqs = dataset.get_full_sequences_for_action("walking")
used_joint_indexes = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
for motion, root in seqs:
    motion = motion[:, :, [0, 2, 1]]
    motions.append(motion[:, used_joint_indexes, :])


# motions = np.array(motions)
frame_rate = 25.0  # Hz

# --- Detect gait cycles using RToe y-axis ---
toe_joint_idx = joint_names.index("RAnkle")
toe_axis_idx = 1  # y-axis

all_cycles = []  # List of (cycle_start, cycle_end) tuples for all motions

for motion in motions:
    toe_traj = motion[:, toe_joint_idx, toe_axis_idx]
    frame_count = toe_traj.shape[0]
    sign_changes = np.where(np.diff(np.sign(toe_traj)))[0]
    transition_frames = np.append(sign_changes, frame_count)
    min_cycle_len = int(frame_rate * 0.3)
    # Use every two sign changes for full gait cycles
    for i in range(0, len(transition_frames)-2, 2):
        start, end = transition_frames[i], transition_frames[i+2]
        if end - start < min_cycle_len:
            continue
        all_cycles.append((motion, start, end))

print(f"Detected {len(all_cycles)} gait cycles across {len(motions)} motions.")

# --- Calculate global min/max displacement ---
all_displacements = []
for joint_idx in range(len(joint_names)):
    for axis_idx in range(3):
        for motion, start, end in all_cycles:
            mean_val = motion[:, joint_idx, axis_idx].numpy().mean()
            cycle_vals = motion[start:end, joint_idx, axis_idx].numpy() - mean_val
            all_displacements.append(cycle_vals)
all_displacements_flat = np.concatenate(all_displacements)
global_min = all_displacements_flat.min()
global_max = all_displacements_flat.max()

# --- For each joint and axis, collect values and times from all cycles ---
for joint_idx, joint_name in enumerate(joint_names):
    joint_mean = motion[:, joint_idx, :].numpy().mean(axis=0)
    mean_txt = f"Mean: [{joint_mean[0]:.3f}, {joint_mean[1]:.3f}, {joint_mean[2]:.3f}]"
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f'{joint_name} relative data distribution over gait cycle\n{mean_txt}', fontsize=fontsize)
    gs = fig.add_gridspec(1, 4, width_ratios=[2, 2, 2, 1.5])
    axis_surfaces = []
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        ax_surface = fig.add_subplot(gs[axis_idx], projection='3d')
        axis_surfaces.append(ax_surface)
        all_times = []
        all_vals = []
        cycle_durations = []

        for motion, start, end in all_cycles:
            mean_val = motion[:, joint_idx, axis_idx].numpy().mean()
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

        surf = ax_surface.plot_surface(X, Y, Z, cmap='viridis')

        ax_surface.view_init(elev=45, azim=-50)
        ax_surface.set_xlabel(f'Displacement \nfrom mean (m)', fontsize=fontsize-3, labelpad=10)
        ax_surface.set_ylabel('Gait Cycle Time (s)', fontsize=fontsize-3)

        ax_surface.set_title(f'{axis_name}-axis', fontsize=fontsize)
    
    xlims = [axis_surfaces[i].get_xlim() for i in range(3)]
    ylims = [axis_surfaces[i].get_ylim() for i in range(3)]
    zlims = [axis_surfaces[i].get_zlim() for i in range(3)]

    axis_surfaces[2].set_zlabel('Density', fontsize=fontsize-3)

    for i in range(3):
        axis_surfaces[i].set_xlim([min(xlims, key=lambda x: x[0])[0], max(xlims, key=lambda x: x[1])[1]])
        axis_surfaces[i].set_ylim([min(ylims, key=lambda y: y[0])[0], max(ylims, key=lambda y: y[1])[1]])
        axis_surfaces[i].set_zlim([min(zlims, key=lambda z: z[0])[0], max(zlims, key=lambda z: z[1])[1]])


    # --- Skeleton Plot ---
    mean_positions = np.mean(motion.numpy(), axis=0)
    mean_positions = np.vstack([mean_positions, np.array([0, 0, 0])])
    ax_skel = fig.add_subplot(gs[3], projection='3d')
    ax_skel.set_xlim([-0.5, 0.5])
    ax_skel.set_ylim([-0.5, 0.5])
    ax_skel.set_zlim([-1, 1])
    ax_skel.set_box_aspect([1, 1, 2])
    ax_skel.set_xlabel("x")
    ax_skel.set_ylabel("y")
    ax_skel.set_zlabel("z")
    ax_skel.set_xticks(np.linspace(-0.5, 0.5, 3))
    ax_skel.set_yticks(np.linspace(-0.5, 0.5, 3))

    for (i, j) in incomplete_h36m_connections:
        ax_skel.plot(
            [mean_positions[i, 0], mean_positions[j, 0]],
            [mean_positions[i, 1], mean_positions[j, 1]],
            [mean_positions[i, 2], mean_positions[j, 2]],
            color='gray', linewidth=1.5
        )

    joint_pos = mean_positions[joint_idx]
    axis_length = 0.2
    ax_skel.quiver(joint_pos[0], joint_pos[1], joint_pos[2], axis_length, 0, 0, color='green', linewidth=2, label='x')
    ax_skel.quiver(joint_pos[0], joint_pos[1], joint_pos[2], 0, axis_length, 0, color='blue', linewidth=2, label='y')
    ax_skel.quiver(joint_pos[0], joint_pos[1], joint_pos[2], 0, 0, axis_length, color='red', linewidth=2, label='z')

    ax_skel.view_init(elev=8.5, azim=62)
    ax_skel.set_title("Mean Skeleton \n(active joint highlighted)", fontsize=fontsize)
    ax_skel.legend(loc='upper left', bbox_to_anchor=(0.8, 1), fontsize=12)

    # plt.tight_layout()
    plt.savefig(f'walking_dataset_overview/h36m/h36m_joint_gaitcycle_surface_{joint_name}.png', dpi=300)
    # plt.show()
    plt.close(fig)
