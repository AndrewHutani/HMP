import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config  import config
from model import GCNext as Model
from datasets.h36m_eval import H36MEval
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

results_keys = ['#2', '#10', '#14', '#25']
time_steps = [2, 10, 14, 25]

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale without making the limits the same."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_motion_with_ground_truth(predicted_positions, ground_truth_positions, time_steps, title="Predicted vs Ground Truth Motion"):
    """
    Visualize the predicted motion and ground truth in 3D for specific time steps, with skeleton connections.

    :param predicted_positions: Tensor of shape [num_frames, num_joints, 3] (predicted motion)
    :param ground_truth_positions: Tensor of shape [num_frames, num_joints, 3] (ground truth motion)
    :param time_steps: List of time steps to visualize (e.g., [2, 10, 14, 25])
    :param title: Title of the plot
    """
    # Define the connections between joints
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (11, 12), (12, 13), (13, 14), (14, 15),
        (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the default viewing angle
    ax.view_init(elev=20, azim=55)  # Adjust elevation and azimuth as needed
    ax.set_title(title)

    for frame_idx in time_steps:
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
        # ax.scatter(predicted_joints[:, 0], predicted_joints[:, 2], predicted_joints[:, 1], c='r', marker='o', label='Predicted')

        # Plot ground truth joints for the specific frame
        ground_truth_joints = ground_truth_positions[frame_idx - 1]
        # ax.scatter(ground_truth_joints[:, 0], ground_truth_joints[:, 2], ground_truth_joints[:, 1], c='b', marker='^', label='Ground Truth')

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

        # Add dummy scatter plots for the legend
        ax.scatter([], [], [], c='r', marker='o', label='Predicted')
        ax.scatter([], [], [], c='b', marker='^', label='Ground Truth')

        ax.legend()
        # Set the axes to equal scale
        set_axes_equal(ax)    

        plt.pause(1.0)  # Pause to display each frame

    plt.show()

def visualize_all_timesteps(predicted_positions, ground_truth_positions, time_steps, title="Predicted Motion"):
    """
    Visualize all predicted and ground truth motions in a grid of 3D plots.

    :param predicted_positions: Tensor of shape [num_frames, num_joints, 3] (predicted motion)
    :param ground_truth_positions: Tensor of shape [num_frames, num_joints, 3] (ground truth motion)
    :param time_steps: List of time steps to visualize (e.g., [1, 2, ..., 25])
    :param title: Title of the plot
    """
    num_timesteps = len(time_steps)
    cols = 8  # Number of columns in the grid
    rows = (num_timesteps + cols - 1) // cols  # Calculate the number of rows needed

    fig = plt.figure(figsize=(5, rows * 4))  # Adjust figure size as needed
    fig.suptitle(title, fontsize=16)

    # Define the connections between joints
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (11, 12), (12, 13), (13, 14), (14, 15),
        (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31)
    ]

    for i, frame_idx in enumerate(time_steps):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.set_xlim([-0.75, 0.75])
        ax.set_ylim([-0.75, 0.75])
        ax.set_zlim([-0.75, 0.75])

         # Hide the tick values but keep the gridlines and axis labels
        ax.tick_params(axis='x', labelbottom=False)  # Hide X-axis tick values
        ax.tick_params(axis='y', labelleft=False)   # Hide Y-axis tick values
        ax.tick_params(axis='z', labelleft=False)   # Hide Z-axis tick values


        # Update the title to include the current time step
        # ax.set_title(f"Time Step #{frame_idx}", pad=-15)
        ax.view_init(elev=20, azim=55)  # Adjust elevation and azimuth as needed

        # Plot predicted joints for the specific frame
        predicted_joints = predicted_positions[frame_idx]  # Subtract 1 because time_steps are 1-based
        # Draw skeleton connections for predicted motion
        for connection in connections:
            joint1, joint2 = connection
            ax.plot([predicted_joints[joint1, 0], predicted_joints[joint2, 0]],
                    [predicted_joints[joint1, 2], predicted_joints[joint2, 2]],
                    [predicted_joints[joint1, 1], predicted_joints[joint2, 1]], 'r', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.savefig("predicted_vs_ground_truth.png", format="png", dpi=300)
    plt.show()
    
def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36, tau):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)

    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,n,c,_ = motion_input.shape
        num_samples += b

        motion_input = motion_input.reshape(b, n, 32, 3)
        motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
        outputs = []
        step = config.motion.h36m_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :config.motion.h36m_input_length, :], motion_input_.cuda())
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_, tau)
                output = torch.matmul(idct_m[:, :, :config.motion.h36m_input_length], output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            output = output.reshape(-1, 22*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        motion_target = motion_target.detach()
        b,n,c,_ = motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b,n,22,3)
        motion_pred = motion_target.clone().reshape(b,n,32,3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]

        # Should be of shape [batch_size, num_frames, num_joints, 3]
        print(f"motion_pred shape: {motion_pred.shape}, motion_gt shape: {motion_gt.shape}")

        time_steps_indices = np.linspace(0, 24, 24).astype(np.int64)

        # Extract the frames for the specified time steps
        predicted_positions = motion_pred[:, time_steps_indices, :, :].reshape(-1, 32, 3).cpu().numpy()
        ground_truth_positions = motion_gt[:, time_steps_indices, :, :].reshape(-1, 32, 3).cpu().numpy()
        # visualize_all_timesteps(predicted_positions, ground_truth_positions, time_steps_indices)

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(config, model, dataloader, tau) :

    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    num_samples = 0
    pbar = dataloader
    m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36, tau)
    ret = {}
    for j in range(config.motion.h36m_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [ret[key][0] for key in results_keys]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    parser.add_argument('--dyna', nargs='+', type=int, default=[0, 48])
    args = parser.parse_args()

    model = Model(config, args.dyna)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)

    # Model preparation
    model.eval()
    model.cuda()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval

    dataset = H36MEval(config, 'test')
    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=0, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)
    
    TAU = 0.1
    mpjpe = test(config, model, dataloader, TAU)

    # Print the MPJPE results
    print("Evaluation Results (MPJPE in mm):")
    for i, key in enumerate(results_keys):
        print(f"{key}: {mpjpe[i]:.2f} mm")