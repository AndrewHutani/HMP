import torch
import numpy as np
import argparse
from config import config

from model import GCNext as Model
import matplotlib.pyplot as plt

import time

class RealTimePrediction():
    def __init__(self, model, config, tau):
        self.config = config
        self.tau = tau

        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
        self.joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
        self.joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)
        
        dct_m, idct_m = self.get_dct_matrix(config.motion.h36m_input_length_dct)
        self.dct_m = dct_m.unsqueeze(0)
        self.idct_m = idct_m.unsqueeze(0)

        self.total_prediction_horizon = 25


        self.observed_motion = [] # Will be a list of tensors, each tensor is [num_joints, 3]
        self.predicted_motion = [] # Will be a numpy array of shape [self.total_prediction_horizon, 32, 3] after prediction
        self.ground_truth = [] # Will be a numpy array of shape [self.total_prediction_horizon, 32, 3] for evaluation

    def visualize_motion(self, motion_sequence, ground_truth = None, title = "Visualized motion"):
        # Define the connections between joints
        # connections = [
        #     (0, 1), (1, 2), (2, 3),
        #     (4, 5), (5, 6), (6, 7),
        #     (8 ,9), (9, 10), (10, 11),
        #     (8, 12), (12, 13), (13, 14), (14, 15), (15, 16),
        #     (8, 17), (17, 18), (18, 19), (19, 20), (20, 21),
        # ]
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

        for frame_idx in range(motion_sequence.shape[0]):
            ax.clear()
            fig_title = title + " - Frame: {}".format(frame_idx)
            ax.set_title(fig_title)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            joints = motion_sequence[frame_idx]
            # Draw skeleton connections for predicted motion
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints[joint1, 0], joints[joint2, 0]],
                        [joints[joint1, 2], joints[joint2, 2]],
                        [joints[joint1, 1], joints[joint2, 1]], 'r', alpha=0.5)
                if ground_truth is not None:
                    gt_joints = ground_truth[frame_idx]
                    ax.plot([gt_joints[joint1, 0], gt_joints[joint2, 0]],
                        [gt_joints[joint1, 2], gt_joints[joint2, 2]],
                        [gt_joints[joint1, 1], gt_joints[joint2, 1]], 'b', alpha=0.5)



            plt.pause(0.1)  # Pause to display each frame
        plt.show(block=True)

    def plot_multiple_skeletons(self, global_observed_motion, number_of_samples = None, visualize_observed = False, title="Multiple Skeletons"):
        """
        Plots all skeletons from a motion sequence in a single 3D plot.
        Each skeleton corresponds to one frame.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm

        # Define the connections between joints (same as before)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
            (11, 12), (12, 13), (13, 14), (14, 15),
            (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
            (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31)
        ]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)

        if visualize_observed:
            observed_motion = torch.stack(global_observed_motion).cpu().numpy()  # Convert to numpy array

        # Prepare indices for observed and predicted motions
        if number_of_samples is not None:
            # For observed motion
            if observed_motion is not None:
                if len(observed_motion) >= number_of_samples:
                    obs_indices = np.linspace(0, len(observed_motion) - 1, number_of_samples, dtype=int)
                else:
                    obs_indices = np.arange(len(observed_motion))
            else:
                obs_indices = []
            
            # For predicted motion
            if len(self.predicted_motion) >= number_of_samples:
                pred_indices = np.linspace(0, len(self.predicted_motion) - 1, number_of_samples, dtype=int)
            else:
                pred_indices = np.arange(len(self.predicted_motion))
        else:
            obs_indices = np.arange(len(observed_motion)) if observed_motion is not None else []
            pred_indices = np.arange(len(self.predicted_motion))
        
        # Plot observed motion frames (e.g., blue)
        if observed_motion is not None and len(obs_indices) > 0:
            obs_colors = cm.Blues(np.linspace(0.5, 1, len(obs_indices)))
            for i, frame_idx in enumerate(obs_indices):
                joints = observed_motion[frame_idx]
                color = obs_colors[i]
                for connection in connections:
                    joint1, joint2 = connection
                    ax.plot([joints[joint1, 0], joints[joint2, 0]],
                            [joints[joint1, 2], joints[joint2, 2]],
                            [joints[joint1, 1], joints[joint2, 1]],
                            color=color, alpha=0.7, linewidth=1.5)

        # Plot predicted motion frames (e.g., orange)
        pred_colors = cm.Oranges(np.linspace(0.5, 1, len(pred_indices)))
        for i, frame_idx in enumerate(pred_indices):
            joints = self.predicted_motion[frame_idx]
            color = pred_colors[i]
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints[joint1, 0], joints[joint2, 0]],
                        [joints[joint1, 2], joints[joint2, 2]],
                        [joints[joint1, 1], joints[joint2, 1]],
                        color=color, alpha=0.7, linewidth=1.5, linestyle='--')
                
        # Dynamically set equal axis scale for all axes
        all_points = []
        if observed_motion is not None and len(obs_indices) > 0:
            all_points.append(observed_motion[obs_indices].reshape(-1, 3))
        if len(pred_indices) > 0:
            all_points.append(self.predicted_motion[pred_indices].reshape(-1, 3))
        if all_points:
            all_points = np.concatenate(all_points, axis=0)
            xyz_min = all_points.min(axis=0)
            xyz_max = all_points.max(axis=0)
            center = (xyz_max + xyz_min) / 2
            max_range = (xyz_max - xyz_min).max() / 2
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[2] - max_range, center[2] + max_range)
            ax.set_zlim(center[1] - max_range, center[1] + max_range)

        plt.show()

    def visualize_input_and_output(self, gif_path="input_output.gif"):
        import matplotlib.animation as animation

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
            (11, 12), (12, 13), (13, 14), (14, 15),
            (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
            (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31)
        ]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=55)


        input_motion = self.observed_motion
        output_motion = self.predicted_motion

        total_frames = len(input_motion) + len(output_motion)

        def update(frame_idx):
            ax.clear()
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            if frame_idx < len(input_motion):
                ax.set_title(f"Input Motion - Frame: {frame_idx}")
                joints = input_motion[frame_idx]
                color = 'r'
                for connection in connections:
                    joint1, joint2 = connection
                    ax.plot([joints[joint1, 0], joints[joint2, 0]],
                            [joints[joint1, 2], joints[joint2, 2]],
                            [joints[joint1, 1], joints[joint2, 1]], color, alpha=0.7)
            else:
                out_idx = frame_idx - len(input_motion)
                ax.set_title(f"Predicted vs Ground Truth - Frame: {out_idx}")
                # Plot predicted output (e.g., green)
                joints_pred = output_motion[out_idx]
                for connection in connections:
                    joint1, joint2 = connection
                    ax.plot([joints_pred[joint1, 0], joints_pred[joint2, 0]],
                            [joints_pred[joint1, 2], joints_pred[joint2, 2]],
                            [joints_pred[joint1, 1], joints_pred[joint2, 1]], 'g', alpha=0.7, label='Prediction' if connection == connections[0] else "")
                # Plot ground truth (e.g., blue)
                joints_gt = ground_truth[out_idx]
                for connection in connections:
                    joint1, joint2 = connection
                    ax.plot([joints_gt[joint1, 0], joints_gt[joint2, 0]],
                            [joints_gt[joint1, 2], joints_gt[joint2, 2]],
                            [joints_gt[joint1, 1], joints_gt[joint2, 1]], 'b', alpha=0.7, label='Ground Truth' if connection == connections[0] else "")
                # Add legend only once
                handles, labels = ax.get_legend_handles_labels()
                if not handles:
                    ax.legend(["Prediction", "Ground Truth"])

        ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100)
        ani.save(gif_path, writer='pillow', fps=10)
        plt.close(fig)
        print(f"Saved GIF to {gif_path}")

    def get_dct_matrix(self, N):
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        return torch.tensor(dct_m).float().cuda(), torch.tensor(idct_m).float().cuda()

    def predict(self, observed_motion, ground_truth, visualize=False, debug=False):
        self.observed_motion.append(observed_motion)
        self.ground_truth = ground_truth.cpu().detach().numpy()
        self.regress_pred(visualize, debug)
        return self.observed_motion
    
    def batch_predict(self, observed_motion, ground_truth, visualize=False, debug=False):
        self.observed_motion = []
        for i in range(observed_motion.shape[0]):
            self.observed_motion.append(observed_motion[i])
        if visualize:
            self.visualize_motion(observed_motion.cpu(), title="Observed motion")
        self.ground_truth = ground_truth.cpu().detach().numpy()
        self.regress_pred(visualize, debug)
        return self.observed_motion

    def regress_pred(self, visualize=False, debug=False):
        t0 = time.time()
        input_length = self.config.motion.h36m_input_length_dct
        if debug:
            print("Stacked observed motion shape:", torch.stack(self.observed_motion).shape)
        observed_motion = torch.stack(self.observed_motion).cuda()
        n, c, _ = observed_motion.shape  # n: number of timesteps, c: number of joints
        # Prepare input
        motion_input = observed_motion[:, self.joint_used_xyz, :].reshape(n, -1)  # Shape: [n, len(joint_used_xyz) * 3]

        # Start with the first input_length frames
        if observed_motion.shape[0] < input_length:
            # Pad with the first frame if not enough
            pad = observed_motion[0:1].repeat(input_length - observed_motion.shape[0], 1, 1)
            motion_window = torch.cat([pad, observed_motion], dim=0)
        else:
            # Use the last input_length frames
            motion_window = observed_motion[-input_length:].clone()
        

        outputs = []
        chunk_prediction_length = config.motion.h36m_target_length_train
        if chunk_prediction_length == self.total_prediction_horizon:
            num_prediction_chunks = 1
        else:
            num_prediction_chunks = self.total_prediction_horizon // chunk_prediction_length + 1
        
        for idx in range(num_prediction_chunks):
            motion_input = motion_window[:, self.joint_used_xyz, :].reshape(1, input_length, -1)
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = torch.matmul(self.dct_m[:, :input_length, :], motion_input)
                else:
                    motion_input_ = motion_input

            output = self.model(motion_input_, self.tau)
            output = torch.matmul(self.idct_m[:, :, :config.motion.h36m_input_length], output)[:, :chunk_prediction_length, :]

            if config.deriv_output:
                output = output + motion_input[:, -1:, :].repeat(1,chunk_prediction_length,1)

            output = output.reshape(chunk_prediction_length, -1, 3)  # [prediction_horizon, 22, 3]

            # Fill in the predicted joints into a full skeleton
            motion_pred = motion_window[-1].unsqueeze(0).repeat(chunk_prediction_length, 1, 1)
            motion_pred[:, self.joint_used_xyz, :] = output
            motion_pred[:, self.joint_to_ignore, :] = motion_pred[:, self.joint_equal, :]

            outputs.append(motion_pred)

            # Slide the window: remove first 'step' frames, append new predictions
            motion_window = torch.cat([motion_window[chunk_prediction_length:], motion_pred], dim=0)

        # Concatenate all predictions
        predictions = torch.cat(outputs, dim=0)[:self.total_prediction_horizon]  # [target_length, 32, 3]

        self.predicted_motion = predictions.cpu().detach().numpy()
        # self.add_global_translation()

        if debug:
            print("Predictions shape after filling in joints:", predictions.shape)
            print("----------------------------------------------------------\n")
        if visualize:
            # self.plot_multiple_skeletons(self.predicted_motion)
            self.visualize_motion(self.predicted_motion, self.ground_truth, title="Predicted Motion")
            # np.save("realtime_predictions.npy", output.cpu())
        
        t1 = time.time()
        # if debug:
        # print(f"Prediction time: {t1 - t0:.2f} seconds")

    def evaluate(self):
        """
        ground_truth: [self.total_prediction_horizon, 32, 3] - the ground truth motion
        """
        mpjpe = np.mean(np.linalg.norm(self.predicted_motion*1000 - self.ground_truth*1000, axis=2), axis=1)
        selected_timesteps = [1, 9, 14, 24]
        return mpjpe[selected_timesteps]
    
    def evaluate_upper_and_lower_seperately(self):
        """
        ground_truth: [self.total_prediction_horizon, 32, 3] - the ground truth motion
        """
        lower_body_indices = np.arange(0, 11).tolist()
        upper_body_indices = np.arange(11, 32).tolist()

        predicted_upper = self.predicted_motion[:, upper_body_indices] * 1000
        predicted_lower = self.predicted_motion[:, lower_body_indices] * 1000
        ground_truth_upper = self.ground_truth[:, upper_body_indices] * 1000
        ground_truth_lower = self.ground_truth[:, lower_body_indices] * 1000

        mpjpe_upper = np.mean(np.linalg.norm(predicted_upper - ground_truth_upper, axis=2), axis=1)
        mpjpe_lower = np.mean(np.linalg.norm(predicted_lower - ground_truth_lower, axis=2), axis=1)

        selected_timesteps = [1, 9, 14, 24]
        return mpjpe_upper[selected_timesteps], mpjpe_lower[selected_timesteps]