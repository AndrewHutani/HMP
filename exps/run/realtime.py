import torch
import numpy as np
import argparse
from config import config

from utils.misc import rotmat2xyz_torch, rotmat2euler_torch
from model import GCNext as Model
from datasets.h36m_eval import H36MEval
import matplotlib.pyplot as plt

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


        self.observed_motion = []
        self.predicted_motion = []
        self.ground_truth = []

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

    def plot_multiple_skeletons(self, motion_sequence, title="Multiple Skeletons"):
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
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        num_frames = motion_sequence.shape[0]
        colors = cm.viridis(np.linspace(0, 1, num_frames))  # Colormap for time

        for frame_idx in range(num_frames):
            joints = motion_sequence[frame_idx]
            color = colors[frame_idx]
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints[joint1, 0], joints[joint2, 0]],
                        [joints[joint1, 2], joints[joint2, 2]],
                        [joints[joint1, 1], joints[joint2, 1]],
                        color=color, alpha=0.7)

        plt.show()

    def visualize_input_and_output(self, input_motion, output_motion, gif_path="input_output.gif"):
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

        total_frames = input_motion.shape[0] + output_motion.shape[0]

        def update(frame_idx):
            ax.clear()
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            if frame_idx < input_motion.shape[0]:
                ax.set_title(f"Input Motion - Frame: {frame_idx}")
                joints = input_motion[frame_idx]
                color = 'r'
            else:
                out_idx = frame_idx - input_motion.shape[0]
                ax.set_title(f"Predicted Motion - Frame: {out_idx}")
                joints = output_motion[out_idx]
                color = 'g'
            for connection in connections:
                joint1, joint2 = connection
                ax.plot([joints[joint1, 0], joints[joint2, 0]],
                        [joints[joint1, 2], joints[joint2, 2]],
                        [joints[joint1, 1], joints[joint2, 1]], color, alpha=0.7)

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
    
    def add_global_translation(self):
        timesteps = self.predicted_motion.shape[0]

        offsets = np.arange(timesteps) * 0.03
        offsets = offsets.reshape(-1, 1)  # Reshape to [timesteps, 1]
        self.predicted_motion[:, :, 2] += offsets  # Add offsets to x-coordinates


    def regress_pred(self, visualize=False, debug=False):
        input_length = self.config.motion.h36m_input_length_dct
        target_length = self.config.motion.h36m_target_length_train
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
            motion_window = observed_motion[:input_length].clone()


        dct_m, idct_m = self.get_dct_matrix(config.motion.h36m_input_length_dct)
        dct_m = dct_m.unsqueeze(0)
        idct_m = idct_m.unsqueeze(0)
        if debug:
            print("dct_m shape:", dct_m.shape)
            print("idct_m shape:", idct_m.shape)

        outputs = []
        step = config.motion.h36m_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        
        for idx in range(num_step):
            motion_input = motion_window[:, self.joint_used_xyz, :].reshape(1, input_length, -1)
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = torch.matmul(dct_m[:, :input_length, :], motion_input)
                else:
                    motion_input_ = motion_input

            output = model(motion_input_, self.tau)
            output = torch.matmul(idct_m[:, :, :config.motion.h36m_input_length], output)[:, :step, :]

            if config.deriv_output:
                output = output + motion_input[:, -1:, :].repeat(1,step,1)

            output = output.reshape(step, -1, 3)  # [step, 22, 3]

            # Fill in the predicted joints into a full skeleton
            motion_pred = motion_window[-1].unsqueeze(0).repeat(step, 1, 1)
            motion_pred[:, self.joint_used_xyz, :] = output
            motion_pred[:, self.joint_to_ignore, :] = motion_pred[:, self.joint_equal, :]

            outputs.append(motion_pred)

            # Slide the window: remove first 'step' frames, append new predictions
            motion_window = torch.cat([motion_window[step:], motion_pred], dim=0)

        # Concatenate all predictions
        predictions = torch.cat(outputs, dim=0)[:25]  # [target_length, 32, 3]

        self.predicted_motion = predictions.cpu().detach().numpy()
        # self.add_global_translation()

        if debug:
            print("Predictions shape after filling in joints:", predictions.shape)
            print("----------------------------------------------------------\n")
        if visualize:
            # self.plot_multiple_skeletons(self.predicted_motion)
            self.visualize_motion(self.predicted_motion, self.ground_truth, title="Predicted Motion")
            # self.visualize_input_and_output(observed_motion.cpu(), self.predicted_motion)
            # np.save("realtime_predictions.npy", output.cpu())

    def evaluate(self):
        """
        ground_truth: [25, 32, 3] - the ground truth motion
        """
        mpjpe = np.mean(np.linalg.norm(self.predicted_motion*1000 - self.ground_truth*1000, axis=2), axis=1)
        selected_timesteps = [1, 9, 14, 24]
        return mpjpe[selected_timesteps]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="ckpt/baseline/model-iter-84000.pth", help='=encoder path')
parser.add_argument('--dyna', nargs='+', type=int, default=[0, 48], help='dynamic layer index')
args = parser.parse_args()

# Prepare model
model = Model(config, args.dyna)
state_dict = torch.load(args.model_pth)
model.load_state_dict(state_dict, strict=True)


actions = ["walking", "eating", "smoking", "discussion", "directions",
                        "greeting", "phoning", "posing", "purchases", "sitting",
                        "sittingdown", "takingphoto", "waiting", "walkingdog",
                        "walkingtogether"]

for action in actions:
    print(f"Evaluating action: {action}")
    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    dataset = H36MEval(config, 'test')
    walking_indices = dataset.get_indices_for_action(action)

    realtime_predictor = RealTimePrediction(model, config, tau=0.5)
    visualize = True
    debug = True
    mpjpe_all_samples = []

    for idx in walking_indices[0:4]:
        print("Progress: {}/{}".format(idx+1, len(walking_indices)))
        test_input, test_output = dataset.__getitem__(idx)
        full_motion = torch.cat([test_input, test_output], dim=0)
        realtime_predictor = RealTimePrediction(model, config, tau=0.5)  # re-init to clear state
        mpjpe_per_obs = []
        for i in range(1):
            test_input_ = test_input
            ground_truth = full_motion[-25:, :, :]
            visualize = True
            debug = False
            motion_input = realtime_predictor.batch_predict(test_input_, ground_truth, visualize, debug)
            mpjpe = realtime_predictor.evaluate()  # shape: (4,) for your 4 selected timesteps
            mpjpe_per_obs.append(mpjpe)
        mpjpe_all_samples.append(np.stack(mpjpe_per_obs))  # shape: (obs_len, 4)

    mpjpe_all_samples = np.stack(mpjpe_all_samples)  # shape: (num_samples, obs_len, 4)
    mpjpe_mean = np.mean(mpjpe_all_samples, axis=0)  # shape: (obs_len, 4)

    print("Averaged MPJPE for each observation length and each selected timestep: {}".format(action))
    for obs_len in range(mpjpe_mean.shape[0]):
        print(f"Obs {obs_len+1}: {mpjpe_mean[obs_len]}")