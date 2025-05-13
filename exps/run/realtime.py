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
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
        self.joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
        self.joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)


        self.observed_motion = []
        self.predicted_motion = []

    def visualize_motion(self, motion_sequence, title = "Visualized motion"):
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

        for frame_idx in range(motion_sequence.shape[0]):
            ax.clear()
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
                

            plt.pause(0.1)  # Pause to display each frame

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

    def predict(self, observed_motion):
        self.observed_motion.append(observed_motion)
        self.regress_pred()

    def regress_pred(self):
        print("Stacked observed motion shape:", torch.stack(self.observed_motion).shape)
        observed_motion = torch.stack(self.observed_motion).cuda()
        n,c,_ = observed_motion.shape  # n: number of timesteps, c: number of joints
        # Prepare input
        motion_input = observed_motion[:, self.joint_used_xyz, :].reshape(n, -1)  # Shape: [n, len(joint_used_xyz) * 3]

        with torch.no_grad():
            dct_m, _ = self.get_dct_matrix(n)
            _, idct_m = self.get_dct_matrix(50)
            print("dct_m shape:", dct_m.shape)
            print("idct_m shape:", idct_m.shape)
            if config.deriv_input:
                motion_input_ = motion_input.clone()
                motion_input_ = torch.matmul(dct_m[:n, :], motion_input_.cuda())
            else:
                motion_input_ = motion_input.clone()
            print("motion_input_ shape:", motion_input_.shape)
            motion_input_ = motion_input_.unsqueeze(0)
            print("motion_input_ shape after unsqueeze:", motion_input_.shape)

            if motion_input_.shape[1] < config.motion.h36m_input_length_dct:
                # Pad with the first motion frame if the number of timesteps is less than 50
                padding = motion_input_[:, :1, :].repeat(1, config.motion.h36m_input_length_dct - motion_input_.shape[1], 1)
                motion_input_ = torch.cat([padding, motion_input_], dim=1)
            elif motion_input_.shape[1] > config.motion.h36m_input_length_dct:
                # Truncate if the number of timesteps is greater than 50
                motion_input_ = motion_input_[:, -config.motion.h36m_input_length_dct:, :]

            output = model(motion_input_, self.tau)
            print("output shape:", output.shape)
            output = torch.matmul(idct_m, output)
            if config.deriv_output:
                last_frame = motion_input[-1:, :]  # Last observed frame
                output = output + last_frame.unsqueeze(0).repeat(1, 50, 1)
            output = output.reshape(50, -1, 3)
            print("output shape after adding last frame:", output.shape)
            motion_pred = torch.zeros(50, 32, 3).to(output.device)

            motion_pred[:, self.joint_used_xyz, :] = output  # Fill in the predicted joints
            motion_pred[:, self.joint_to_ignore, :] = motion_pred[:, self.joint_equal, :]  # Equalize ignored joints

            output = motion_pred
            print("output shape after filling in joints:", output.shape)
            print("----------------------------------------------------------\n")
            self.visualize_motion(output.cpu())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="ckpt/baseline/model-iter-84000.pth", help='=encoder path')
parser.add_argument('--dyna', nargs='+', type=int, default=[0, 48], help='dynamic layer index')
args = parser.parse_args()

# Prepare model
model = Model(config, args.dyna)
state_dict = torch.load(args.model_pth)
model.load_state_dict(state_dict, strict=True)

config.motion.h36m_target_length = config.motion.h36m_target_length_eval
dataset = H36MEval(config, 'test')
test_input, _ = dataset.__getitem__(0)
print("test_input shape:", test_input.shape)

reatime_predictor = RealTimePrediction(model, config, tau=0.5)
reatime_predictor.predict(test_input[0])
reatime_predictor.predict(test_input[1])
reatime_predictor.predict(test_input[2])
reatime_predictor.predict(test_input[3])