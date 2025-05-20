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
        # predicted_connections = [
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
            
            # Add joint indices as text annotations
            for joint_idx, (x, y, z) in enumerate(joints):
                ax.text(x, z, y, str(joint_idx), color='blue', fontsize=8)

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

    def predict(self, observed_motion, visualize=False, debug=False):
        self.observed_motion.append(observed_motion)
        self.regress_pred(visualize, debug)
        return self.observed_motion

    def regress_pred(self, visualize=False, debug=False):
        if debug:
            print("Stacked observed motion shape:", torch.stack(self.observed_motion).shape)
        observed_motion = torch.stack(self.observed_motion).cuda()
        # if visualize:
        #     self.visualize_motion(observed_motion.cpu(), title="Observed motion")
        n, c, _ = observed_motion.shape  # n: number of timesteps, c: number of joints
        # Prepare input
        motion_input = observed_motion[:, self.joint_used_xyz, :].reshape(n, -1)  # Shape: [n, len(joint_used_xyz) * 3]

        # Check if the input has enough timesteps
        if motion_input.shape[0] < config.motion.h36m_input_length_dct:
            # Pad the beginning with the first motion frame
            padding = motion_input[:1, :].repeat(config.motion.h36m_input_length_dct - motion_input.shape[0], 1)
            if debug:
                print("Padded motion with {} timesteps".format(config.motion.h36m_input_length_dct - motion_input.shape[0]))
            motion_input = torch.cat([padding, motion_input], dim=0)
            
        elif motion_input.shape[0] > config.motion.h36m_input_length_dct:
            # Truncate the input to the last 50 timesteps
            motion_input = motion_input[-config.motion.h36m_input_length_dct:, :]
            if debug:
                print("Truncated motion to {} timesteps".format(config.motion.h36m_input_length_dct))

        motion_input = motion_input.unsqueeze(0) # Add batch dimension since the model expects a batch input. Shape: [1, 50, len(joint_used_xyz) * 3]
        if debug:
            print("motion_input shape:", motion_input.shape)

        dct_m, idct_m = self.get_dct_matrix(config.motion.h36m_input_length_dct)
        dct_m = dct_m.unsqueeze(0)
        idct_m = idct_m.unsqueeze(0)
        if debug:
                print("dct_m shape:", dct_m.shape)
                print("idct_m shape:", idct_m.shape)
        
        with torch.no_grad():
            if config.deriv_input:
                motion_input_ = motion_input.clone()
                motion_input_ = torch.matmul(dct_m[:, :config.motion.h36m_input_length, :], motion_input_.cuda())
            else:
                motion_input_ = motion_input.clone()

            output = model(motion_input_, self.tau)
            output = torch.matmul(idct_m[:, :, :config.motion.h36m_input_length], output)[:, :config.motion.h36m_target_length, :]
            if debug:
                print("Output shape after idct_m:", output.shape)

            if config.deriv_output:
                output = output + motion_input[:, -1:, :].repeat(1,config.motion.h36m_target_length,1)
            if debug:
                print("Output shape after deriv_output:", output.shape)

            output = output.reshape(config.motion.h36m_target_length, -1, 3)
            if debug:
                print("output shape after reshaping:", output.shape)

            motion_pred = torch.zeros(25, 32, 3).to(output.device)

            motion_pred[:, self.joint_used_xyz, :] = output  # Fill in the predicted joints
            motion_pred[:, self.joint_to_ignore, :] = motion_pred[:, self.joint_equal, :]  # Equalize ignored joints

            # output = motion_pred
            if debug:
                print("output shape after filling in joints:", output.shape)
                print("----------------------------------------------------------\n")
            if visualize:
                self.visualize_motion(motion_pred.cpu())
                # np.save("realtime_predictions.npy", output.cpu())

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

realtime_predictor = RealTimePrediction(model, config, tau=0.5)
for i in range(test_input.shape[0]):
    visualize = False if i != 49 else True
    debug = True if i != 49 else True
    motion_input = realtime_predictor.predict(test_input[i], visualize, debug)

print("Model uses the following motion_input: ", torch.stack(motion_input).shape)

# The motion_input from the model should be the same as the test_input
assert torch.all(torch.eq(test_input, torch.stack(motion_input)))