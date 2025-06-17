import torch
import numpy as np
import argparse
from config import config

from realtime import RealTimePrediction

from model import GCNext as Model
from datasets.h36m_eval import H36MEval
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

class RealTimeGlobalPrediction(RealTimePrediction):
    def add_global_translation(self):
        CONSTANT_VELOCITY = 0.03  # Adjust this value as needed
        
        # observed_motion: list of torch tensors, each [num_joints, 3]
        offsets = torch.arange(len(self.observed_motion), dtype=self.observed_motion[0].dtype, device=self.observed_motion[0].device) * CONSTANT_VELOCITY
        global_observed_motion = []
        for i in range(len(self.observed_motion)):
            frame = self.observed_motion[i].clone()  # Clone to avoid modifying the original
            frame[:, 2] += offsets[i]                # Add offset to the z-coordinate
            global_observed_motion.append(frame)
                
        timesteps = self.predicted_motion.shape[0] + len(self.observed_motion)

        offsets = np.arange(timesteps) * CONSTANT_VELOCITY

        offsets = offsets.reshape(-1, 1)  # Reshape to [timesteps, 1]
        

        for i in range(self.predicted_motion.shape[0]):
            self.predicted_motion[i, :, 2] += offsets[len(self.observed_motion) + i]

        return global_observed_motion

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

action = "walking"  # Change this to the action you want to evaluate

config.motion.h36m_target_length = config.motion.h36m_target_length_eval
dataset = H36MEval(config, 'test')
action_indices = dataset.get_indices_for_action(action)

realtime_predictor = RealTimeGlobalPrediction(model, config, tau=0.5)
visualize = False
debug = False

test_input, test_output = dataset.__getitem__(action_indices[0])
full_motion = torch.cat([test_input, test_output], dim=0)

all_observed_motion = []
all_predicted_motion = []

for i in range(test_input.shape[0]):
    test_input_ = test_input[i]
    ground_truth = full_motion[i:i+config.motion.h36m_target_length]
    realtime_predictor.predict(test_input_, ground_truth, visualize, debug)
    global_observed_motion = realtime_predictor.add_global_translation()  # Add global translation to the predicted motion

    all_observed_motion.append(global_observed_motion)
    all_predicted_motion.append(realtime_predictor.predicted_motion)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')

all_points = np.array(all_predicted_motion)
xyz_min = all_points.min(axis=(0, 1, 2))  # shape: (3,)
xyz_max = all_points.max(axis=(0, 1, 2))  # shape: (3,)
center = (xyz_max + xyz_min) / 2
max_range = (xyz_max - xyz_min).max() / 2



def update(frame_idx):
    ax.clear()
    ax.set_title(f"Time step {frame_idx}")
    ax.view_init(elev=30, azim=-30)  # <-- Set your desired viewing angle here
    # Plot observed
    observed_motion = all_observed_motion[frame_idx]
    obs_indices = np.linspace(0, len(observed_motion) - 1, 4, dtype=int)
    obs_colors = cm.Blues(np.linspace(0.5, 1, len(obs_indices)))
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (11, 12), (12, 13), (13, 14), (14, 15),
        (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31)
    ]
    for i, frame_idx_obs in enumerate(obs_indices):
        joints = observed_motion[frame_idx_obs]
        color = obs_colors[i]
        for connection in connections:
            joint1, joint2 = connection
            ax.plot([joints[joint1, 0], joints[joint2, 0]],
                    [joints[joint1, 2], joints[joint2, 2]],
                    [joints[joint1, 1], joints[joint2, 1]],
                    color=color, alpha=0.7, linewidth=1.5)
    # Plot predicted
    predicted_motion = all_predicted_motion[frame_idx]
    pred_indices = np.linspace(0, len(predicted_motion) - 1, 4, dtype=int)
    pred_colors = cm.Oranges(np.linspace(0.5, 1, len(pred_indices)))
    for i, frame_idx_pred in enumerate(pred_indices):
        joints = predicted_motion[frame_idx_pred]
        color = pred_colors[i]
        for connection in connections:
            joint1, joint2 = connection
            ax.plot([joints[joint1, 0], joints[joint2, 0]],
                    [joints[joint1, 2], joints[joint2, 2]],
                    [joints[joint1, 1], joints[joint2, 1]],
                    color=color, alpha=0.7, linewidth=1.5, linestyle='--')
    # Set axis limits (optional: you can compute these globally for all frames)
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[2] - max_range, center[2] + max_range)
    ax.set_zlim(center[1] - max_range, center[1] + max_range)

ani = FuncAnimation(fig, update, frames=len(all_observed_motion), interval=40)
ani.save("motion_progression.gif", writer="pillow", fps=15)
# plt.show()