import time
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
from prediction_times import prediction_times

class RealTimeGlobalPrediction(RealTimePrediction):
    # def add_global_translation(self, root_translation=None):
    #     """
    #     Add global translation to the predicted motion based on the root translation.

    #     :param root_translation: Optional; if provided, will use this translation instead of calculating it. tensor([num_frames, 32, 3])
    #     """
    #     global_observed_motion = []
    #     if root_translation is None:
    #         CONSTANT_VELOCITY = 0.03  # Adjust this value as needed
            
    #         # observed_motion: list of torch tensors, each [num_joints, 3]
    #         offsets = torch.arange(len(self.observed_motion), dtype=self.observed_motion[0].dtype, device=self.observed_motion[0].device) * CONSTANT_VELOCITY
            
    #         for i in range(len(self.observed_motion)):
    #             frame = self.observed_motion[i].clone()  # Clone to avoid modifying the original
    #             frame[:, 2] += offsets[i]                # Add offset to the z-coordinate
    #             global_observed_motion.append(frame)
                    
    #         timesteps = self.predicted_motion.shape[0] + len(self.observed_motion)

    #         offsets = np.arange(timesteps) * CONSTANT_VELOCITY

    #         offsets = offsets.reshape(-1, 1)  # Reshape to [timesteps, 1]
            

    #         for i in range(self.predicted_motion.shape[0]):
    #             self.predicted_motion[i, :, 2] += offsets[len(self.observed_motion) + i]

    #     else:
    #         root_translation_observed = root_translation[:len(self.observed_motion), :, :]  # shape: (num_frames, 32, 3)
    #         for i in range(len(self.observed_motion)):
    #             global_observed_motion.append(self.observed_motion[i] + root_translation_observed[i])  # Add root translation to each observed frame
            
    #         for i in range(self.predicted_motion.shape[0]):
    #             self.predicted_motion[i] += root_translation[len(self.observed_motion) + i].cpu().numpy()
            

    #     return global_observed_motion
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
state_dict = torch.load(args.model_pth, map_location = torch.device("cpu"))
model.load_state_dict(state_dict, strict=True)
model.to(torch.device("cpu"))  # Use CPU for inference


actions = ["walking", "eating", "smoking", "discussion", "directions",
                        "greeting", "phoning", "posing", "purchases", "sitting",
                        "sittingdown", "takingphoto", "waiting", "walkingdog",
                        "walkingtogether"]

action = "walking"  # Change this to the action you want to evaluate

config.motion.h36m_target_length = config.motion.h36m_target_length_eval
dataset = H36MEval(config, 'test')
walking_sample, root_sample = dataset.get_full_sequences_for_action(action)[0]
# print("Walking sample shape: ", walking_sample.shape)
# print("Root sample shape: ", root_sample.shape)
# print("Root sample: ", root_sample[:, 0, :])

realtime_predictor = RealTimeGlobalPrediction(model, config, tau=0.5)
visualize = False
debug = False


all_observed_motion = []
all_predicted_motion = []
latency_times = []

for i in range(walking_sample.shape[0] - config.motion.h36m_target_length):
# for i in range(100):
    test_input_ = walking_sample[i]
    ground_truth = walking_sample[i:i+config.motion.h36m_target_length]
    t0 = time.perf_counter()
    realtime_predictor.predict(test_input_, ground_truth, visualize, debug)
    t1 = time.perf_counter()
    global_observed_motion = realtime_predictor.add_global_translation()  # Add global translation to the predicted motion

    all_observed_motion.append(global_observed_motion[-config.motion.h36m_target_length:])
    all_predicted_motion.append(realtime_predictor.predicted_motion)
    latency_times.append(t1 - t0)

if prediction_times:
    avg_prediction_time = sum(prediction_times) / len(prediction_times)
    print(f"Average prediction time: {avg_prediction_time:.4f} seconds")
    

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')

all_points = np.array(all_predicted_motion)
xyz_min = all_points.min(axis=(0, 1, 2))  # shape: (3,)
xyz_max = all_points.max(axis=(0, 1, 2))  # shape: (3,)
center = (xyz_max + xyz_min) / 2
max_range = (xyz_max - xyz_min).max() / 2
max_range = 1.5


def update(frame_idx):
    num_skeletons = 3
    ax.clear()
    ax.set_title(f"Time step {frame_idx}")
    ax.view_init(elev=30, azim=-30)  # <-- Set your desired viewing angle here
    # Plot observed
    observed_motion = all_observed_motion[frame_idx]
    obs_indices = np.linspace(0, len(observed_motion) - 1, num_skeletons, dtype=int)
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
    pred_indices = np.linspace(0, len(predicted_motion) - 1, num_skeletons, dtype=int)
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
            
    # Compute center for current frame
    observed_motion = all_observed_motion[frame_idx]
    predicted_motion = all_predicted_motion[frame_idx]
    frame_points = np.concatenate([np.stack(observed_motion), predicted_motion], axis=0)
    center = (frame_points.max(axis=(0, 1)) + frame_points.min(axis=(0, 1))) / 2

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[2] - max_range, center[2] + max_range)
    ax.set_zlim(center[1] - max_range, center[1] + max_range)

ani = FuncAnimation(fig, update, frames=len(all_observed_motion), interval=40)
# ani.save("motion_progression.gif", writer="pillow", fps=15)
plt.show()