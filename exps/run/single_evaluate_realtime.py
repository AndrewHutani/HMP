import time
import torch
import numpy as np
import argparse
from gcnext_model.config import config

from gcnext_model.RealtimeGCNext import RealtimeGCNext

from model import GCNext as Model
from datasets.h36m_eval import H36MEval
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from prediction_times import prediction_times

from visualize_motion import visualize_continuous_motion, visualize_motion_with_ground_truth

class RealTimeGlobalPrediction(RealtimeGCNext):
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
    def add_global_translation(self, passthrough = False):
        if passthrough:
            return self.observed_motion
        
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

amass_sample = np.loadtxt("gt_J_treadmill_norm.txt", delimiter=',')
amass_sample = amass_sample.reshape(-1, 22, 3)  # Reshape to [num_frames, num_joints, 3]
# the xyz axes are also not in the same order so reorder them
amass_sample = np.stack((amass_sample[:, :, 1], 
                               amass_sample[:, :, 2],
                               amass_sample[:, :, 0]), axis=2)  # [num_frames, num_joints, 3]
print("AMASS sample shape: ", amass_sample.shape)


realtime_predictor = RealTimeGlobalPrediction(model, config, tau=0.5)
visualize = False
debug = False


# for i in range(walking_sample.shape[0] - config.motion.h36m_target_length):
# for i in range(100):
idx = 2
test_input_ = torch.tensor(walking_sample[(idx*config.motion.h36m_input_length):((idx+1)*config.motion.h36m_input_length)],  dtype=torch.float32)
print(f"Test input shape: {test_input_.shape}")
ground_truth = torch.tensor(walking_sample[(idx+1)*config.motion.h36m_input_length:((idx+1)*config.motion.h36m_input_length + config.motion.h36m_target_length)], dtype=torch.float32)
realtime_predictor.batch_predict(test_input_, ground_truth, visualize, debug)
global_observed_motion = realtime_predictor.add_global_translation(passthrough=True)  # Add global translation to the predicted motion
visualize_continuous_motion(amass_sample, skeleton_type='incomplete_h36m', title="Ground Truth Motion", save_gif_path="gt_motion.gif")
# visualize_motion_with_ground_truth(realtime_predictor.predicted_motion, ground_truth, 
#                                    title="Predicted vs Ground Truth Motion",
#                                    skeleton_type='incomplete_h36m',)


