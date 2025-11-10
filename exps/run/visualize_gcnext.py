import torch
import numpy as np
import argparse
import os

from tqdm import tqdm
from gcnext_model.config import config

import random

from gcnext_model.RealtimeGCNext import RealtimeGCNext

from model import GCNext as Model
from datasets.h36m_eval import H36MEval
from visualize_motion import visualize_input_and_output_gcnext


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="ckpt/baseline/model-iter-84000.pth", help='=encoder path')
parser.add_argument('--dyna', nargs='+', type=int, default=[0, 48], help='dynamic layer index')
args = parser.parse_args()

# Prepare model
model = Model(config, args.dyna)
state_dict = torch.load(args.model_pth)
model.load_state_dict(state_dict, strict=True)


actions = ["walking", "eating", 
           "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"
            ]

for action in actions:
    # Ensure directory for visualizations exists
    os.makedirs(f"motion_examples/gcnext/{action}", exist_ok=True)

    print(f"Visualizing action: {action}")
    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    dataset = H36MEval(config, 'test')
    action_indices = dataset.get_indices_for_action(action)

    # Select a random index per action
    random_idx = random.choice(action_indices)

    realtime_predictor = RealtimeGCNext(model, config, tau=0.5)
    visualize = False
    debug = False

    with torch.no_grad():
        test_input, test_output = dataset.__getitem__(random_idx)
        full_motion = torch.cat([test_input, test_output], dim=0)
        save_intervals = save_intervals = [0, 1, 2, 3, 4, 9, 14, 19, 29, 39, 49]
        for i in save_intervals:
            test_input_ = test_input[:i+1]
            ground_truth = full_motion[i+1:i+1+config.motion.h36m_target_length_eval, :, :]
            motion_input = realtime_predictor.batch_predict(test_input_, ground_truth, visualize, debug)
            if isinstance(motion_input, torch.Tensor):
                motion_input = motion_input.cpu().numpy()
            elif isinstance(motion_input, list) and isinstance(motion_input[0], torch.Tensor):
                motion_input = np.stack([t.cpu().numpy() for t in motion_input], axis=0)
            visualize_input_and_output_gcnext(  motion_input[:, realtime_predictor.joint_used_xyz.astype(int)],
                                                realtime_predictor.predicted_motion[:, realtime_predictor.joint_used_xyz.astype(int)],
                                                ground_truth.cpu().numpy()[:, realtime_predictor.joint_used_xyz.astype(int)],
                                                skeleton_type='incomplete_h36m',
                                                save_mp4_path=f"motion_examples/gcnext/{action}/{i+1}_input_frames.mp4"
            )
