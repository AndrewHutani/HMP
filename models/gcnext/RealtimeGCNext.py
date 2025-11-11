import torch
import numpy as np
import argparse
from models.gcnext.config import config

from models.gcnext.model import GCNext as Model
import matplotlib.pyplot as plt

import time

from models.prediction_times import prediction_times, single_forward_pass_times

class RealtimeGCNext():
    def __init__(self, model, config, tau):
        self.config = config
        self.tau = tau

        self.model = model
        self.model.eval()
        self.device = torch.device("cpu")
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


    def get_dct_matrix(self, N):
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        return torch.tensor(dct_m).float().to(self.device), torch.tensor(idct_m).float().to(self.device)

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
        input_length = self.config.motion.h36m_input_length_dct
        if debug:
            print("Stacked observed motion shape:", torch.stack(self.observed_motion).shape)
        observed_motion = torch.stack(self.observed_motion).to(self.device)
        n, c, _ = observed_motion.shape  # n: number of timesteps, c: number of joints

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
        
        t0 = time.perf_counter()
        for idx in range(num_prediction_chunks):
            t0_chunk = time.perf_counter()
            if c == 22: 
                # This means that the input is already in the correct format
                motion_input = motion_window.reshape(1, input_length, -1)
            else:
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
                if c == 22:
                    # If the input is already in the correct format, we just need to fill in the joints
                    motion_pred = output
                else:
                    motion_pred = motion_window[-1].unsqueeze(0).repeat(chunk_prediction_length, 1, 1)
                    motion_pred[:, self.joint_used_xyz, :] = output
                    motion_pred[:, self.joint_to_ignore, :] = motion_pred[:, self.joint_equal, :]

                outputs.append(motion_pred)

                # Slide the window: remove first 'step' frames, append new predictions
                motion_window = torch.cat([motion_window[chunk_prediction_length:], motion_pred], dim=0)
            t1_chunk = time.perf_counter()
            single_forward_pass_times.append(t1_chunk - t0_chunk)
        t1 = time.perf_counter()
        prediction_times.append(t1 - t0)
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
        
        # t1 = time.time()
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
        lower_body_indices =[0, 1, 2, 3, 4, 5, 6, 7]
        upper_body_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        predicted_upper = self.predicted_motion[:, upper_body_indices] * 1000
        predicted_lower = self.predicted_motion[:, lower_body_indices] * 1000
        ground_truth_upper = self.ground_truth[:, upper_body_indices] * 1000
        ground_truth_lower = self.ground_truth[:, lower_body_indices] * 1000

        mpjpe_upper = np.mean(np.linalg.norm(predicted_upper - ground_truth_upper, axis=2), axis=1)
        mpjpe_lower = np.mean(np.linalg.norm(predicted_lower - ground_truth_lower, axis=2), axis=1)

        selected_timesteps = [1, 9, 14, 24]
        return mpjpe_upper[selected_timesteps], mpjpe_lower[selected_timesteps]