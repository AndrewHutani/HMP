import torch
import torch.nn as nn

import numpy as np 

from models.PhysMoP import PhysMoP
from models.humanmodel import SMPL, SMPLH

import utils.config as config
from utils.utils import remove_singlular_batch, smoothness_constraint, batch_roteulerSMPL, compute_errors, compute_error_accel_T

import utils.constants as constants
from dataclasses import dataclass

from visualize_motion import visualize_continuous_motion, visualize_motion_with_ground_truth

from dataset.action_aware_dataset import ActionAwareDataset, ActionSampler

from tqdm import tqdm

@dataclass
class BatchInfo:
    process_size_test: int
    test_batchsize: int
    gt_shape: torch.Tensor
    gt_gender_id: torch.Tensor
    gt_joints: torch.Tensor
    gt_joints_smpl: torch.Tensor


class RealtimePhysMop:
    def __init__(self, checkpoint_path, device='auto'):
        self.set_device(device)
             
        self.smpl = SMPL(device=self.device)
        self.smplh_m = SMPLH(gender='male', device=self.device)
        self.smplh_f = SMPLH(gender='female', device=self.device)

        self.model = PhysMoP(hist_length=config.hist_length,
                                       physics=True,
                                       data=True,
                                       fusion=True,
                                       device=self.device
                                       ).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()

        self.observed_motion = []
        self.predicted_motion = []
        self.ground_truth = [] 

    def set_device(self, device):
        # Device selection with availability checking
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is not available on this system")
            self.device = torch.device('cuda')
        elif device == 'mps':
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS device requested but MPS is not available on this system")
            self.device = torch.device('mps')
        elif device == 'cpu':
            self.device = torch.device('cpu')
        else:
            raise ValueError(f"Invalid device '{device}'. Valid options are: 'auto', 'cuda', 'mps', 'cpu'")
        
        print(f"Using device: {self.device}")


    def predict(self, data_test):
        """
        data_test: dict
            Contains the following keys:
            - 'q': Tensor of shape (Batch size, Frames = 50, 3) representing the joint angles
            - 'shape': Tensor of shape (Batch size, Frames, 10) representing SMPL shape parameters (?)
            - gender_id: Tensor of shape (Batch size, Frames) representing which gender to use
        """
        # Pad all time-dependent tensors to config.total_length at the front if needed
        pad_len = config.total_length - data_test['q'].shape[1]
        if pad_len > 0:
            for k in ['q', 'shape', 'gender_id']:
                v = data_test[k]
                pad_shape = list(v.shape)
                pad_shape[1] = pad_len
                pad_tensor = v[:, :1, ...].expand(*pad_shape)  # Repeat first frame
                data_test[k] = torch.cat([pad_tensor, v], dim=1)
        

        # Data Format Conversion & Extraction
        gt_q = data_test['q'].type(torch.float32)
        test_batchsize = gt_q.shape[0]
        process_size_test = config.total_length * test_batchsize

        # NxTx10 to N*Tx10
        gt_shape = data_test['shape'].type(torch.float32).view(process_size_test, 10)
        # NxTx1 to N*Tx1
        gt_gender_id = data_test['gender_id'].type(torch.float32).view(process_size_test)

        # Global Translation Normalization
        gt_q[:,:,:3] = gt_q[:,:,:3] - gt_q[:,0:1,:3]

        # Singular Batch Removal
        gt_q = remove_singlular_batch(gt_q)
        
        # Velocity and Acceleration Computation
        gt_q_dot, gt_q_ddot, _ = smoothness_constraint(gt_q, constants.dt)
        gt_q_ddot = gt_q_ddot[:, 1:]  # Remove first frame (no acceleration defined)

        #SMPL Pose Parameter Conversion + Forward Kinematics
        gt_pose = torch.zeros([test_batchsize, config.total_length, 72]).type(torch.float32).to(self.device)
        gt_pose[:, :, constants.G2Hpose_idx] = gt_q[:,:,3:].to(self.device)  # Map 60 dims to 72 SMPL dims
        gt_pose = gt_pose.view(process_size_test, 72)
        
        # Send all tensors to device
        gt_q = gt_q.to(self.device)
        gt_q_ddot = gt_q_ddot.to(self.device)
        gt_shape = gt_shape.to(self.device)
        gt_gender_id = gt_gender_id.to(self.device)
        gt_pose = gt_pose.to(self.device)

        gt_vertices, gt_joints, gt_joints_smpl, gt_rotMat_individual = self.forward_kinematics(gt_pose, gt_shape, gt_gender_id, process_size_test, joints_smpl=True, vertices=True)
        batch_info = BatchInfo(process_size_test, test_batchsize, gt_shape, gt_gender_id, gt_joints, gt_joints_smpl)
        gt_vertices_norm, gt_M_inv, gt_JcT = None, None, None

        # Predict future joint angle data
        # In this function, the first hist_length frames are used as input, and the remaining frames are predicted
        model_output = self.model.forward_dynamics(gt_vertices_norm, gt_q, gt_q_ddot, gt_M_inv, gt_JcT, self.device, mode='test')
        return model_output, batch_info

    
    def forward_kinematics(self, pose, shape, gender_id, process_size_test, vertices=False, joints_smpl=False):
        rotmat, rotMat_individual = batch_roteulerSMPL(pose)

        output_smplh_m = self.smplh_m.forward(betas=shape[gender_id==0], rotmat=rotmat[gender_id==0])
        output_smplh_f = self.smplh_f.forward(betas=shape[gender_id==1], rotmat=rotmat[gender_id==1])
        output_smpl = self.smpl.forward(betas=shape[gender_id==2], rotmat=rotmat[gender_id==2])

        if vertices:
            vertices = torch.zeros([process_size_test, 6890, 3]).float().to(self.device)
            vertices[gender_id==0] = output_smplh_m.vertices
            vertices[gender_id==1] = output_smplh_f.vertices
            vertices[gender_id==2] = output_smpl.vertices
        else:
            vertices = None

        joints = torch.zeros([process_size_test, 17, 3]).float().to(self.device)
        joints[gender_id==0] = output_smplh_m.joints[:, :17]
        joints[gender_id==1] = output_smplh_f.joints[:, :17]
        joints[gender_id==2] = output_smpl.joints[:, :17]

        if joints_smpl==True:
            joints_smpl = torch.zeros([process_size_test, 24, 3]).float().to(self.device)
            joints_smpl[gender_id==0] = output_smplh_m.joints_smpl[:, :24]
            joints_smpl[gender_id==1] = output_smplh_f.joints_smpl[:, :24]
            joints_smpl[gender_id==2] = output_smpl.joints_smpl[:, :24]
        else:
            joints_smpl = None

        return vertices, joints, joints_smpl, rotMat_individual

    def model_output_to_3D_joints(self, model_output, batch_info, mode='train'):
        # Unpack model output and batch info
        pred_q_data, pred_q_physics_gt, pred_q_physics_pred, pred_q_fusion, pred_q_ddot_data, pred_q_ddot_physics_gt, fusion_weight = model_output
        process_size_test = batch_info.process_size_test
        test_batchsize = batch_info.test_batchsize
        gt_shape = batch_info.gt_shape
        gt_gender_id = batch_info.gt_gender_id
        gt_joints = batch_info.gt_joints
        gt_joints_smpl = batch_info.gt_joints_smpl

        # data-driven
        pred_pose_data = torch.zeros([test_batchsize,config.total_length,constants.n_smplpose]).type(torch.float32).to(self.device)

        pred_pose_data[:,:,constants.G2Hpose_idx] = pred_q_data[:,:,3:]
        pred_pose_data = pred_pose_data.reshape([process_size_test, 72])
        _, pred_joints_data, pred_joints_smpl_data, _ = self.forward_kinematics(pred_pose_data, gt_shape, gt_gender_id, process_size_test, joints_smpl=True)

        # physics-gt
        pred_pose_physics_gt = torch.zeros([test_batchsize,config.total_length,constants.n_smplpose]).type(torch.float32).to(self.device)
        if mode == 'train':
            pred_pose_physics_gt[:,:,constants.G2Hpose_idx] = pred_q_physics_gt[:,:,3:]
        else:
            pred_pose_physics_gt[:,:,constants.G2Hpose_idx] = pred_q_physics_pred[:,:,3:]
        pred_pose_physics_gt = pred_pose_physics_gt.reshape([process_size_test, 72])
        _, pred_joints_physics_gt, pred_joints_smpl_physics_gt, _ = self.forward_kinematics(pred_pose_physics_gt, gt_shape, gt_gender_id, process_size_test, joints_smpl=True)

        # fusion
        pred_pose_fusion = torch.zeros([test_batchsize,config.total_length,constants.n_smplpose]).type(torch.float32).to(self.device)
        pred_pose_fusion[:,:,constants.G2Hpose_idx] = pred_q_fusion[:,:,3:]
        pred_pose_fusion = pred_pose_fusion.reshape([process_size_test, 72])
        pred_vertices_fusion, pred_joints_fusion, pred_joints_smpl_fusion, _ = self.forward_kinematics(pred_pose_fusion, gt_shape, gt_gender_id, process_size_test, joints_smpl=True, vertices=True)

        if config.test_mode == 'H36M':
            gt_J = torch.cat([gt_joints, gt_joints[:, 8:9], gt_joints[:, 8:9],
                            gt_joints[:, 13:14], gt_joints[:, 16:17],
                            gt_joints_smpl[:, 22:23], gt_joints_smpl[:, 22:23], gt_joints_smpl[:, 22:23], 
                            gt_joints_smpl[:, 23:], gt_joints_smpl[:, 23:], gt_joints_smpl[:, 23:],
                            ], dim=1)
            gt_J[:, [0,1,4]] = 0
            pred_J_data = torch.cat([pred_joints_data, pred_joints_data[:, 8:9], pred_joints_data[:, 8:9],
                            pred_joints_data[:, 13:14], pred_joints_data[:, 16:17],
                            pred_joints_smpl_data[:, 22:23], pred_joints_smpl_data[:, 22:23], pred_joints_smpl_data[:, 22:23], 
                            pred_joints_smpl_data[:, 23:], pred_joints_smpl_data[:, 23:], pred_joints_smpl_data[:, 23:],
                            ], dim=1)
            pred_J_data[:, [0,1,4]] = 0
            pred_J_physics_gt = torch.cat([pred_joints_physics_gt, pred_joints_physics_gt[:, 8:9], pred_joints_physics_gt[:, 8:9],
                                    pred_joints_physics_gt[:, 13:14], pred_joints_physics_gt[:, 16:17],
                                    pred_joints_smpl_physics_gt[:, 22:23], pred_joints_smpl_physics_gt[:, 22:23], pred_joints_smpl_physics_gt[:, 22:23], 
                                    pred_joints_smpl_physics_gt[:, 23:], pred_joints_smpl_physics_gt[:, 23:], pred_joints_smpl_physics_gt[:, 23:],
                                    ], dim=1)
            pred_J_physics_gt[:, [0,1,4]] = 0
            pred_J_fusion = torch.cat([pred_joints_fusion, pred_joints_fusion[:, 8:9], pred_joints_fusion[:, 8:9],
                                    pred_joints_fusion[:, 13:14], pred_joints_fusion[:, 16:17],
                                    pred_joints_smpl_fusion[:, 22:23], pred_joints_smpl_fusion[:, 22:23], pred_joints_smpl_fusion[:, 22:23], 
                                    pred_joints_smpl_fusion[:, 23:], pred_joints_smpl_fusion[:, 23:], pred_joints_smpl_fusion[:, 23:],
                                    ], dim=1)
            pred_J_fusion[:, [0,1,4]] = 0
        else:
            gt_J = gt_joints_smpl[:, 3:22]
            pred_J_data = pred_joints_smpl_data[:, 3:22]
            pred_J_physics_gt = pred_joints_smpl_physics_gt[:, 3:22]
            pred_J_fusion = pred_joints_smpl_fusion[:, 3:22]

        return gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion

    def evaluation_metrics(self, gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion):
        gt_J = gt_J.detach().cpu().numpy()
        pred_J_data = pred_J_data.detach().cpu().numpy()
        pred_J_physics_gt = pred_J_physics_gt.detach().cpu().numpy()
        pred_J_fusion = pred_J_fusion.detach().cpu().numpy()

        _, errors = compute_errors(gt_J.copy(), pred_J_data, 0)
        error_test_data = np.array(errors).reshape([-1, config.total_length])

        _, errors = compute_errors(gt_J.copy(), pred_J_physics_gt, 0)
        error_test_physics_gt = np.array(errors).reshape([-1, config.total_length])

        _, errors = compute_errors(gt_J.copy(), pred_J_fusion, 0)
        error_test_fusion = np.array(errors).reshape([-1, config.total_length])

        perjaccel = compute_error_accel_T(gt_J.copy(), pred_J_data, config.total_length, 0)
        accel_data = (np.array(perjaccel) * constants.m2mm)

        perjaccel = compute_error_accel_T(gt_J.copy(), pred_J_physics_gt, config.total_length, 0)
        accel_physics_gt = (np.array(perjaccel) * constants.m2mm)
        
        perjaccel = compute_error_accel_T(gt_J.copy(), pred_J_fusion, config.total_length, 0)
        accel_fusion = (np.array(perjaccel) * constants.m2mm)

        return error_test_data, error_test_physics_gt, error_test_fusion, accel_data, accel_physics_gt, accel_fusion

from torch.utils.data import DataLoader, Subset
from dataset.base_dataset_test import BaseDataset_test
import time

def get_specific_batch(data_loader, batch_index):
    """Get a specific batch by index without loading all batches into memory"""
    for i, batch in enumerate(data_loader):
        if i == batch_index:
            return batch
    return None

actions = [
    "walking", "eating", 
    "smoking", "discussion", "directions",
    "greeting", "phoning", "posing", "purchases", "sitting",
    "sittingdown", "takingphoto", "waiting", "walkingdog",
    "walkingtogether"
]
time_idx = [1, 3, 7, 9, 13, 17, 21, 24] # Corresponds to idx*40 ms in the future
selected_indices = [t + config.hist_length - 1 for t in time_idx]

ds = "H36M" 
if __name__ == "__main__":
    realtime_model = RealtimePhysMop('ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt', device='cpu')

    log_files = ["physmop_data_mpjpe_log.txt", "physmop_physics_mpjpe_log.txt", "physmop_fusion_mpjpe_log.txt"]
    log_file_handles = [open(f, "w") for f in log_files]  # Open all files once in write mode

    for action in actions:
        print(f"Evaluating action: {action}")
        dataset = ActionAwareDataset(
            'data/data_processed/h36m_test_50.pkl',
            specific_action=action
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        mpjpe_data_all = []
        mpjpe_physics_gt_all = []
        mpjpe_fusion_all = []
        accel_data_all = []
        accel_physics_gt_all = []
        accel_fusion_all = []

        for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f"{action} samples"):
            # Assume batch['q'] shape: (1, T, 3), T = total frames
            T = batch['q'].shape[1]
            mpjpe_data_per_obs = []
            mpjpe_physics_gt_per_obs = []
            mpjpe_fusion_per_obs = []
            accel_data_per_obs = []
            accel_physics_gt_per_obs = []
            accel_fusion_per_obs = []
            for i in range(config.hist_length):
                start_idx = 24 - i
                # Trim input to last obs_len frames
                batch_trimmed = {k: v[:, start_idx:config.total_length, ...] if isinstance(v, torch.Tensor) and v.shape[1] == config.total_length else v for k, v in batch.items()}
                model_output, batch_info = realtime_model.predict(batch_trimmed)
                gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion = realtime_model.model_output_to_3D_joints(model_output, batch_info, mode='test')
                # Compute MPJPE for this observation length (example: using pred_J_fusion)
                error_test_data, error_test_physics_gt, error_test_fusion, accel_data, accel_physics_gt, accel_fusion = realtime_model.evaluation_metrics(gt_J, pred_J_data, pred_J_physics_gt, pred_J_fusion)

                mpjpe_data_per_obs.append(error_test_data[0, selected_indices]) 
                mpjpe_physics_gt_per_obs.append(error_test_physics_gt[0, selected_indices])
                mpjpe_fusion_per_obs.append(error_test_fusion[0, selected_indices])
                # accel_data_per_obs.append(accel_data[0, selected_indices])
                # accel_physics_gt_per_obs.append(accel_physics_gt[0, selected_indices])
                # accel_fusion_per_obs.append(accel_fusion[0, selected_indices])

            mpjpe_data_all.append(mpjpe_data_per_obs) 
            mpjpe_physics_gt_all.append(mpjpe_physics_gt_per_obs)
            mpjpe_fusion_all.append(mpjpe_fusion_per_obs)
            # accel_data_all.append(accel_data_per_obs)
            # accel_physics_gt_all.append(accel_physics_gt_per_obs)
            # accel_fusion_all.append(accel_fusion_per_obs)

        mpjpe_data_all = np.array(mpjpe_data_all)  # shape: (num_samples, obs_len, 8)
        mpjpe_physics_gt_all = np.array(mpjpe_physics_gt_all)  # shape: (num_samples, obs_len, 8)
        mpjpe_fusion_all = np.array(mpjpe_fusion_all)  # shape: (num_samples, obs_len, 8)

#         mpjpe_mean = np.mean(mpjpe_all_samples, axis=0)  # shape: (obs_len,)
        log_files = ["physmop_data_mpjpe_log.txt", "physmop_physics_mpjpe_log.txt", "physmop_fusion_mpjpe_log.txt"]
        for i, log_file in enumerate(log_file_handles):
            if i == 0:
                mjpe_mean = np.mean(mpjpe_data_all, axis=0) # shape: (obs_len, 8)
            elif i == 1:
                mjpe_mean = np.mean(mpjpe_physics_gt_all, axis=0)  # shape: (obs_len, 8)
            else:
                mjpe_mean = np.mean(mpjpe_fusion_all, axis=0)  # shape: (obs_len, 8)

            # Write to log file

            log_file.write(f"Averaged MPJPE for each observation length and each selected timestep: {action}\n")
            for obs_len in range(mjpe_mean.shape[0]):
                log_file.write(f"Obs {obs_len+1}: {mjpe_mean[obs_len]}\n")
            log_file.write("\n")

    # Close all log files at the end
    for log_file in log_file_handles:
        log_file.close()
