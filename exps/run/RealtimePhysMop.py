import torch
import torch.nn as nn

import numpy as np 

from PhysMoP import PhysMoP
import utils.config as config
from utils.utils import remove_singlular_batch, smoothness_constraint, batch_roteulerSMPL

import utils.constants as constants


class RealtimePhysMop:
    def __init__(self, checkpoint_path,):
        self.model = PhysMoP(hist_length=config.hist_length,
                                       physics=True,
                                       data=True,
                                       fusion=True
                                       ).to(self.device)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()

    def predict(self, data_test):
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
        gt_pose[:, :, constants.G2Hpose_idx] = gt_q[:,:,3:]  # Map 60 dims to 72 SMPL dims

        gt_vertices, gt_joints, gt_joints_smpl, gt_rotMat_individual = self.forward_kinematics(gt_pose, gt_shape, gt_gender_id, process_size_test, joints_smpl=True, vertices=True)
        gt_vertices_norm, gt_M_inv, gt_JcT = None, None, None

        model_output = self.model.forward_dynamics(gt_vertices_norm, gt_q, gt_q_ddot, gt_M_inv, gt_JcT, self.device, mode='test')
    
    
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

    def post_process_output(self):
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

    def evaluation_metrics(self):
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