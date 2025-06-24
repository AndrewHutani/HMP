import torch
import numpy as np

from scipy.spatial.transform import Rotation as R
from config import config
from utils.angle_to_joint import ang2joint

# from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp

import matplotlib.pyplot as plt
def visualize_continuous_motion(motion_sequence, title="Continuous Motion Visualization"):
    """
    Visualize a continuous motion sequence in 3D.

    :param motion_sequence: Numpy array of shape [num_frames, num_joints, 3] (motion sequence).
    :param title: Title of the plot.
    """
    axes_limit = 1
    # Define the connections between joints (skeleton structure)
    connections = [
        (2, 1), (1, 4), (4, 7),
        (2, 0), (0, 3), (3, 6),
        (2, 5), (5, 8), (8, 11),
        (5, 10), (10, 13), (13, 15), (15, 17),
        (5, 9), (9, 12), (12, 14), (14, 16)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    for frame_idx in range(motion_sequence.shape[0]):
        ax.clear()
        ax.set_xlim([-axes_limit, axes_limit])
        ax.set_ylim([-axes_limit, axes_limit])
        ax.set_zlim([0, 2*axes_limit])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        joints = motion_sequence[frame_idx]

        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', marker='o')


        for connection in connections:
            joint1, joint2 = connection
            ax.plot([joints[joint1, 0], joints[joint2, 0]],
                    [joints[joint1, 1], joints[joint2, 1]],
                    [joints[joint1, 2], joints[joint2, 2]], 'r', alpha=0.5)

        plt.pause(0.05)  # Adjust the pause duration for smoother animation

    plt.show()

def load_skeleton(skeleton_path):
    skeleton_info = np.load(skeleton_path)
    p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
    parents = skeleton_info['parents']
    parent = {}
    for i in range(len(parents)):
        parent[i] = parents[i]
    
    return p3d0, parent

support_dir = 'data/amass/'

# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

amass_npz_fname = osp.join(support_dir, '20_13_poses.npz') # the path to body data
bdata = np.load(amass_npz_fname)

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = bdata['gender']

print('Data keys available:%s'%list(bdata.keys()))

print('The subject of the mocap sequence is  {}.'.format(subject_gender))

num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters
time_length = len(bdata['trans'])

body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
    'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
}

print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
print('time_length = {}'.format(time_length))
print("Root orientation: ", body_parms['root_orient'].shape)
print("Body translation: ", body_parms['trans'].shape)

pose_body_numpy = bdata['poses'] # 156 joints ????
N = len(pose_body_numpy)
print(N)

amass_motion_poses = R.from_rotvec(pose_body_numpy.reshape(-1, 3)).as_rotvec()
print(amass_motion_poses.shape)
amass_motion_poses = amass_motion_poses.reshape(N, 52, 3)
# amass_motion_poses[:, 0] = 0

p3d0, parents = load_skeleton(osp.join('body_models', 'smpl_skeleton.npz'))
print("skeleton shape: ", p3d0.shape)

p3d0_tmp = p3d0.repeat([amass_motion_poses.shape[0], 1, 1])
amass_motion_poses = ang2joint(p3d0_tmp, torch.tensor(amass_motion_poses).float(), parents).reshape(-1, 52, 3)[:, 4:22].reshape(N, -1)
print(amass_motion_poses.shape)
amass_motion_poses = amass_motion_poses.cpu().numpy()
amass_motion_poses = amass_motion_poses.reshape(N, -1, 3)
print(amass_motion_poses[29, 5, :])

# Add body translation to each joint for each timestep
amass_motion_poses_translated = amass_motion_poses + body_parms['trans'].cpu().numpy().reshape(N, 1, 3)

visualize_continuous_motion(amass_motion_poses_translated, title="AMASS Motion Visualization")