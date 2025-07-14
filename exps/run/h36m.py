import os
import os.path as osp
from tqdm import tqdm

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter

from utils.utils import batch_rodrigues, rotmat2eulerzyx, rotmat2euleryzx, rotmat2eulerzxy, map_h36m_to_amass

device = torch.device('cpu')

h36m_splits = {
    'train': [
        'S1',
        'S6',
        'S7',
        'S8',
        'S9'
    ],
    'test': [
        'S5',
    ],
}

def read_data(trunk_datafolder, trunk_path, folder, sequences, seqlen, overlap=0.75, fps=30):

    for seq_name in sequences:
        print(f'Reading {seq_name} sequence...')
        seq_folder = osp.join(folder, seq_name)

        read_sequence(trunk_datafolder, trunk_path, seq_folder, seq_name, fps, seqlen, overlap)

    return trunk_path

def read_sequence(trunk_datafolder, trunk_path, folder, seq_name, fps, seqlen, overlap):
    # List all .txt files in the sequence folder
    files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    print(files)
    for file in tqdm(files, desc=f"Processing {seq_name}"):
        file_path = os.path.join(folder, file)
        # Load the sequence: shape (num_frames, num_joints*3)
        data = np.loadtxt(file_path, delimiter=',')

        mocap_framerate = 50 # Frquency of H36M is 50Hz
        sampling_freq = mocap_framerate // fps # Downsample to the desired fps (should be 25?)
        n_frames, n_feats = data.shape
        num_joints = n_feats // 3

        # pose       
        # Select joints
        # This is awful, we need the following joints: 
            # 0: pelvis, 1: left_hip, 2: right_hip, 3: spine1, 4: left_knee, 5: right_knee, 6: spine2, 
            # 7: left_ankle, 8: right_ankle, 9: spine3, 10: left_foot, 11: right_foot, 12: neck, 
            # 13: left_collar, 14: right_collar, 15: head, 16: left_shoulder, 17: right_shoulder, 
            # 18: left_elbow, 19: right_elbow, 20: left_wrist, 21: right_wrist
        sampling_freq = mocap_framerate // fps
        pose = data[::sampling_freq]
        pose = np.reshape(pose, (pose.shape[0], -1, 3))  # (sampling_freq, num_joints, 3)
        pose = map_h36m_to_amass(pose)  # shape (frames, 22, 3)
        # pose[:, 0, :] = 0  # Zero out root orientation
        pose = torch.from_numpy(pose).type(torch.float32).to(device)
        # print(f'Pose shape after selecting joints: {pose.shape}')
        rotmat = batch_rodrigues(pose.reshape(-1,1,3)).view(-1,22,3,3)

        # 2. Convert to Euler angles (same as AMASS)
        euler_root = rotmat2euleryzx(rotmat[:,:1,:,:].clone().view(-1,3,3)).view(-1,1,3)
        euler_s = rotmat2eulerzyx(rotmat[:,1:16,:,:].clone().view(-1,3,3)).view(-1,15,3)
        euler_shoulder = rotmat2eulerzxy(rotmat[:,16:18,:,:].clone().view(-1,3,3)).view(-1,2,3)
        euler_elbow = rotmat2euleryzx(rotmat[:,18:20,:,:].clone().view(-1,3,3)).view(-1,2,3)
        euler_e = rotmat2eulerzyx(rotmat[:,20:,:,:].clone().view(-1,3,3)).view(-1,2,3)
        euler = torch.cat((euler_root,euler_s,euler_shoulder,euler_elbow,euler_e), dim=1).detach().numpy()
        euler = np.delete(euler, [9,10], 1).reshape([-1,20*3])
        euler[:, :3] = 0
        
        # 3. Smooth Euler angles
        euler_smooth = euler.copy()
        for angle in range(euler.shape[1]):
            for t in range(euler.shape[0]-1):
                while np.abs(euler[t+1,angle]-euler[t,angle])>np.pi:
                    if euler[t+1,angle]<euler[t,angle]:
                        euler[t+1,angle] = euler[t+1,angle] + 2*np.pi
                    else:
                        euler[t+1,angle] = euler[t+1,angle] - 2*np.pi
            euler_smooth[:,angle] = median_filter(euler[:,angle], smooth_sigma//2)
            euler_smooth[:,angle] = gaussian_filter1d(euler_smooth[:,angle].copy(), smooth_sigma) 

        # 4. Translation (use root joint, convert mm to meters if needed)
        trans = data[:, :3] / 1000.0  # (n_frames, 3)
        for i in range(3):
            trans[:, i] = median_filter(trans[:, i], smooth_sigma_va//2)
            trans[:, i] = gaussian_filter1d(trans[:, i], smooth_sigma_va)

        # 5. Concatenate and wrap angles
        q = np.concatenate([trans, euler_smooth], axis=1)[15:-15]
        q[:,3:] = np.remainder(q[:,3:], 2*np.pi)
        q_pose = q[:,3:].copy()
        q_pose[q_pose>np.pi] = q_pose[q_pose>np.pi] - 2*np.pi
        q[:,3:] = q_pose

        # 6. Shape and gender
        shape = np.zeros((q.shape[0], 10), dtype=np.float32)  # average shape
        gender_id = np.zeros((q.shape[0], 1), dtype=np.float32)  # gender_id = 0

        data_i = np.concatenate([q, shape, gender_id], axis=1)

        vid_name = f'{seq_name}_{file[:-4]}'  # Use file name instead of subject/action for H36M

        n_frames = data_i.shape[0]
        for frame in range(0, n_frames-seqlen, int(seqlen * (1-overlap))):
            trunk_path_f = trunk_datafolder + '/%s_vid_%d' % (vid_name, frame)
            np.save(trunk_path_f, data_i[frame:frame+seqlen])
            trunk_path.append(trunk_path_f)

    return trunk_path

import pickle

h36m_dir = 'data/h36m'
save_dir = 'data/data_processed'
seqlen = 50

smooth_sigma = 6
smooth_sigma_va = 8
    
for split in ['train', 'test']:
    if split == 'train':
        overlap = 9/10
    else:
        overlap = 0
    trunk_datafolder = osp.join(h36m_dir, split, str(seqlen))
    if not osp.exists(trunk_datafolder):
        os.makedirs(trunk_datafolder)
        
    trunk_path = []
    data = read_data(trunk_datafolder, trunk_path, h36m_dir, sequences=h36m_splits[split], seqlen=seqlen, overlap=overlap)

    data_file = osp.join(save_dir, f'h36m_{split}_%d.pkl' % seqlen)
    
    print(f'Saving h36m dataset to {data_file} with total number of data %d' % len(data))
    pickle.dump(data, open(data_file, 'wb'))