import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

from config import config
from datasets.h36m_eval import H36MEval
import torch

def visualize_continuous_motion(motion_sequence, title="Continuous Motion Visualization"):
    """
    Visualize a continuous motion sequence in 3D.

    :param motion_sequence: Numpy array of shape [num_frames, num_joints, 3] (motion sequence).
    :param title: Title of the plot.
    """
    axes_limit = 5
    # Define the connections between joints (skeleton structure)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (11, 12), (12, 13), (13, 14), (14, 15),
        (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    for frame_idx in range(motion_sequence.shape[0]):
        ax.clear()
        ax.set_xlim([-axes_limit, axes_limit])
        ax.set_ylim([-axes_limit, axes_limit])
        ax.set_zlim([0, axes_limit])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        joints = motion_sequence[frame_idx]
        for connection in connections:
            joint1, joint2 = connection
            ax.plot([joints[joint1, 0], joints[joint2, 0]],
                    [joints[joint1, 2], joints[joint2, 2]],
                    [joints[joint1, 1], joints[joint2, 1]], 'r', alpha=0.5)

        plt.pause(0.05)  # Adjust the pause duration for smoother animation

    plt.show()

config.motion.h36m_target_length = config.motion.h36m_target_length_eval
dataset = H36MEval(config, 'test')
# print(np.shape(torch.Tensor.numpy(dataset.h36m_seqs[0])))
visualize_continuous_motion(torch.Tensor.numpy(dataset.h36m_seqs[0]/1000.), title="Continuous Motion Visualization")