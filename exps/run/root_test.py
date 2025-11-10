import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from gcnext_model.config import config
import argparse
from datasets.h36m_eval import H36MEval


config.motion.h36m_target_length = config.motion.h36m_target_length_eval
dataset = H36MEval(config, 'test')
walking_sample, root_sample = dataset.get_full_sequences_for_action("walking")[0]

# Suppose root_translation is your tensor of shape (num_frames, 32, 3)
# Get the root translation for each frame (just take the first joint, since all are the same)
root_traj = root_sample[:, 0, :].cpu().numpy()/1000.  # shape: (num_frames, 3)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(root_traj[:, 0], root_traj[:, 1], root_traj[:, 2], label='Root trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Root Translation Trajectory')
ax.legend()
plt.show()