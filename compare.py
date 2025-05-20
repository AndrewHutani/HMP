import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from matplotlib import animation

# Load predictions
test_predictions = np.load("first_sample_pred.npy")
realtime_predictions = np.load("realtime_predictions.npy")

print("test_predictions is shape: {} \n realtime_predictions is shape: {}".format(test_predictions.shape, realtime_predictions.shape))

# Ensure shapes match
assert test_predictions.shape == realtime_predictions.shape, "Shapes of predictions do not match!\n"

# Compute Mean Per Joint Position Error (MPJPE)
mpjpe = np.mean(np.linalg.norm(test_predictions - realtime_predictions, axis=-1))
print(f"MPJPE between test.py and realtime.py predictions: {mpjpe:.2f} mm")

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale without making the limits the same."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_motion_with_ground_truth(realtime_positions, offline_positions, time_steps, title="Predicted vs Ground Truth Motion", save_path=None):
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    predicted_connections = [
        (2, 3), (3, 4), (4, 5),
        (7, 8), (8, 9), (9, 10),
        (12, 13), (13, 14), (14, 15),
        (17, 18), (18, 19), (21, 22),
        (25, 26), (26, 27), (29, 30)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=55)
    ax.set_title(title)

    def update(frame_idx):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{title} - Time Step #{frame_idx}")

        predicted_joints = realtime_positions[frame_idx]
        ground_truth_joints = offline_positions[frame_idx]

        for connection in predicted_connections:
            joint1, joint2 = connection
            # ax.plot([predicted_joints[joint1, 0], predicted_joints[joint2, 0]],
            #         [predicted_joints[joint1, 2], predicted_joints[joint2, 2]],
            #         [predicted_joints[joint1, 1], predicted_joints[joint2, 1]], 'r', alpha=0.5)
            ax.plot([ground_truth_joints[joint1, 0], ground_truth_joints[joint2, 0]],
                    [ground_truth_joints[joint1, 2], ground_truth_joints[joint2, 2]],
                    [ground_truth_joints[joint1, 1], ground_truth_joints[joint2, 1]], c='b')
        ax.scatter([], [], [], c='r', marker='o', label='Real time predictions')
        ax.scatter([], [], [], c='b', marker='^', label='Offline predictions')
        ax.legend()
        set_axes_equal(ax)

    anim = animation.FuncAnimation(fig, update, frames=time_steps, interval=100)

    if save_path:
        # Save as MP4 (requires ffmpeg) or GIF (requires pillow)
        if save_path.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=10)
            anim.save(save_path, writer=writer)
        elif save_path.endswith('.gif'):
            writer = animation.PillowWriter(fps=10)
            anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

visualize_motion_with_ground_truth(
    realtime_positions=realtime_predictions,
    offline_positions=test_predictions,
    time_steps=np.arange(0, 25, 1),  # Time steps to visualize
    title="Predicted Motion vs Ground Truth",
    save_path="motion_comparison.mp4"  # Save as MP4
)
# # Visualize a specific frame for comparison in 3D
# frame_idx = 0  # Choose a frame to visualize
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

# # Plot test predictions
# ax.scatter(
#     test_predictions[frame_idx, :, 0],
#     test_predictions[frame_idx, :, 2],
#     test_predictions[frame_idx, :, 1],
#     label="Test Predictions",
#     c='r'
# )

# # Plot realtime predictions
# ax.scatter(
#     realtime_predictions[frame_idx, :, 0],
#     realtime_predictions[frame_idx, :, 2],
#     realtime_predictions[frame_idx, :, 1],
#     label="Realtime Predictions",
#     c='b'
# )

# # Set labels and title
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title(f"Frame {frame_idx} Comparison (3D)")
# ax.legend()

# # Show the plot
# plt.show()