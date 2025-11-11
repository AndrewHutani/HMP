import numpy as np
import matplotlib.pyplot as plt

from exps.C_temporal_resolution.utils_resample_eval import compute_confidence_interval, export_mean_error_resample_rate_csv

save_directory = "exps/C_temporal_resolution/performance_logs/"
figure_directory = "exps/C_temporal_resolution/figures/"
downsample_rates = np.arange(3, 0.1, -0.2)

# ---------------------------CONSISTENT OUTPUT EVALUATION------------------------------
mpjpe_data_all = np.loadtxt(save_directory+"resampled_physmop_data_consistent_output_mean.csv", delimiter=",")
mpjpe_physics_gt_all = np.loadtxt(save_directory+"resampled_physmop_physics_consistent_output_mean.csv", delimiter=",")
mpjpe_fusion_all = np.loadtxt(save_directory+"resampled_physmop_fusion_consistent_output_mean.csv", delimiter=",")

mpjpe_data_std_all = np.loadtxt(save_directory+"resampled_physmop_data_consistent_output_std.csv", delimiter=",")
mpjpe_physics_gt_std_all = np.loadtxt(save_directory+"resampled_physmop_physics_consistent_output_std.csv", delimiter=",")
mpjpe_fusion_std_all = np.loadtxt(save_directory+"resampled_physmop_fusion_consistent_output_std.csv", delimiter=",")

mjpe_gcnext_all = np.loadtxt(save_directory+"resampled_gcn_consistent_output_gcn_mean.csv", delimiter=",")
mjpe_gcnext_std_all = np.loadtxt(save_directory+"resampled_gcn_consistent_output_gcn_std.csv", delimiter=",")

# Seperate mean and n_predictions for all logs
mpjpe_data_all, data_n_predictions = mpjpe_data_all[:, :-1], mpjpe_data_all[:, -1]
data_mean, data_confidence_interval = compute_confidence_interval(mpjpe_data_all, mpjpe_data_std_all, data_n_predictions)

mpjpe_physics_gt_all, physics_n_predictions = mpjpe_physics_gt_all[:, :-1], mpjpe_physics_gt_all[:, -1]
physics_mean, physics_confidence_interval = compute_confidence_interval(mpjpe_physics_gt_all, mpjpe_physics_gt_std_all, physics_n_predictions)

mpjpe_fusion_all, fusion_n_predictions = mpjpe_fusion_all[:, :-1], mpjpe_fusion_all[:, -1]
fusion_mean, fusion_confidence_interval = compute_confidence_interval(mpjpe_fusion_all, mpjpe_fusion_std_all, fusion_n_predictions)

mjpe_gcnext_all, gcnext_n_predictions = mjpe_gcnext_all[:, :-1], mjpe_gcnext_all[:, -1]
gcnext_mean, gcnext_confidence_interval = compute_confidence_interval(mjpe_gcnext_all, mjpe_gcnext_std_all, gcnext_n_predictions)

# Export CSVs for easy table generation
export_mean_error_resample_rate_csv(
    data_mean, data_confidence_interval,
    save_directory+"resampled_physmop_data_matching_output_mean_with_error.csv",
    resample_rates=downsample_rates,
    time_labels=["80ms", "400ms", "560ms", "1000ms"]
)
export_mean_error_resample_rate_csv(
    physics_mean, physics_confidence_interval,
    save_directory+"resampled_physmop_physics_matching_output_mean_with_error.csv",
    resample_rates=downsample_rates,
    time_labels=["80ms", "400ms", "560ms", "1000ms"]
)
export_mean_error_resample_rate_csv(
    gcnext_mean, gcnext_confidence_interval,
    save_directory+"resampled_gcn_matching_output_gcn_mean_with_error.csv",
    resample_rates=downsample_rates,
    time_labels=["80ms", "400ms", "560ms", "1000ms"]
)

branch_results = {
    "data": data_mean,
    "physics_gt": physics_mean,
    "fusion": fusion_mean,
    "gcnext": gcnext_mean
}
branch_titles = {
    "data": "MPJPE for the data branch vs Downsample Rate",
    "physics_gt": "MPJPE for the physics branch vs Downsample Rate",
    "fusion": "MPJPE for the fusion branch vs Downsample Rate",
    "gcnext": "MPJPE for the GCNext model vs Downsample Rate"
}
branch_filenames = {
    "data": "mpjpe_vs_downsample_rate_data.png",
    "physics_gt": "mpjpe_vs_downsample_rate_physics.png",
    "fusion": "mpjpe_vs_downsample_rate_fusion.png",
    "gcnext": "mpjpe_vs_downsample_rate_gcnext.png"
}

all_arrays = [
data_mean,
physics_mean,
fusion_mean,
gcnext_mean
]

all_data = np.concatenate([arr.flatten() for arr in all_arrays if arr is not None])
y_min = np.min(all_data[all_data > 0])  # Avoid zero for log scale
y_max = np.percentile(all_data, 100)
y_limits = (y_min, y_max)

for branch, results in branch_results.items():
    plt.figure()
    for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
        plt.plot(downsample_rates, results[:, i], marker='o', label=label)
    plt.xlabel(r'Resample rate $\alpha$', fontsize=16)
    plt.ylabel(r'Mean MPJPE (mm)', fontsize=16)
    plt.title(branch_titles[branch], fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.gca().invert_xaxis()
    plt.grid(True)
    # plt.legend(title='Prediction horizon')
    plt.ylim(y_limits)
    plt.savefig(figure_directory+branch_filenames[branch] + "_consistent_output.png")
    plt.close()



# ---------------------------MATCHING OUTPUT EVALUATION------------------------------
mpjpe_data_all = np.loadtxt(save_directory+"resampled_physmop_data_matching_output_mean.csv", delimiter=",")
mpjpe_physics_gt_all = np.loadtxt(save_directory+"resampled_physmop_physics_matching_output_mean.csv", delimiter=",")
mpjpe_fusion_all = np.loadtxt(save_directory+"resampled_physmop_fusion_matching_output_mean.csv", delimiter=",")

mpjpe_data_std_all = np.loadtxt(save_directory+"resampled_physmop_data_matching_output_std.csv", delimiter=",")
mpjpe_physics_gt_std_all = np.loadtxt(save_directory+"resampled_physmop_physics_matching_output_std.csv", delimiter=",")
mpjpe_fusion_std_all = np.loadtxt(save_directory+"resampled_physmop_fusion_matching_output_std.csv", delimiter=",")

mjpe_gcnext_all = np.loadtxt(save_directory+"resampled_gcn_matching_output_gcn_mean.csv", delimiter=",")
mjpe_gcnext_std_all = np.loadtxt(save_directory+"resampled_gcn_matching_output_gcn_std.csv", delimiter=",")

# Seperate mean and n_predictions for all logs
mpjpe_data_all, data_n_predictions = mpjpe_data_all[:, :-1], mpjpe_data_all[:, -1]
data_mean, data_confidence_interval = compute_confidence_interval(mpjpe_data_all, mpjpe_data_std_all, data_n_predictions)

mpjpe_physics_gt_all, physics_n_predictions = mpjpe_physics_gt_all[:, :-1], mpjpe_physics_gt_all[:, -1]
physics_mean, physics_confidence_interval = compute_confidence_interval(mpjpe_physics_gt_all, mpjpe_physics_gt_std_all, physics_n_predictions)

mpjpe_fusion_all, fusion_n_predictions = mpjpe_fusion_all[:, :-1], mpjpe_fusion_all[:, -1]
fusion_mean, fusion_confidence_interval = compute_confidence_interval(mpjpe_fusion_all, mpjpe_fusion_std_all, fusion_n_predictions)

mjpe_gcnext_all, gcnext_n_predictions = mjpe_gcnext_all[:, :-1], mjpe_gcnext_all[:, -1]
gcnext_mean, gcnext_confidence_interval = compute_confidence_interval(mjpe_gcnext_all, mjpe_gcnext_std_all, gcnext_n_predictions)

# Export CSVs for easy table generation
export_mean_error_resample_rate_csv(
    data_mean, data_confidence_interval,
    save_directory+"resampled_physmop_data_matching_output_mean_with_error.csv",
    resample_rates=downsample_rates,
    time_labels=["80ms", "400ms", "560ms", "1000ms"]
)
export_mean_error_resample_rate_csv(
    physics_mean, physics_confidence_interval,
    save_directory+"resampled_physmop_physics_matching_output_mean_with_error.csv",
    resample_rates=downsample_rates,
    time_labels=["80ms", "400ms", "560ms", "1000ms"]
)
export_mean_error_resample_rate_csv(
    gcnext_mean, gcnext_confidence_interval,
    save_directory+"resampled_gcn_matching_output_gcn_mean_with_error.csv",
    resample_rates=downsample_rates,
    time_labels=["80ms", "400ms", "560ms", "1000ms"]
)

branch_results = {
    "data": data_mean,
    "physics_gt": physics_mean,
    "fusion": fusion_mean,
    "gcnext": gcnext_mean
}
branch_titles = {
    "data": "MPJPE for the data branch vs Downsample Rate",
    "physics_gt": "MPJPE for the physics branch vs Downsample Rate",
    "fusion": "MPJPE for the fusion branch vs Downsample Rate",
    "gcnext": "MPJPE for the GCNext model vs Downsample Rate"
}
branch_filenames = {
    "data": "mpjpe_vs_downsample_rate_data.png",
    "physics_gt": "mpjpe_vs_downsample_rate_physics.png",
    "fusion": "mpjpe_vs_downsample_rate_fusion.png",
    "gcnext": "mpjpe_vs_downsample_rate_gcnext.png"
}

all_arrays = [
data_mean,
physics_mean,
fusion_mean,
gcnext_mean
]

all_data = np.concatenate([arr.flatten() for arr in all_arrays if arr is not None])
y_min = np.min(all_data[all_data > 0])  # Avoid zero for log scale
y_max = np.percentile(all_data, 100)
y_limits = (y_min, y_max)

for branch, results in branch_results.items():
    plt.figure()
    for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
        plt.plot(downsample_rates, results[:, i], marker='o', label=label)
    plt.xlabel(r'Resample rate $\alpha$', fontsize=16)
    plt.ylabel(r'Mean MPJPE (mm)', fontsize=16)
    plt.title(branch_titles[branch], fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.gca().invert_xaxis()
    plt.grid(True)
    # plt.legend(title='Prediction horizon')
    plt.ylim(y_limits)
    plt.savefig(figure_directory+branch_filenames[branch]+ "_matching_output.png")
    plt.close()

labels = ["80ms", "400ms", "560ms", "1000ms"]
markers = ['o', 'o', 'o', 'o']
colors = ['C0', 'C1', 'C2', 'C3']

fig, ax = plt.subplots(figsize=(4, 1))
lines = [
    plt.Line2D([0], [0], marker = markers[i], color=colors[i], linestyle='-', label=labels[i], linewidth=1.5)
    for i in range(len(labels))
]
legend = ax.legend(
    handles=lines,
    title='Prediction horizon',
    loc='lower center',
    frameon=False,
    fontsize=11,
    title_fontsize=12,
    ncol=len(labels),  # All in one row
    handletextpad=2,   # Increase space between marker and label
    columnspacing=2
)
legend.get_title().set_position((0,10))
ax.axis('off')

fig.savefig(figure_directory+"legend_prediction_horizon.png", bbox_inches='tight', pad_inches=0.05)
plt.close(fig)