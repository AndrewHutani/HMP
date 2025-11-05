import csv
import re
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

actions = ["walking", "eating", "smoking", "discussion", "directions",
                        "greeting", "phoning", "posing", "purchases", "sitting",
                        "sittingdown", "takingphoto", "waiting", "walkingdog",
                        "walkingtogether"]
static_actions = ["sitting", "sittingdown", "posing"]
combination_actions =["Discussion", "Directions", "Phoning", "Eating", "Waiting"]
dynamic_actions = ["walking", "walkingtogether", "walkingdog", "greeting"]

fontsize = 16

def plot_and_save(upper_data, lower_data, 
                  title, 
                  model_name, branch,
                  x_vals, colors, y_limits = None, fps = 25, time_unit="ms"):
    plt.figure(figsize=(10,7))
    for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
        plt.plot(x_vals, upper_data[:, i], label=f"{label} (Upper)", color=colors[i], linestyle='-')
        plt.plot(x_vals, lower_data[:, i], label=f"{label} (Lower)", color=colors[i], linestyle='--')
        # plt.plot(lower_physics[:, i], label=f"{label} (Physics)", color=colors[i], linestyle='-.')
    plt.xlabel("Number of Observed Frames", fontsize=fontsize)
    plt.ylabel("Absolute MPJPE (mm)", fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize-4)
    plt.title(title, fontsize=fontsize)
    
    # add secondary x-axis for time
    ax = plt.gca()
    # map frames -> time (seconds) and inverse
    forward = lambda frames: (frames) / float(fps)   # frame index -> seconds (frame 1 -> time 0)
    inverse = lambda seconds: seconds * float(fps)  # seconds -> frame index

    secax = ax.secondary_xaxis('bottom', functions=(forward, inverse))
    secax.spines['bottom'].set_position(('outward', 60))

    # Align secondary ticks exactly with primary ticks (prevents horizontal offset)
    prim_ticks = ax.get_xticks()
    sec_ticks = forward(np.asarray(prim_ticks))
    secax.set_xticks(sec_ticks)
    secax.tick_params(axis='x', labelsize=fontsize-4)
    # Ensure secondary axis covers the exact transformed visible range of the primary axis
    secax.set_xlim(forward(ax.get_xlim()[0]), forward(ax.get_xlim()[1]))

    if time_unit == 'ms':
        secax.set_xlabel(f"Observed Time Span (ms)", fontsize=fontsize)
        # show ticks in milliseconds
        secax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x*1000:.0f}"))
    else:
        secax.set_xlabel(f"Observed Time Span (s)", fontsize=fontsize)
        secax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))

    fig = plt.gcf()

    plt.grid(True)
    plt.tight_layout()
    plt.yscale('log')
    if y_limits is not None:
        plt.ylim(y_limits)
    save_name = f"figures/mpjpe_{model_name.lower()}_{branch.lower()}_upper_lower.png"
    plt.savefig(save_name)
    plt.close()

# Parse the data
def parse_physmop_data(filename, body_part):
    '''
    Note that there are 8 timehorizons logged per observation length.
    time_idx = [1, 3, 7, 9, 13, 17, 21, 24] # Corresponds to (idx+1)*40 ms in the future, i.e.,
    time horizons are: 80ms, 160ms, 320ms, 400ms, 560ms, 720ms, 880ms, 1000ms
    we want: 80ms, 400ms, 560ms, 1000ms -> indices 0, 3, 4, 7
    '''
    mean_data = []
    std_data = []
    used_indices = [0, 3, 4, 7]
    found_section = False
    header = f"Averaged MPJPE ({body_part}) for each observation length and each selected timestep:"
    with open(filename, "r") as f:
        for line in f:
            if line.strip() == header:
                found_section = True
                continue
            if found_section:
                if line.startswith("Obs"):
                    arr = re.findall(r"\[([^\]]+)\]", line)
                    if arr:
                        mean_data.append([float(x) for x in arr[0].split()])
                elif line.startswith("Std"):
                    arr = re.findall(r"\[([^\]]+)\]", line)
                    if arr:
                        std_data.append([float(x) for x in arr[0].split()])
                elif line.strip() == "":
                    break  # End of section
    mean_data = np.array(mean_data)
    std_data = np.array(std_data)
    if mean_data.shape[1] == 4:
        return mean_data, std_data
    else:
        return np.array(mean_data)[:, used_indices], np.array(std_data)[:, used_indices]

# Parse the data
def parse_gcn_data(filename, body_part):
    import re
    action_data_avg = {}
    action_data_std = {}
    current_action = None
    with open(filename, "r") as f:
        for line in f:
            m = re.match(r"Averaged MPJPE \((.+)\) for each observation length and each selected timestep: (.+)", line)
            if m and m.group(1).strip().lower() == body_part.lower():
                current_action = m.group(2).strip().lower()  # <-- action name as key
                action_data_avg[current_action] = []
                action_data_std[current_action] = []
            elif line.startswith("Obs") and current_action:
                arr = re.findall(r"\[([^\]]+)\]", line)
                if arr:
                    action_data_avg[current_action].append([float(x) for x in arr[0].split()])
            elif line.startswith("Std") and current_action:
                arr = re.findall(r"\[([^\]]+)\]", line)
                if arr:
                    action_data_std[current_action].append([float(x) for x in arr[0].split()])
    return action_data_avg, action_data_std

def parse_percentile_data(filename, body_part, percentile):
    percentile_data = {}
    current_action = None
    with open(filename, "r") as f:
        for line in f:
            m = re.match(
                rf"{percentile} percentile \(({body_part})\) for each observation length and each selected timestep: (.+)",
                line.strip()
            )
            if m:
                current_action = m.group(2).strip().lower()  # <-- action name as key
                percentile_data[current_action] = []
            elif line.startswith("Obs") and current_action:
                arr = re.findall(r"\[([^\]]+)\]", line)
                if arr:
                    percentile_data[current_action].append([float(x) for x in arr[0].split()])
    return percentile_data

def parse_percentile_data_physmop(filename, body_part, percentile):
    data = []
    found_section = False
    header = f"{percentile} percentile ({body_part}) for each observation length and each selected timestep:"
    with open(filename, "r") as f:
        for line in f:
            if line.strip() == header:
                found_section = True
                continue
            if found_section:
                if line.startswith("Obs"):
                    arr = re.findall(r"\[([^\]]+)\]", line)
                    if arr:
                        data.append([float(x) for x in arr[0].split()])
                elif line.strip() == "":
                    break  # End of section
    return np.array(data)

def compute_confidence_interval(mean, std, n, confidence=0.95):
    """
    mean: array of means
    std: array of standard deviations
    n: sample size (number of runs)
    confidence: confidence level (default 0.95)
    Returns lower and upper bounds for the confidence interval
    """
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)  # z-score for 95% CI ≈ 1.96
    error = z * std / np.sqrt(n)
    return mean, error

def export_upper_lower_mean_error_csv(
    upper_mean, upper_error, lower_mean, lower_error, filename,
    frame_indices=None, time_labels=None
):
    """
    Exports both upper and lower body mean±error to a CSV file.
    Each row: Number of input frames, upper body mean±error at 80ms, ..., lower body mean±error at 1000ms
    """
    num_frames = upper_mean.shape[0]
    num_timesteps = upper_mean.shape[1]
    if frame_indices is None:
        frame_indices = list(range(num_frames))
    if time_labels is None:
        time_labels = ["80ms", "400ms", "560ms", "1000ms"]

    with open(filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = (
            ["Number of input frames"] +
            [f"upper body mean\pmerror at {label}" for label in time_labels] +
            [f"lower body mean\pmerror at {label}" for label in time_labels]
        )
        writer.writerow(header)
        for idx in frame_indices:
            row = [idx + 1]
            # Upper body
            for t in range(num_timesteps):
                val = f"{upper_mean[idx, t]:.2f}\pm{upper_error[idx, t]:.2f}"
                row.append(val)
            # Lower body
            for t in range(num_timesteps):
                val = f"{lower_mean[idx, t]:.2f}\pm{lower_error[idx, t]:.2f}"
                row.append(val)
            writer.writerow(row)

upper_data_mean, upper_data_std = parse_physmop_data("physmop_data_mpjpe_log.txt", "upper body")
lower_data_mean, lower_data_std = parse_physmop_data("physmop_data_mpjpe_log.txt", "lower body")
upper_physics_mean, upper_physics_std = parse_physmop_data("physmop_physics_mpjpe_log.txt", "upper body")
lower_physics_mean, lower_physics_std = parse_physmop_data("physmop_physics_mpjpe_log.txt", "lower body")
upper_fusion_mean, upper_fusion_std = parse_physmop_data("physmop_fusion_mpjpe_log.txt", "upper body")
lower_fusion_mean, lower_fusion_std = parse_physmop_data("physmop_fusion_mpjpe_log.txt", "lower body")

upper_gcn_mean, upper_gcn_std = parse_gcn_data("mpjpe_log.txt", "upper body")
lower_gcn_mean, lower_gcn_std = parse_gcn_data("mpjpe_log.txt", "lower body")

# upper_data_longer = parse_physmop_data("physmop_data_longer_mpjpe_log.txt", "upper body")
# lower_data_longer = parse_physmop_data("physmop_data_longer_mpjpe_log.txt", "lower body")

upper_gcn_on_amass_mean, upper_gcn_on_amass_std = parse_physmop_data("gcnext_on_amass.txt", "upper body")
lower_gcn_on_amass_mean, lower_gcn_on_amass_std = parse_physmop_data("gcnext_on_amass.txt", "lower body")

# Aggregate by group
def group_average(actions, action_data):
    group = []
    for act in actions:
        if act in action_data:
            group.append(np.array(action_data[act][:50]))  # shape: (50, 4)
    if group:
        return np.mean(np.stack(group), axis=0)  # shape: (50, 4)
    else:
        return None
    
def find_diminishing_returns_percentage(data, improvement_threshold=2.0):
    """Find where percentage improvement drops below threshold"""
    diminishing_points = []
    
    for timestep in range(data.shape[1]):
        timeseries = data[:, timestep]
        percentage_improvements = []
        
        for i in range(1, len(timeseries)):
            if timeseries[i-1] != 0:  # Avoid division by zero
                improvement = abs((timeseries[i-1] - timeseries[i]) / timeseries[i-1] * 100)
                percentage_improvements.append(improvement)
            else:
                percentage_improvements.append(0)
        
        # Find first point where improvement drops below threshold
        below_threshold = np.where(np.array(percentage_improvements) < improvement_threshold)[0]
        if len(below_threshold) > 0:
            diminishing_points.append(below_threshold[0] + 1)  # +1 because we started from index 1
        else:
            diminishing_points.append(len(data) - 1)
    
    return diminishing_points



upper_gcn_mean = group_average(actions, upper_gcn_mean)
upper_gcn_std = group_average(actions, upper_gcn_std)
lower_gcn_mean = group_average(actions, lower_gcn_mean)
lower_gcn_std = group_average(actions, lower_gcn_std)

upper_gcn_mean, upper_gcn_error = compute_confidence_interval(upper_gcn_mean, upper_gcn_std, 3840)
lower_gcn_mean, lower_gcn_error = compute_confidence_interval(lower_gcn_mean, lower_gcn_std, 3840)
export_upper_lower_mean_error_csv(
    upper_gcn_mean, upper_gcn_error, lower_gcn_mean, lower_gcn_error,
    "gcnext_upper_lower_mean_error.csv",
    frame_indices=[0,2,4,9,14,19,24,29,34,39,44,49],
    time_labels=["80ms", "400ms", "560ms", "1000ms"]
)
print(upper_data_mean.shape, lower_data_mean.shape)

upper_physics_mean, upper_physics_error = compute_confidence_interval(upper_physics_mean, upper_physics_std, 15467)
lower_physics_mean, lower_physics_error = compute_confidence_interval(lower_physics_mean, lower_physics_std, 15467)
export_upper_lower_mean_error_csv(
    upper_physics_mean, upper_physics_error, lower_physics_mean, lower_physics_error,
    "physmop_physics_upper_lower_mean_error.csv",   
    frame_indices=[0,1,2,3,4,9,14,19, 24],
    time_labels=["80ms", "400ms", "560ms", "1000ms"]
)

upper_data_mean, upper_data_error = compute_confidence_interval(upper_data_mean, upper_data_std, 15467)
lower_data_mean, lower_data_error = compute_confidence_interval(lower_data_mean, lower_data_std, 15467)
print(upper_data_mean.shape, lower_data_mean.shape)

export_upper_lower_mean_error_csv(
    upper_data_mean, upper_data_error, lower_data_mean, lower_data_error,
    "physmop_data_upper_lower_mean_error.csv",
    frame_indices=[0,1,2,3,4,9,14,19, 24],
    time_labels=["80ms", "400ms", "560ms", "1000ms"]
)

upper_gcn_on_amass_mean, upper_gcn_on_amass_error = compute_confidence_interval(upper_gcn_on_amass_mean, upper_gcn_on_amass_std, 9203)
lower_gcn_on_amass_mean, lower_gcn_on_amass_error = compute_confidence_interval(lower_gcn_on_amass_mean, lower_gcn_on_amass_std, 9203)
export_upper_lower_mean_error_csv(
    upper_gcn_on_amass_mean, upper_gcn_on_amass_error, lower_gcn_on_amass_mean, lower_gcn_on_amass_error,
    "gcnext_on_amass_upper_lower_mean_error.csv",
    frame_indices=[0,2,4,9,14,19,24,29,34,39,44,49],
    time_labels=["80ms", "400ms", "560ms", "1000ms"]
)

upper_diminishing_pct = find_diminishing_returns_percentage(upper_gcn_mean, improvement_threshold=1.0)
lower_diminishing_pct = find_diminishing_returns_percentage(lower_gcn_mean, improvement_threshold=1.0)


print("Upper body diminishing returns (1% threshold) at frames:", upper_diminishing_pct)
print("Lower body diminishing returns (1% threshold) at frames:", lower_diminishing_pct)


colors = plt.get_cmap('tab10').colors  # 4 distinct colors
x_vals = np.arange(1, len(upper_data_mean) + 1)

all_arrays = [
    upper_data_mean, lower_data_mean,
    upper_physics_mean, lower_physics_mean,
    upper_gcn_mean, lower_gcn_mean,
    upper_gcn_on_amass_mean, lower_gcn_on_amass_mean,
    # upper_data_longer, lower_data_longer
]
all_data = np.concatenate([arr.flatten() for arr in all_arrays if arr is not None])
y_min = np.min(all_data[all_data > 0])  # Avoid zero for log scale
y_max = np.percentile(all_data, 100)
y_limits = (y_min, y_max)



plot_and_save(upper_data_mean, lower_data_mean, 
              "MPJPE for the Data branch of the PhysMoP model vs. Observed Frames",
              "PhysMoP", "Data", x_vals, colors, y_limits)
plot_and_save(upper_physics_mean, lower_physics_mean, 
              "MPJPE for the Physics branch of the PhysMoP model vs. Observed Frames",
              "PhysMoP", "Physics", x_vals, colors, y_limits)

x_vals = np.arange(1, len(upper_gcn_mean) + 1)
print(upper_gcn_on_amass_mean.shape, lower_gcn_on_amass_mean.shape)
plot_and_save(upper_gcn_mean, lower_gcn_mean, 
              "MPJPE for the GCNext model vs. Observed Frames",
              "GCNext", "Data", x_vals, colors, y_limits)
plot_and_save(upper_gcn_on_amass_mean, lower_gcn_on_amass_mean, 
              "MPJPE for the GCNext model on AMASS dataset vs. Observed Frames", 
              "GCNext", "on_AMASS", x_vals, colors, y_limits)

colors = plt.get_cmap('tab10').colors  # 4 distinct colors

# First legend (colored lines)
lines_color = [
    Line2D([], [], color=colors[0], linestyle='-', linewidth=1.5, label='80ms'),
    Line2D([], [], color=colors[1], linestyle='-', linewidth=1.5, label='400ms'),
    Line2D([], [], color=colors[2], linestyle='-', linewidth=1.5, label='560ms'),
    Line2D([], [], color=colors[3], linestyle='-', linewidth=1.5, label='1000ms'),
]

# Second legend (line styles)
lines_style = [
    Line2D([], [], color='black', linestyle='-', linewidth=1.5, label='Upper body'),
    Line2D([], [], color='black', linestyle='--', linewidth=1.5, label='Lower body'),
]

# --- Legend-only figure ---
fig = plt.figure(figsize=(5, 1.3))
ax = fig.add_subplot(111)
ax.axis("off")

# First legend (Prediction Horizon)
legend1 = ax.legend(
    handles=lines_color,
    loc='upper center',
    ncol=4,
    frameon=False,
    fontsize=11,
    handlelength=2,
    handletextpad=0.8,
    columnspacing=1.5,
    bbox_to_anchor=(0.5, 1.0),
    title='Prediction Horizon',
    title_fontsize=12            
)
# Move the title upward slightly
legend1.get_title().set_position((0, 10))  # (x, y) offset in points
ax.add_artist(legend1)

# Second legend (Line styles)
legend2 = ax.legend(
    handles=lines_style,
    loc='lower center',
    ncol=2,
    frameon=False,
    fontsize=11,
    handlelength=2,
    handletextpad=0.8,
    columnspacing=2,
    bbox_to_anchor=(0.5, 0.0)  
)

fig.savefig(
    "figures/legend_upper_lower.png",
    dpi=300,
    bbox_inches="tight",
    bbox_extra_artists=(legend1, legend2),
    pad_inches=0.05
)
plt.close(fig)
