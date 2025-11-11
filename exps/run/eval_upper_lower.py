import csv
import re
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

actions = ["walking", "eating", "smoking", "discussion", "directions",
                        "greeting", "phoning", "posing", "purchases", "sitting",
                        "sittingdown", "takingphoto", "waiting", "walkingdog",
                        "walkingtogether"]

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
                    action_data[current_action].append([float(x) for x in arr[0].split()])
    return action_data

# upper_data = parse_physmop_data("physmop_data_mpjpe_log_front_to_back.txt", "upper body")
# lower_data = parse_physmop_data("physmop_data_mpjpe_log_front_to_back.txt", "lower body")
# upper_physics = parse_physmop_data("physmop_physics_mpjpe_log_front_to_back.txt", "upper body")
# lower_physics = parse_physmop_data("physmop_physics_mpjpe_log_front_to_back.txt", "lower body")
# upper_fusion = parse_physmop_data("physmop_fusion_mpjpe_log_front_to_back.txt", "upper body")
# lower_fusion = parse_physmop_data("physmop_fusion_mpjpe_log_front_to_back.txt", "lower body")

upper_gcn = parse_gcn_data("gcnext_hist_length_16_pred_length_13.txt", "upper body")
lower_gcn = parse_gcn_data("gcnext_hist_length_16_pred_length_13.txt", "lower body")

# Aggregate by group
def group_average(actions, action_data):
    group = []
    for act in actions:
        if act in action_data:
            group.append(np.array(action_data[act][:16]))  # shape: (50, 4)
    if group:
        return np.mean(np.stack(group), axis=0)  # shape: (50, 4)
    else:
        return None


# upper_data = upper_data[:, [0, 1, 4, 7]]
# lower_data = lower_data[:, [0, 1, 4, 7]]
# upper_physics = upper_physics[:, [0, 1, 4, 7]]
# lower_physics = lower_physics[:, [0, 1, 4, 7]]
# upper_fusion = upper_fusion[:, [0, 1, 4, 7]]
# lower_fusion = lower_fusion[:, [0, 1, 4, 7]]

upper_gcn = group_average(actions, upper_gcn)
lower_gcn = group_average(actions, lower_gcn)

gcn = np.concatenate([upper_gcn, lower_gcn], axis=1)  # shape: (50, 8)

# Compute relative MPJPE (percentage of first observation)
def relative_mpjpe(avg):
    return 100 * avg / avg[0]  # shape: (50, 4)

# upper_rel = relative_mpjpe(upper_data)
# lower_rel = relative_mpjpe(lower_data)

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
