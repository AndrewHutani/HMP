import re
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

fontsize = 16
experiment_directory = "exps/A_feeding_strategy/"

# Define your groups
static_actions = ["Sitting", "SittingDown", "Posing"]
combination_actions =["Discussion", "Directions", "Phoning", "Eating", "Waiting", "Smoking", "Purchases", "TakingPhoto"]
dynamic_actions = ["Walking", "WalkingTogether", "WalkingDog", "Greeting"]

all_actions = static_actions + combination_actions + dynamic_actions

# Parse the data
def parse_physmop_data(filename, body_part):
    data = []
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
                        data.append([float(x) for x in arr[0].split()])
                elif line.strip() == "":
                    break  # End of section
    return np.array(data)

def parse_gcn_data_average(filename):
    import re
    import numpy as np
    all_actions = []
    current_action = None
    with open(filename, "r") as f:
        for line in f:
            m = re.match(r"Averaged MPJPE for each observation length and each selected timestep: (.+)", line)
            if m:
                current_action = []
                all_actions.append(current_action)
            elif line.startswith("Obs") and current_action is not None:
                arr = re.findall(r"\[([^\]]+)\]", line)
                if arr:
                    current_action.append([float(x) for x in arr[0].split()])
    # Filter out any incomplete actions
    actions_np = [np.array(a) for a in all_actions if len(a) == 50]
    if not actions_np:
        raise ValueError("No complete actions found in file.")
    stacked = np.stack(actions_np)  # shape: (num_actions, 50, 4)
    print("Number of actions parsed:", stacked.shape[0])
    avg = np.mean(stacked, axis=0)  # shape: (50, 4)
    return avg

# Parse the data
def parse_gcn_data(filename, body_part):
    import re
    action_data = {}
    current_action = None
    with open(filename, "r") as f:
        for line in f:
            m = re.match(r"Averaged MPJPE \((.+)\) for each observation length and each selected timestep: (.+)", line)
            if m and body_part.lower() in m.group(1).strip().lower():
                current_action = m.group(2).strip().lower()
                action_data[current_action] = []
            elif line.startswith("Obs") and current_action:
                arr = re.findall(r"\[([^\]]+)\]", line)
                if arr:
                    action_data[current_action].append([float(x) for x in arr[0].split()])
            elif line.strip() == "" and current_action:
                current_action = None
    return action_data

# Aggregate by group
def group_average(actions, action_data):
    group = []
    for act in actions:
        key = act.lower()
        if key in action_data:
            group.append(np.array(action_data[key][:50]))  # shape: (50, 4)
    if group:
        return np.mean(np.stack(group), axis=0)  # shape: (50, 4)
    else:
        return None

# GCN parsing
gcn_back_to_front = parse_gcn_data_average(experiment_directory+"performance_logs/gcnext_performance_back_to_front.txt")
gcn_front_to_back_upper = parse_gcn_data(experiment_directory+"performance_logs/gcnext_performance_front_to_back.txt", "upper")
gcn_front_to_back_lower = parse_gcn_data(experiment_directory+"performance_logs/gcnext_performance_front_to_back.txt", "lower")

front_to_back_upper = group_average(all_actions, gcn_front_to_back_upper)
front_to_back_lower = group_average(all_actions, gcn_front_to_back_lower)
gcn_front_to_back = np.mean([front_to_back_upper, front_to_back_lower], axis=0)  # shape: (50, 4)

# PhysMoP parsing
physmop_data_back_to_front_upper = parse_physmop_data(experiment_directory+"performance_logs/physmop_data_mpjpe_log_back_to_front.txt", "upper body")[:, [0, 3, 4, 7]]
physmop_data_back_to_front_lower = parse_physmop_data(experiment_directory+"performance_logs/physmop_data_mpjpe_log_back_to_front.txt", "lower body")[:, [0, 3, 4, 7]]
physmop_data_back_to_front = np.mean([physmop_data_back_to_front_upper, physmop_data_back_to_front_lower], axis=0)  # shape: (50, 4)

physmop_data_front_to_back_upper = parse_physmop_data(experiment_directory+"performance_logs/physmop_data_mpjpe_log_front_to_back.txt", "upper body")[:, [0, 3, 4, 7]]
physmop_data_front_to_back_lower = parse_physmop_data(experiment_directory+"performance_logs/physmop_data_mpjpe_log_front_to_back.txt", "lower body")[:, [0, 3, 4, 7]]
physmop_data_front_to_back = np.mean([physmop_data_front_to_back_upper, physmop_data_front_to_back_lower], axis=0)  # shape: (50, 4)

physmop_physics_back_to_front_upper = parse_physmop_data(experiment_directory+"performance_logs/physmop_physics_mpjpe_log_back_to_front.txt", "upper body")[:, [0, 3, 4, 7]]
physmop_physics_back_to_front_lower = parse_physmop_data(experiment_directory+"performance_logs/physmop_physics_mpjpe_log_back_to_front.txt", "lower body")[:, [0, 3, 4, 7]]
physmop_physics_back_to_front = np.mean([physmop_physics_back_to_front_upper, physmop_physics_back_to_front_lower], axis=0)  # shape: (50, 4)

physmop_physics_front_to_back_upper = parse_physmop_data(experiment_directory+"performance_logs/physmop_physics_mpjpe_log_front_to_back.txt", "upper body")[:, [0, 3, 4, 7]]
physmop_physics_front_to_back_lower = parse_physmop_data(experiment_directory+"performance_logs/physmop_physics_mpjpe_log_front_to_back.txt", "lower body")[:, [0, 3, 4, 7]]
physmop_physics_front_to_back = np.mean([physmop_physics_front_to_back_upper, physmop_physics_front_to_back_lower], axis=0)  # shape: (50, 4)



# Calculate Mean Absolute Difference (MAD) between ftb and btf for each prediction horizon
mad = np.mean(np.abs(gcn_front_to_back - gcn_back_to_front), axis=0)
print("Mean Absolute Difference (MAD) between Front-to-Back and Back-to-Front for each prediction horizon:")
for i, horizon in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    print(f"{horizon}: {mad[i]:.2f} mm")

# Mean MPJPE per horizon (across frames)
mean_back = np.mean(gcn_back_to_front, axis=0)   # shape: (4,)
mean_front = np.mean(gcn_front_to_back, axis=0)  # shape: (4,)

percentage_diff = np.abs(mean_back - mean_front) / ((mean_back + mean_front) / 2) * 100
print("Percentage difference between back-to-front and front-to-back (%):", percentage_diff)

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
    Line2D([], [], color='black', linestyle='-', linewidth=1.5, label='Back-to-front'),
    Line2D([], [], color='black', linestyle='--', linewidth=1.5, label='Front-to-back'),
]

# --- GCNext Plot ---
plt.figure(figsize=(10,6))
for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    plt.plot(gcn_back_to_front[:, i], label=f"{label} Back-to-front", color=colors[i], linestyle='-')
    plt.plot(gcn_front_to_back[:, i], label=f"{label} Front-to-back", color=colors[i], linestyle='--')
plt.xlabel("Number of Observed Frames", fontsize=fontsize)
plt.ylabel("Absolute MPJPE (mm)", fontsize=fontsize)
plt.tick_params(axis='both', labelsize=fontsize-4)
plt.title("Absolute MPJPE vs. Observed Frames\nGCNext model", fontsize=fontsize)
# plt.legend(handles=all_handles, labels=all_labels, loc='best', title='Prediction Horizon')
plt.grid(True)
plt.tight_layout()
plt.savefig(experiment_directory+"figures/feeding_order_gcnext.png")
plt.close()

# --- PhysMoP Data Plot ---
plt.figure(figsize=(10,6))
for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    plt.plot(physmop_data_back_to_front[:, i], label=f"{label} Back-to-front", color=colors[i], linestyle='-')
    plt.plot(physmop_data_front_to_back[:, i], label=f"{label} Front-to-back", color=colors[i], linestyle='--')
plt.xlabel("Number of Observed Frames", fontsize=fontsize)
plt.ylabel("Absolute MPJPE (mm)", fontsize=fontsize)
plt.tick_params(axis='both', labelsize=fontsize-4)
plt.title("Absolute MPJPE vs. Observed Frames\nPhysMoP Data", fontsize=fontsize)
# plt.legend(handles=all_handles, labels=all_labels, loc='best', title='Prediction Horizon')
plt.grid(True)
plt.tight_layout()
plt.savefig(experiment_directory+"figures/feeding_order_physmop_data.png")
plt.close()

# --- PhysMoP Physics Plot ---
plt.figure(figsize=(10,6))
for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    plt.plot(physmop_physics_back_to_front[:, i], label=f"{label} Back-to-front", color=colors[i], linestyle='-')
    plt.plot(physmop_physics_front_to_back[:, i], label=f"{label} Front-to-back", color=colors[i], linestyle='--')
plt.xlabel("Number of Observed Frames", fontsize=fontsize)
plt.ylabel("Absolute MPJPE (mm)", fontsize=fontsize)
plt.tick_params(axis='both', labelsize=fontsize-4)
plt.title("Absolute MPJPE vs. Observed Frames\nPhysMoP Physics", fontsize=fontsize)
# plt.legend(handles=all_handles, labels=all_labels, loc='best', title='Prediction Horizon')
plt.grid(True)
plt.tight_layout()
plt.savefig(experiment_directory+"figures/feeding_order_physmop_physics.png")
plt.close()

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
    experiment_directory+"figures/legend_ftb_btf.png",
    dpi=300,
    bbox_inches="tight",
    bbox_extra_artists=(legend1, legend2),
    pad_inches=0.05
)
plt.close(fig)