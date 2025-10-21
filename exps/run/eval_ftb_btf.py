import re
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

# Define your groups
static_actions = ["Sitting", "SittingDown", "Posing"]
combination_actions =["Discussion", "Directions", "Phoning", "Eating", "Waiting", "Smoking", "Purchases", "TakingPhoto"]
dynamic_actions = ["Walking", "WalkingTogether", "WalkingDog", "Greeting"]

all_actions = static_actions + combination_actions + dynamic_actions

# Parse the data
def parse_action_data(filename, body_part):
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

back_to_front = parse_gcn_data_average("performance_logs/gcnext_performance_back_to_front.txt")
# back_to_front_lower = parse_gcn_data_average("performance_logs/gcnext_performance_back_to_front.txt")
front_to_back_upper = parse_gcn_data("performance_logs/gcnext_performance_front_to_back.txt", "upper")
front_to_back_lower = parse_gcn_data("performance_logs/gcnext_performance_front_to_back.txt", "lower")
print(front_to_back_upper.keys())

front_to_back_upper = group_average(all_actions, front_to_back_upper)
front_to_back_lower = group_average(all_actions, front_to_back_lower)
front_to_back = np.mean([front_to_back_upper, front_to_back_lower], axis=0)  # shape: (50, 4)
# Combine upper and lower body data for back-to-front and front-to-back
# back_to_front = np.mean([back_to_front_upper, back_to_front_lower], axis=0)  # shape: (50, 8)
# front_to_back = np.mean([front_to_back_upper, front_to_back_lower], axis=0)  # shape: (50, 8)

# # Select the relevant columns for the 4 timesteps
# back_to_front = back_to_front[:, [0, 3, 4, 7]]  # shape: (50, 4)
# front_to_back = front_to_back[:, [0, 3, 4, 7]]  # shape: (50, 4)
print("Back-to-front shape:", back_to_front.shape)
print("Front-to-back shape:", front_to_back.shape)

# Calculate Mean Absolute Difference (MAD) between ftb and btf for each prediction horizon
mad = np.mean(np.abs(front_to_back - back_to_front), axis=0)
print("Mean Absolute Difference (MAD) between Front-to-Back and Back-to-Front for each prediction horizon:")
for i, horizon in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    print(f"{horizon}: {mad[i]:.2f} mm")

# Mean MPJPE per horizon (across frames)
mean_back = np.mean(back_to_front, axis=0)   # shape: (4,)
mean_front = np.mean(front_to_back, axis=0)  # shape: (4,)

# Percentage difference (Tim J. Cole, 2000)
percentage_diff = np.abs(mean_back - mean_front) / ((mean_back + mean_front) / 2) * 100
print("Percentage difference between back-to-front and front-to-back (%):", percentage_diff)

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

# back_to_front_avg = group_average(all_actions, back_to_front)
# front_to_back_avg = group_average(all_actions, front_to_back)

# Compute relative MPJPE (percentage of first observation)
def relative_mpjpe(avg):
    return 100 * avg / avg[0]  # shape: (50, 4)

# back_to_front_rel = relative_mpjpe(back_to_front_avg)
# front_to_back_rel = relative_mpjpe(front_to_back_avg)

colors = plt.get_cmap('tab10').colors  # 4 distinct colors

plt.figure(figsize=(10,6))
for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    plt.plot(back_to_front[:, i], label=f"{label} Back-to-front", color=colors[i], linestyle='-')
    plt.plot(front_to_back[:, i], label=f"{label} Front-to-back", color=colors[i], linestyle='--')
plt.xlabel("Number of Observed Frames")
plt.ylabel("Absolute MPJPE (mm)")
plt.title("Absolute MPJPE vs. Observed Frames\n GCNext model")

# First legend
first_line = Line2D([], [], color=colors[0], linestyle='-', linewidth=1.5, label='80ms')
second_line = Line2D([], [], color=colors[1], linestyle='-', linewidth=1.5, label='400ms')
third_line = Line2D([], [], color=colors[2], linestyle='-', linewidth=1.5, label='560ms')
fourth_line = Line2D([], [], color=colors[3], linestyle='-', linewidth=1.5, label='1000ms')

# Second legend
line_solid = Line2D([], [], color='black', linestyle='-', linewidth=1.5, label="Back-to-front")
line_dashed = Line2D([], [], color='black', linestyle='--', linewidth=1.5, label="Front-to-back")


# first_legend = plt.legend(handles=[first_line, second_line, third_line, fourth_line], loc='upper right', 
#                           bbox_to_anchor=(1.0, 1.0), 
#                           title='Timesteps into the future')
# ax = plt.gca().add_artist(first_legend)
# second_legend = plt.legend(handles=[line_solid, line_dashed], loc='upper right', 
#                            bbox_to_anchor=(0.88, 1.0),
#                            title='Line types')
# Combine all handles and labels into one legend
all_handles = [
    first_line, second_line, third_line, fourth_line,  # Timesteps/colors
    line_solid, line_dashed                           # Line types
]
all_labels = [
    '80ms', '400ms', '560ms', '1000ms',               # Timesteps/colors
    'Back-to-front', 'Front-to-back'                  # Line types
]

plt.legend(
    handles=all_handles,
    labels=all_labels,
    loc='best',
    # bbox_to_anchor=(1.6, 1.0),
    title='Prediction Horizon'
)
plt.grid(True)
plt.tight_layout()
plt.show()