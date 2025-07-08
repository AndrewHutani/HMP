import re
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

# Define your groups
static_actions = ["Sitting", "SittingDown", "Posing"]
combination_actions =["Discussion", "Directions", "Phoning", "Eating", "Waiting"]
dynamic_actions = ["Walking", "WalkingTogether", "WalkingDog", "Greeting"]

all_actions = static_actions + combination_actions + dynamic_actions

# Parse the data
def parse_action_data(filename):
    action_data = {}
    current_action = None
    with open(filename, "r") as f:
        for line in f:
            m = re.match(r"Averaged MPJPE for each observation length and each selected timestep: ?([A-Za-z]+)", line)
            if m:
                current_action = m.group(1)
                action_data[current_action] = []
            elif line.startswith("Obs") and current_action:
                arr = re.findall(r"\[([^\]]+)\]", line)
                if arr:
                    action_data[current_action].append([float(x) for x in arr[0].split()])
    return action_data

data_perf = parse_action_data("physmop_data_mpjpe_log.txt")
physics_perf = parse_action_data("physmop_physics_mpjpe_log.txt")
fusion_perf = parse_action_data("physmop_fusion_mpjpe_log.txt")

# Aggregate by group
def group_average(actions, action_data):
    group = []
    # Create a lowercase mapping of action_data for case-insensitive lookup
    action_data_lower = {k.lower(): v for k, v in action_data.items()}
    for act in actions:
        act_lower = act.lower()
        if act_lower in action_data_lower:
            group.append(np.array(action_data_lower[act_lower][:50]))  # shape: (50, 4)
    if group:
        return np.mean(np.stack(group), axis=0)  # shape: (50, 4)
    else:
        return None

data_avg = group_average(all_actions, data_perf)
physics_avg = group_average(all_actions, physics_perf)
fusion_avg = group_average(all_actions, fusion_perf)

# # Compute relative MPJPE (percentage of first observation)
# def relative_mpjpe(avg):
#     return 100 * avg / avg[0]  # shape: (50, 4)

# static_rel = relative_mpjpe(static_avg)
# dynamic_rel = relative_mpjpe(dynamic_avg)

colors = plt.get_cmap('tab10').colors  # 4 distinct colors

plt.figure(figsize=(10,6))
for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    # plt.plot(data_avg[:, i], label=f"{label} (Data)", color=colors[i], linestyle='--')
    # plt.plot(physics_avg[:, i], label=f"{label} (Physics)", color=colors[i], linestyle='-.')
    plt.plot(fusion_avg[:, i], label=f"{label} (Fusion)", color=colors[i], linestyle='-')
plt.xlabel("Number of Observed Frames")
plt.ylabel("Absolute MPJPE (mm)")
plt.title("Absolute MPJPE vs. Observed Frames\nFusion Model")

# First legend
first_line = Line2D([], [], color=colors[0], linestyle='-', linewidth=1.5, label='80ms')
second_line = Line2D([], [], color=colors[1], linestyle='-', linewidth=1.5, label='400ms')
third_line = Line2D([], [], color=colors[2], linestyle='-', linewidth=1.5, label='560ms')
fourth_line = Line2D([], [], color=colors[3], linestyle='-', linewidth=1.5, label='1000ms')

# Second legend
line_solid = Line2D([], [], color='black', linestyle='-', linewidth=1.5, label="Fusion")
line_dashed = Line2D([], [], color='black', linestyle='--', linewidth=1.5, label="Data")
line_dotted = Line2D([], [], color='black', linestyle='-.', linewidth=1.5, label="Physics")


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
    line_solid, line_dashed, line_dotted               # Line types
]
all_labels = [
    '80ms', '400ms', '560ms', '1000ms',               # Timesteps/colors
    'Fusion', 'Data', 'Physics'                       # Line types
]

plt.legend(
    handles=all_handles,
    labels=all_labels,
    loc='best',
    # bbox_to_anchor=(1.6, 1.0),
    title='Predicted Timesteps into the Future'
)
plt.grid(True)
plt.tight_layout()
plt.show()