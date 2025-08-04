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

# Parse the data
def parse_gcn_data(filename, body_part):
    import re
    action_data = {}
    current_action = None
    with open(filename, "r") as f:
        for line in f:
            m = re.match(r"Averaged MPJPE \((.+)\) for each observation length and each selected timestep: (.+)", line)
            if m and m.group(1).strip().lower() == body_part.lower():
                current_action = m.group(2).strip().lower()  # <-- action name as key
                action_data[current_action] = []
            elif line.startswith("Obs") and current_action:
                arr = re.findall(r"\[([^\]]+)\]", line)
                if arr:
                    action_data[current_action].append([float(x) for x in arr[0].split()])
    return action_data

upper_data = parse_physmop_data("physmop_data_mpjpe_log_front_to_back.txt", "upper body")
lower_data = parse_physmop_data("physmop_data_mpjpe_log_front_to_back.txt", "lower body")
upper_physics = parse_physmop_data("physmop_physics_mpjpe_log_front_to_back.txt", "upper body")
lower_physics = parse_physmop_data("physmop_physics_mpjpe_log_front_to_back.txt", "lower body")
upper_fusion = parse_physmop_data("physmop_fusion_mpjpe_log_front_to_back.txt", "upper body")
lower_fusion = parse_physmop_data("physmop_fusion_mpjpe_log_front_to_back.txt", "lower body")

upper_gcn = parse_gcn_data("mpjpe_log.txt", "upper body")
lower_gcn = parse_gcn_data("mpjpe_log.txt", "lower body")

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


upper_data = upper_data[:, [0, 1, 4, 7]]
lower_data = lower_data[:, [0, 1, 4, 7]]
upper_physics = upper_physics[:, [0, 1, 4, 7]]
lower_physics = lower_physics[:, [0, 1, 4, 7]]
upper_fusion = upper_fusion[:, [0, 1, 4, 7]]
lower_fusion = lower_fusion[:, [0, 1, 4, 7]]

upper_gcn = group_average(actions, upper_gcn)
lower_gcn = group_average(actions, lower_gcn)

# Compute relative MPJPE (percentage of first observation)
def relative_mpjpe(avg):
    return 100 * avg / avg[0]  # shape: (50, 4)

upper_rel = relative_mpjpe(upper_data)
lower_rel = relative_mpjpe(lower_data)

colors = plt.get_cmap('tab10').colors  # 4 distinct colors

plt.figure(figsize=(10,6))
for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    plt.plot(upper_gcn[:, i], label=f"{label} (Upper)", color=colors[i], linestyle='-')
    plt.plot(lower_gcn[:, i], label=f"{label} (Lower)", color=colors[i], linestyle='--')
    # plt.plot(lower_physics[:, i], label=f"{label} (Physics)", color=colors[i], linestyle='-.')
plt.xlabel("Number of Observed Frames")
plt.ylabel("Relative MPJPE (mm)")
plt.title("Relative MPJPE for the Data Branch vs. Observed Frames")

# First legend
first_line = Line2D([], [], color=colors[0], linestyle='-', linewidth=1.5, label='80ms')
second_line = Line2D([], [], color=colors[1], linestyle='-', linewidth=1.5, label='400ms')
third_line = Line2D([], [], color=colors[2], linestyle='-', linewidth=1.5, label='560ms')
fourth_line = Line2D([], [], color=colors[3], linestyle='-', linewidth=1.5, label='1000ms')

# Second legend
line_solid = Line2D([], [], color='black', linestyle='-', linewidth=1.5, label="Upper")
line_dashed = Line2D([], [], color='black', linestyle='--', linewidth=1.5, label="Lower")
# line_dotted = Line2D([], [], color='black', linestyle='-.', linewidth=1.5, label="Physics")

# Combine all handles and labels into one legend
all_handles = [
    first_line, second_line, third_line, fourth_line,  # Timesteps/colors
    line_solid, line_dashed, #line_dotted               # Line types
]
all_labels = [
    '80ms', '400ms', '560ms', '1000ms',               # Timesteps/colors
    'Upper', 'Lower', #'Physics'                       # Line types
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