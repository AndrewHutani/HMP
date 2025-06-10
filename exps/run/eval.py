import re
import numpy as np
import matplotlib.pyplot as plt

# Define your groups
static_actions = ["Eating", "Sitting", "SittingDown", "Discussion", "Waiting", "Phoning", "Greeting", "Posing"]
dynamic_actions = ["Walking", "WalkingTogether", "WalkingDog", "Directions", "TakingPhoto", "Purchases"]

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

action_data = parse_action_data("temp_performance_data.txt")

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

static_avg = group_average(static_actions, action_data)
dynamic_avg = group_average(dynamic_actions, action_data)

# Compute relative MPJPE (percentage of first observation)
def relative_mpjpe(avg):
    return 100 * avg / avg[0]  # shape: (50, 4)

static_rel = relative_mpjpe(static_avg)
dynamic_rel = relative_mpjpe(dynamic_avg)

colors = plt.get_cmap('tab10').colors  # 4 distinct colors

plt.figure(figsize=(10,6))
for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    plt.plot(static_rel[:, i], label=f"{label} (Static)", color=colors[i], linestyle='-')
    plt.plot(dynamic_rel[:, i], label=f"{label} (Dynamic)", color=colors[i], linestyle='--')
plt.xlabel("Number of Observed Frames")
plt.ylabel("MPJPE (% of initial)")
plt.title("Relative MPJPE vs. Observed Frames\nStatic vs Dynamic Actions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()