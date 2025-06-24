import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define your action groups
static_actions = ["sitting", "sittingdown", "posing"]
dynamic_actions = ["walking", "walkingtogether", "walkingdog", "greeting"]

# Parse the data
def parse_longterm_data(filename, body_part):
    action_data = {}
    with open(filename, "r") as f:
        for line in f:
            m = re.match(rf"Averaged MPJPE \({body_part}\) for ([a-zA-Z0-9]+): \[([^\]]+)\]", line)
            if m:
                action = m.group(1).lower()
                values = [float(x) for x in m.group(2).split()]
                action_data[action] = values
    return action_data

lower_data = parse_longterm_data("longer_term_mpjpe_log.txt", "upper body")

# Aggregate by group
def group_average(actions, action_data):
    group = []
    for act in actions:
        if act in action_data:
            group.append(action_data[act])
    if group:
        return np.mean(np.stack(group), axis=0)
    else:
        return None

dynamic_avg = group_average(dynamic_actions, lower_data)
static_avg = group_average(static_actions, lower_data)

# Actual time values in ms
timesteps_ms = [80, 400, 560, 1000, 2000]

plt.figure(figsize=(10,6))
plt.plot(timesteps_ms, static_avg, label="Static Actions", color='blue', linestyle='-')
plt.plot(timesteps_ms, dynamic_avg, label="Dynamic Actions", color='orange', linestyle='--')

plt.xticks(timesteps_ms, [f"{t}ms" for t in timesteps_ms])
plt.xlabel("Prediction Horizon (ms)")
plt.ylabel("Absolute MPJPE (mm)")
plt.title("Absolute MPJPE for upper body vs. Prediction Horizon\nDynamic vs Static Actions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()