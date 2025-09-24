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

# Aggregate by group
def group_average(actions, action_data, hist_length):
    group = []
    for act in actions:
        if act in action_data:
            group.append(np.array(action_data[act][:hist_length]))  # shape: (hist_length, 4)
    if group:
        return np.mean(np.stack(group), axis=0)  # shape: (hist_length, 4)
    else:
        return None
    
def get_eval_metrics_at_latest_timestep(path, length):
    upper = parse_gcn_data(path, "upper body")
    lower = parse_gcn_data(path, "lower body")

    upper = group_average(actions, upper, length)
    lower = group_average(actions, lower, length)
    combined = np.mean([upper, lower], axis=0)  # shape: (length, 4)
    return combined[length-1]  

hist_length_50 = get_eval_metrics_at_latest_timestep("performance_logs/gcnext_performance_front_to_back.txt", 50)
hist_length_25 = get_eval_metrics_at_latest_timestep("performance_logs/gcnext_hist_length_25.txt", 25)
hist_length_20 = get_eval_metrics_at_latest_timestep("performance_logs/gcnext_hist_length_20.txt", 20)
hist_length_16 = get_eval_metrics_at_latest_timestep("performance_logs/gcnext_hist_length_16.txt", 16)
hist_length_12 = get_eval_metrics_at_latest_timestep("performance_logs/gcnext_hist_length_12.txt", 12)
hist_length_8 = get_eval_metrics_at_latest_timestep("performance_logs/gcnext_hist_length_8.txt", 8)

performance_data = np.array([
    hist_length_8,
    hist_length_12,
    hist_length_16,
    hist_length_20,
    hist_length_25,
    hist_length_50
])
# Historical lengths (x-axis)
hist_lengths = [8, 12, 16, 20, 25, 50]

dummy_latency_data = [60, 65, 70, 75, 80, 82.7]  # Dummy latency data in ms

percentual_performance = 1/(performance_data / performance_data[-1])
percentual_latency = np.array(dummy_latency_data) / dummy_latency_data[-1]

print(percentual_performance)

time_horizons = ["80ms", "400ms", "560ms", "1000ms"]

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(hist_lengths, performance_data[:, i], marker='o', label=f'{time_horizons[i]}')

plt.xlabel('Retrained historical length (frames)')
plt.xlim(0, None)
plt.ylabel('Absolute MPJPE (mm)')
plt.ylim(0, None)
plt.title('MPJPE vs Retrained historical length for Different Time Horizons')
plt.legend(title='Predicted Timesteps into the Future')
plt.grid(True)
plt.tight_layout()
plt.show()

# Exclude hist_length_50 itself if you want only relative bars
x = np.arange(len(hist_lengths))  # 6 historical lengths
bar_width = 0.1

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(4):
    ax.bar(x + i*bar_width, percentual_performance[:, i], width=bar_width, label=f'{time_horizons[i]}')

# Add latency line on secondary y-axis
ax2 = ax.twinx()
ax2.plot(x + 1.5*bar_width, percentual_latency, color='purple', marker='o', linestyle='-', label='Latency')
ax2.set_ylabel('Percentual Latency (relative to hist 50)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Get handles and labels from both axes
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax.set_ylim(ax2.get_ylim())  # Align y-axis limits
ax.set_xlabel('Historical Length (frames)')
ax.set_ylabel('Percentual Performance (relative to hist 50)')
ax.set_title('Percentual Performance by Historical Length and Time Horizon')
ax.set_xticks(x + 1.5*bar_width)
ax.set_xticklabels(hist_lengths)
ax.legend(handles1 + handles2, labels1 + labels2,title='Predicted Timesteps into the Future', loc='upper left')
ax.grid(True, axis='y')
plt.tight_layout()
plt.show()
