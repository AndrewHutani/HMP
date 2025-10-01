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
    upper = parse_physmop_data(path, "upper body")
    lower = parse_physmop_data(path, "lower body")

    upper = upper[:, [0, 1, 4, 7]]
    lower = lower[:, [0, 1, 4, 7]]
    # lower = group_average(actions, lower, length)
    combined = np.mean([upper, lower], axis=0)  # shape: (length, 4)
    return combined[length-1]  

# hist_length_50 = get_eval_metrics_at_latest_timestep("performance_logs/gcnext_performance_front_to_back.txt", 50)
hist_length_25 = get_eval_metrics_at_latest_timestep("performance_logs/physmop_data_mpjpe_log_front_to_back.txt", 25)
hist_length_20 = get_eval_metrics_at_latest_timestep("performance_logs/physmop_data_mpjpe_log_hist_length_20.txt", 20)
hist_length_16 = get_eval_metrics_at_latest_timestep("performance_logs/physmop_data_mpjpe_log_hist_length_16.txt", 16)
hist_length_12 = get_eval_metrics_at_latest_timestep("performance_logs/physmop_data_mpjpe_log_hist_length_12.txt", 12)
hist_length_8 = get_eval_metrics_at_latest_timestep("performance_logs/physmop_data_mpjpe_log_hist_length_8.txt", 8)

performance_data = np.array([
    hist_length_8,
    hist_length_12,
    hist_length_16,
    hist_length_20,
    hist_length_25,
    # hist_length_50
])
# Historical lengths (x-axis)
hist_lengths = [8, 12, 16, 20, 25]

prediction_time_data = [5.92, 6.16, 6.29, 6.29, 6.85]
dummy_latency_data = [86.47, 87.64, 97.05, 89.11, 101.64]  # Dummy latency data in ms

percentual_performance = 1/(performance_data / performance_data[-1])
percentual_latency = np.array(dummy_latency_data) / dummy_latency_data[-1]

# print(percentual_performance)

time_horizons = ["80ms", "400ms", "560ms", "1000ms"]
# Line plot
fig1, ax1 = plt.subplots(figsize=(10, 6))
for i in range(4):
    ax1.plot(hist_lengths, performance_data[:, i], marker='o', label=f'{time_horizons[i]}')
ax1.set_xlabel('Retrained historical length (frames)')
ax1.set_xlim(0, None)
ax1.set_ylabel('Absolute MPJPE (mm)')
ax1.set_ylim(0, None)
ax1.set_title('MPJPE vs Retrained historical length for Different Time Horizons')
ax1.legend(title='Predicted Timesteps into the Future')
ax1.grid(True)
fig1.tight_layout()
fig1.savefig('figures/physmop_performance_different_lengths.png', dpi=300)

# Use hist_lengths as the x positions
x = np.array(hist_lengths)
bar_width = .5  # Adjust as needed for spacing

# Bar plot
fig2, ax = plt.subplots(figsize=(10, 6))
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

# Set x-ticks to the center of the grouped bars
ax.set_xticks(x + 1.5*bar_width)
ax.set_xticklabels(hist_lengths)
ax.set_xlim(0, max(hist_lengths) + 4*bar_width)  # Match line graph limits
ax2.set_ylim(ax.get_ylim())  # Adjust y-axis limit as needed
ax.set_xlabel('Historical Length (frames)')
ax.set_ylabel('Percentual Performance (relative to hist 50)')
ax.set_title('Percentual Performance by Historical Length and Time Horizon')
ax.legend(handles1 + handles2, labels1 + labels2,title='Predicted Timesteps into the Future', loc='upper left')

fig2.tight_layout()
fig2.savefig('figures/physmop_bar_performance.png', dpi=300)