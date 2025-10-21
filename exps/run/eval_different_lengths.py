import re
import numpy as np
import matplotlib.pyplot as plt

actions = [
    "walking", "eating", "smoking", "discussion", "directions",
    "greeting", "phoning", "posing", "purchases", "sitting",
    "sittingdown", "takingphoto", "waiting", "walkingdog",
    "walkingtogether"
]
time_horizons = ["80ms", "400ms", "560ms", "1000ms"]
figsize = (10, 6)
dpi = 300

# --- PhysMoP Data Loading ---
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

def get_physmop_metrics_at_latest_timestep(path, length):
    upper = parse_physmop_data(path, "upper body")
    lower = parse_physmop_data(path, "lower body")
    # Select columns for the 4 time horizons
    upper = upper[:, [1, 3, 4, 7]]
    lower = lower[:, [1, 3, 4, 7]]
    combined = np.mean([upper, lower], axis=0)  # shape: (length, 4)
    return combined[length-1]  # last timestep for given length

# --- GCNext Data Loading ---
def parse_gcn_data(filename, body_part):
    action_data = {}
    current_action = None
    with open(filename, "r") as f:
        for line in f:
            m = re.match(r"Averaged MPJPE \((.+)\) for each observation length and each selected timestep: (.+)", line)
            if m and m.group(1).strip().lower() == body_part.lower():
                current_action = m.group(2).strip().lower()
                action_data[current_action] = []
            elif line.startswith("Obs") and current_action:
                arr = re.findall(r"\[([^\]]+)\]", line)
                if arr:
                    action_data[current_action].append([float(x) for x in arr[0].split()])
    return action_data

def group_average(actions, action_data, hist_length):
    group = []
    for act in actions:
        if act in action_data:
            group.append(np.array(action_data[act][:hist_length]))
    if group:
        return np.mean(np.stack(group), axis=0)
    else:
        return None

def get_gcn_metrics_at_latest_timestep(path, length):
    upper = parse_gcn_data(path, "upper body")
    lower = parse_gcn_data(path, "lower body")
    upper = group_average(actions, upper, length)
    lower = group_average(actions, lower, length)
    combined = np.mean([upper, lower], axis=0)
    return combined[length-1]

# -------------------------------------------------------------------------------------------------
physmop_hist_lengths = [8, 12, 16, 20, 25]
physmop_prediction_data = [5.93, 6.14, 5.97, 6.26, 6.26]
physmop_latency_data = [86.30, 87.37, 88.75, 88.39, 92.42] 
physmop_jitter = [11.75, 10.90, 13.46, 13.92, 11.83]


physmop_files = [
    "performance_logs/physmop_data_mpjpe_log_hist_length_8.txt",
    "performance_logs/physmop_data_mpjpe_log_hist_length_12.txt",
    "performance_logs/physmop_data_mpjpe_log_hist_length_16.txt",
    "performance_logs/physmop_data_mpjpe_log_hist_length_20.txt",
    "performance_logs/physmop_data_mpjpe_log_front_to_back.txt"
]
physmop_performance_data = np.array([
    get_physmop_metrics_at_latest_timestep(f, l)
    for f, l in zip(physmop_files, physmop_hist_lengths)
])

physmop_percentual_performance = 1/(physmop_performance_data / physmop_performance_data[-1])
physmop_percentual_prediction = np.array(physmop_prediction_data) / physmop_prediction_data[-1]
physmop_percentual_latency = np.array(physmop_latency_data) / physmop_latency_data[-1]

# -------------------------------------------------------------------------------------------------
gcn_hist_lengths =      [8,     12,     16,     20,     25,     50]
gcn_prediction_data =   [71.15, 53.94,  54.57,  67.94,  69.12,  76.02]
gcn_latency_data =      [71.53, 53.31,  54.94,  68.34,  69.55,  76.51]  # these are actual values
gcn_single_pass_times = [17.79, 17.89,  18.19,  22.65,  23.04,  25.34]  # these are actual values
gcn_jitter =            [3.32,  1.54,   1.02,   3.14,   2.73,   6.85]


gcn_files = [
    "performance_logs/gcnext_hist_length_8.txt",
    "performance_logs/gcnext_hist_length_12.txt",
    "performance_logs/gcnext_hist_length_16.txt",
    "performance_logs/gcnext_hist_length_20.txt",
    "performance_logs/gcnext_hist_length_25.txt",
    "performance_logs/gcnext_performance_front_to_back.txt"
]
gcn_performance_data = np.array([
    get_gcn_metrics_at_latest_timestep(f, l)
    for f, l in zip(gcn_files, gcn_hist_lengths)
])

gcn_percentual_performance = 1/(gcn_performance_data / gcn_performance_data[-1])
gcn_percentual_prediction = np.array(gcn_prediction_data) / gcn_prediction_data[-1]
gcn_percentual_latency = np.array(gcn_latency_data) / gcn_latency_data[-1]
gcn_percentual_single_pass = np.array(gcn_single_pass_times) / gcn_single_pass_times[-1]

# --- Shared axis limits for line plots ---
all_hist_lengths = sorted(set(physmop_hist_lengths + gcn_hist_lengths))
x_min = 0
x_max = max(all_hist_lengths)
x_bar_max = x_max + 4  # for bar plots
y_min_abs = min(physmop_performance_data.min(), gcn_performance_data.min())
y_max_abs = max(physmop_performance_data.max(), gcn_performance_data.max())
y_min_rel = min(physmop_percentual_performance.min(), gcn_percentual_performance.min())
y_max_rel = max(physmop_percentual_performance.max(), gcn_percentual_performance.max()) + 0.1

# --- PhysMoP Line Plot ---
fig1, ax1 = plt.subplots(figsize=figsize)
for i in range(4):
    ax1.plot(physmop_hist_lengths, physmop_performance_data[:, i], marker='o', label=f'{time_horizons[i]}')
ax1.set_xlabel('Retrained historical length (frames)')
ax1.set_xlim(x_min, x_max)
ax1.set_ylabel('Absolute MPJPE (mm)')
ax1.set_ylim(y_min_abs, y_max_abs)
ax1.set_title('PhysMoP: MPJPE vs Retrained historical length')
ax1.legend(title='Prediction horizon')
ax1.grid(True)
fig1.tight_layout()
fig1.savefig('figures/physmop_performance_different_lengths.png', dpi=dpi)

# --- PhysMoP Bar Plot ---
x_physmop = np.array(physmop_hist_lengths)
bar_width = .5
fig2, ax2 = plt.subplots(figsize=figsize)
for i in range(4):
    ax2.bar(x_physmop + i*bar_width, physmop_percentual_performance[:, i], width=bar_width, label=f'{time_horizons[i]}')
ax2.plot(x_physmop + 1.5*bar_width, physmop_percentual_latency, color='purple', marker='o', linestyle='-', label='Latency')
# ax2.plot(x_physmop + 1.5*bar_width, physmop_percentual_prediction, color='pink', marker='o', linestyle='-', label='Prediction Time')
ax2.set_xticks(x_physmop + 1.5*bar_width)
ax2.set_xticklabels(physmop_hist_lengths)
ax2.set_xlim(x_min, x_bar_max)
ax2.set_ylim(y_min_rel, y_max_rel)
ax2.set_xlabel('Historical Length (frames)')
ax2.set_ylabel('Relative MPJPE/Latency (to baseline)')
ax2.set_title('PhysMoP: Relative Performance by Historical Length')
ax2.legend(title='Prediction horizon', loc='upper right')
ax2.grid(True)
fig2.tight_layout()
fig2.savefig('figures/physmop_bar_performance.png', dpi=dpi)

# --- GCNext Line Plot ---
fig3, ax3 = plt.subplots(figsize=figsize)
for i in range(4):
    ax3.plot(gcn_hist_lengths, gcn_performance_data[:, i], marker='o', label=f'{time_horizons[i]}')
ax3.set_xlabel('Retrained historical length (frames)')
ax3.set_xlim(x_min, x_max)
ax3.set_ylabel('Absolute MPJPE (mm)')
ax3.set_ylim(y_min_abs, y_max_abs)
ax3.set_title('GCNext: MPJPE vs Retrained historical length')
ax3.legend(title='Prediction horizon')
ax3.grid(True)
fig3.tight_layout()
fig3.savefig('figures/gcnext_performance_different_lengths.png', dpi=dpi)

# --- GCNext Bar Plot ---
x_gcn = np.array(gcn_hist_lengths)
fig4, ax4 = plt.subplots(figsize=figsize)
for i in range(4):
    ax4.bar(x_gcn + i*bar_width, gcn_percentual_performance[:, i], width=bar_width, label=f'{time_horizons[i]}')
ax4.plot(x_gcn + 1.5*bar_width, gcn_percentual_latency, color='purple', marker='o', linestyle='-', label='Latency')
# ax4.plot(x_gcn + 1.5*bar_width, gcn_percentual_prediction, color='pink', marker='o', linestyle='-', label='Prediction Time')
ax4.plot(x_gcn + 1.5*bar_width, gcn_percentual_single_pass, color='orange', marker='o', linestyle='--', label='Single Pass Time')
ax4.set_xticks(x_gcn + 1.5*bar_width)
ax4.set_xticklabels(gcn_hist_lengths)
ax4.set_xlim(x_min, x_bar_max)
ax4.set_ylim(y_min_rel, y_max_rel)
ax4.set_xlabel('Historical Length (frames)')
ax4.set_ylabel('Relative MPJPE/Latency (to baseline)')
ax4.set_title('GCNext: Relative Performance by Historical Length')
ax4.legend(title='Prediction horizon', loc='upper right')
ax4.grid(True)
fig4.tight_layout()
fig4.savefig('figures/gcnext_bar_performance.png', dpi=dpi)