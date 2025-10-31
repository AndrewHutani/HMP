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

def plot_and_save(upper_data, lower_data, model_name, branch, x_vals, colors, y_limits = None, fps = 25, time_unit="ms"):
    plt.figure(figsize=(10,6))
    for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
        plt.plot(x_vals, upper_data[:, i], label=f"{label} (Upper)", color=colors[i], linestyle='-')
        plt.plot(x_vals, lower_data[:, i], label=f"{label} (Lower)", color=colors[i], linestyle='--')
        # plt.plot(lower_physics[:, i], label=f"{label} (Physics)", color=colors[i], linestyle='-.')
    plt.xlabel("Number of Observed Frames")
    plt.ylabel("Absolute MPJPE (mm)")
    if model_name == "GCNext":
        plt.title(f"MPJPE for the {model_name} model vs. Observed Frames")
    else:
        plt.title(f"MPJPE for the {branch} branch of the {model_name} model vs. Observed Frames")
    
    # add secondary x-axis for time
    ax = plt.gca()
    # map frames -> time (seconds) and inverse
    forward = lambda frames: (frames) / float(fps)   # frame index -> seconds (frame 1 -> time 0)
    inverse = lambda seconds: seconds * float(fps)  # seconds -> frame index

    secax = ax.secondary_xaxis('bottom', functions=(forward, inverse))
    secax.spines['bottom'].set_position(('outward', 36))

    # Align secondary ticks exactly with primary ticks (prevents horizontal offset)
    prim_ticks = ax.get_xticks()
    sec_ticks = forward(np.asarray(prim_ticks))
    secax.set_xticks(sec_ticks)
    # Ensure secondary axis covers the exact transformed visible range of the primary axis
    secax.set_xlim(forward(ax.get_xlim()[0]), forward(ax.get_xlim()[1]))

    if time_unit == 'ms':
        secax.set_xlabel(f"Time (ms)")
        # show ticks in milliseconds
        secax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x*1000:.0f}"))
    else:
        secax.set_xlabel(f"Time (s)")
        secax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))

    fig = plt.gcf()

    # First legend
    first_line = Line2D([], [], color=colors[0], linestyle='-', linewidth=1.5, label='80ms')
    second_line = Line2D([], [], color=colors[1], linestyle='-', linewidth=1.5, label='400ms')
    third_line = Line2D([], [], color=colors[2], linestyle='-', linewidth=1.5, label='560ms')
    fourth_line = Line2D([], [], color=colors[3], linestyle='-', linewidth=1.5, label='1000ms')

    # Second legend
    line_solid = Line2D([], [], color='black', linestyle='-', linewidth=1.5, label="Upper")
    line_dashed = Line2D([], [], color='black', linestyle='--', linewidth=1.5, label="Lower")


    # Combine all handles and labels into one legend
    all_handles = [
        first_line, second_line, third_line, fourth_line,  # Timesteps/colors
        line_solid, line_dashed,                           # Line types
    ]
    all_labels = [
        '80ms', '400ms', '560ms', '1000ms',               # Timesteps/colors
        'Upper', 'Lower',                                 # Line types
    ]

    plt.legend(
        handles=all_handles,
        labels=all_labels,
        loc='upper right',
        bbox_to_anchor=(1.2, 1),
        title='Prediction horizon'
    )
    plt.grid(True)
    plt.tight_layout()
    plt.yscale('log')
    if y_limits is not None:
        plt.ylim(y_limits)
    save_name = f"mpjpe_{model_name.lower()}_{branch.lower()}_upper_lower.png"
    plt.savefig(save_name)
    plt.close()

# Parse the data
def parse_physmop_data(filename, body_part):
    '''
    Note that there are 8 timehorizons logged per observation length.
    time_idx = [1, 3, 7, 9, 13, 17, 21, 24] # Corresponds to (idx+1)*40 ms in the future, i.e.,
    time horizons are: 80ms, 160ms, 320ms, 400ms, 560ms, 720ms, 880ms, 1000ms
    we want: 80ms, 400ms, 560ms, 1000ms -> indices 0, 3, 4, 7
    '''
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
                    percentile_data[current_action].append([float(x) for x in arr[0].split()])
    return percentile_data

def parse_percentile_data_physmop(filename, body_part, percentile):
    data = []
    found_section = False
    header = f"{percentile} percentile ({body_part}) for each observation length and each selected timestep:"
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

upper_data = parse_physmop_data("physmop_data_mpjpe_log.txt", "upper body")
lower_data = parse_physmop_data("physmop_data_mpjpe_log.txt", "lower body")
# upper_physics = parse_physmop_data("physmop_physics_mpjpe_log.txt", "upper body")
# lower_physics = parse_physmop_data("physmop_physics_mpjpe_log.txt", "lower body")
# upper_fusion = parse_physmop_data("physmop_fusion_mpjpe_log.txt", "upper body")
# lower_fusion = parse_physmop_data("physmop_fusion_mpjpe_log.txt", "lower body")
upper_data_25 = parse_percentile_data_physmop("physmop_data_mpjpe_log.txt", "upper body", "25th")
upper_data_75 = parse_percentile_data_physmop("physmop_data_mpjpe_log.txt", "upper body", "75th")
lower_data_25 = parse_percentile_data_physmop("physmop_data_mpjpe_log.txt", "lower body", "25th")
lower_data_75 = parse_percentile_data_physmop("physmop_data_mpjpe_log.txt", "lower body", "75th")

upper_gcn_avg, upper_gcn_std = parse_gcn_data("mpjpe_log.txt", "upper body")
lower_gcn_avg, lower_gcn_std = parse_gcn_data("mpjpe_log.txt", "lower body")
upper_25 = parse_percentile_data("mpjpe_log.txt", "upper body", "25th")
upper_75 = parse_percentile_data("mpjpe_log.txt", "upper body", "75th")
lower_25 = parse_percentile_data("mpjpe_log.txt", "lower body", "25th")
lower_75 = parse_percentile_data("mpjpe_log.txt", "lower body", "75th")

print(lower_data_25.shape)  # (50, 8)
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
    
def find_diminishing_returns_percentage(data, improvement_threshold=2.0):
    """Find where percentage improvement drops below threshold"""
    diminishing_points = []
    
    for timestep in range(data.shape[1]):
        timeseries = data[:, timestep]
        percentage_improvements = []
        
        for i in range(1, len(timeseries)):
            if timeseries[i-1] != 0:  # Avoid division by zero
                improvement = abs((timeseries[i-1] - timeseries[i]) / timeseries[i-1] * 100)
                percentage_improvements.append(improvement)
            else:
                percentage_improvements.append(0)
        
        # Find first point where improvement drops below threshold
        below_threshold = np.where(np.array(percentage_improvements) < improvement_threshold)[0]
        if len(below_threshold) > 0:
            diminishing_points.append(below_threshold[0] + 1)  # +1 because we started from index 1
        else:
            diminishing_points.append(len(data) - 1)
    
    return diminishing_points


upper_data = upper_data[:, [0, 1, 4, 7]]
lower_data = lower_data[:, [0, 1, 4, 7]]
# upper_physics = upper_physics[:, [0, 1, 4, 7]]
# lower_physics = lower_physics[:, [0, 1, 4, 7]]
# upper_fusion = upper_fusion[:, [0, 1, 4, 7]]
# lower_fusion = lower_fusion[:, [0, 1, 4, 7]]

upper_gcn_avg = group_average(actions, upper_gcn_avg)
upper_gcn_std = group_average(actions, upper_gcn_std)
lower_gcn_avg = group_average(actions, lower_gcn_avg)
lower_gcn_std = group_average(actions, lower_gcn_std)
upper_25 = group_average(actions, upper_25)
upper_75 = group_average(actions, upper_75)
lower_25 = group_average(actions, lower_25)
lower_75 = group_average(actions, lower_75)

upper_data_25 = upper_data_25[:, [0, 1, 4, 7]]
upper_data_75 = upper_data_75[:, [0, 1, 4, 7]]
lower_data_25 = lower_data_25[:, [0, 1, 4, 7]]
lower_data_75 = lower_data_75[:, [0, 1, 4, 7]]


upper_diminishing_pct = find_diminishing_returns_percentage(upper_gcn, improvement_threshold=1.0)
lower_diminishing_pct = find_diminishing_returns_percentage(lower_gcn, improvement_threshold=1.0)


print("Upper body diminishing returns (1% threshold) at frames:", upper_diminishing_pct)
print("Lower body diminishing returns (1% threshold) at frames:", lower_diminishing_pct)

# Compute relative MPJPE (percentage of first observation)
def relative_mpjpe(avg):
    return 100 * avg / avg[0]  # shape: (50, 4)

# upper_rel = relative_mpjpe(upper_data)
# lower_rel = relative_mpjpe(lower_data)

colors = plt.get_cmap('tab10').colors  # 4 distinct colors
x_vals = np.arange(1, len(upper_data) + 1)

plt.figure(figsize=(10,6))
for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    plt.plot(x_vals, upper_data[:, i], label=f"{label} (Upper)", color=colors[i], linestyle='-')
    plt.fill_between(x_vals, upper_data_25[:, i], upper_data_75[:, i], color=colors[i], alpha=0.2)
    # plt.plot(lower_gcn_avg[:, i], label=f"{label} (Lower)", color=colors[i], linestyle='--')
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

plt.figure(figsize=(10,6))
for i, label in enumerate(["80ms", "400ms", "560ms", "1000ms"]):
    plt.plot(x_vals, lower_data[:, i], label=f"{label} (Lower)", color=colors[i], linestyle='-')
    plt.fill_between(x_vals, lower_data_25[:, i], lower_data_75[:, i], color=colors[i], alpha=0.2)
    # plt.plot(lower_gcn_avg[:, i], label=f"{label} (Lower)", color=colors[i], linestyle='--')
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