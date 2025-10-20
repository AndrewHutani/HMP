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
upper_data_longer = parse_physmop_data("physmop_data_longer_mpjpe_log.txt", "upper body")
lower_data_longer = parse_physmop_data("physmop_data_longer_mpjpe_log.txt", "lower body")

upper_gcn_on_amass = parse_physmop_data("gcnext_on_amass.txt", "upper body")
lower_gcn_on_amass = parse_physmop_data("gcnext_on_amass.txt", "lower body")

upper_gcn = parse_gcn_data("mpjpe_log.txt", "upper body")
lower_gcn = parse_gcn_data("mpjpe_log.txt", "lower body")
print(upper_gcn_on_amass.shape, lower_gcn_on_amass.shape)

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
upper_gcn = group_average(actions, upper_gcn)
lower_gcn = group_average(actions, lower_gcn)

upper_data = upper_data[:, [0, 1, 4, 7]]
lower_data = lower_data[:, [0, 1, 4, 7]]
upper_physics = upper_physics[:, [0, 1, 4, 7]]
lower_physics = lower_physics[:, [0, 1, 4, 7]]
upper_fusion = upper_fusion[:, [0, 1, 4, 7]]
lower_fusion = lower_fusion[:, [0, 1, 4, 7]]

colors = plt.get_cmap('tab10').colors  # 4 distinct colors
x_vals = np.arange(1, len(upper_data) + 1)

all_arrays = [
    upper_data, lower_data,
    upper_physics, lower_physics,
    upper_fusion, lower_fusion,
    upper_gcn, lower_gcn,
    upper_data_longer, lower_data_longer
]
all_data = np.concatenate([arr.flatten() for arr in all_arrays if arr is not None])
y_min = np.min(all_data[all_data > 0])  # Avoid zero for log scale
y_max = np.percentile(all_data, 100)
y_limits = (y_min, y_max)

plot_and_save(upper_data, lower_data, "PhysMoP", "Data", x_vals, colors, y_limits)
plot_and_save(upper_physics, lower_physics, "PhysMoP", "Physics", x_vals, colors, y_limits)
plot_and_save(upper_fusion, lower_fusion, "PhysMoP", "Fusion", x_vals, colors, y_limits)

x_vals = np.arange(1, len(upper_gcn) + 1)
print(upper_gcn.shape, lower_gcn.shape)
plot_and_save(upper_gcn, lower_gcn, "GCNext", "Data", x_vals, colors, y_limits)
plot_and_save(upper_gcn_on_amass, lower_gcn_on_amass, "GCNext_on_AMASS", "Data_on_AMASS", x_vals, colors, y_limits)
plot_and_save(upper_data_longer, lower_data_longer, "PhysMoP", "Data (Longer)", x_vals, colors, y_limits)
