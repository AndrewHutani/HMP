import time
import torch
import numpy as np
import argparse

from tqdm import tqdm
from models.gcnext.config import config

from models.gcnext.RealtimeGCNext import RealtimeGCNext

from models.gcnext.model import GCNext as Model
from dataset.gcnext.h36m_eval import H36MEval

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from models.prediction_times import prediction_times, single_forward_pass_times


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-pth', type=str, default="ckpt/baseline/model-iter-84000.pth", help='=encoder path')
parser.add_argument('--dyna', nargs='+', type=int, default=[0, 48], help='dynamic layer index')
args = parser.parse_args()

config.motion.h36m_target_length = config.motion.h36m_target_length_eval
dataset = H36MEval(config, 'test')
walking_samples = dataset.get_full_sequences_for_action("walking")  # List of (sample, root_sample)

# Prepare model
model = Model(config, args.dyna)
state_dict = torch.load(args.model_pth, map_location = torch.device("cpu"))
model.load_state_dict(state_dict, strict=True)
model.to(torch.device("cpu"))  # Use CPU for inference
realtime_predictor = RealtimeGCNext(model, config, tau=0.5)

latency_times = []

# Run latency measurement for both walking samples
for sample_idx, (walking_sample, root_sample) in enumerate(walking_samples[:2]):
    for i in range(walking_sample.shape[0] - config.motion.h36m_target_length):
        test_input_ = walking_sample[i]
        ground_truth = walking_sample[i:i+config.motion.h36m_target_length]
        t0 = time.perf_counter()
        realtime_predictor.predict(test_input_, ground_truth, visualize=False, debug=False)
        t1 = time.perf_counter()
        latency_times.append(t1 - t0)

if prediction_times:
    avg_prediction_time = sum(prediction_times) / len(prediction_times) * 1000  # convert to ms
    avg_latency_time = sum(latency_times) / len(latency_times) * 1000  # convert to ms
    avg_single_forward_pass_time = sum(single_forward_pass_times) / len(single_forward_pass_times) * 1000  # convert to ms
    print(f"Average prediction time: {avg_prediction_time:.2f} ms")
    print(f"Average end-to-end latency time (including data prep): {avg_latency_time:.2f} ms")
    print(f"Average single forward pass time: {avg_single_forward_pass_time:.2f} ms")
    jitter = np.std(latency_times) * 1000 if latency_times else 0
    print(f"Jitter in latency: {jitter:.2f} ms")

