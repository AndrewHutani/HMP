import math
import torch
import torch.nn.functional as F
import numpy as np

def hann_lowpass_1d_gcnext(x, cutoff_frac, k_min=5):
    """
    x: [T, J, 3] (time=dim0). cutoff_frac in (0,1], relative to Nyquist (1.0 == Nyquist).
    Simple Hann-windowed low-pass via depthwise 1D conv along time.
    """
    T = x.shape[0]
    if T < 3:
        return x
    k = max(k_min, int(math.ceil(8 / max(1e-6, cutoff_frac)))) | 1
    t = torch.arange(k, device=x.device, dtype=x.dtype)
    n = t - (k - 1) / 2
    fc = 0.5 * cutoff_frac
    h = torch.sinc((2 * fc) * n)
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * (t / (k - 1)))
    h = (h * w).to(x.dtype)
    h = h / h.sum()
    pad = k // 2
    # [T, J, 3] -> [J*3, T]
    x_flat = x.reshape(T, -1).transpose(0, 1)
    h = h.view(1, 1, k).to(x.device, x.dtype)
    y = F.conv1d(F.pad(x_flat.unsqueeze(0), (pad, pad), mode='replicate'),
                 h.expand(x_flat.shape[0], 1, k),
                 groups=x_flat.shape[0])
    y = y.squeeze(0).transpose(0, 1).reshape(T, x.shape[1], x.shape[2])
    return y

def resample_sequence_gcnext(sequence, downsample_rate, total_length, start_idx = 0, antialias=True):
    """
    Resample a sequence (downsample or upsample) to match the logic in single_evaluate_physmop.py.
    Args:
        sequence: torch.Tensor, shape [num_frames, num_joints, 3]
        downsample_rate: float
        total_length: int, number of frames to output
    Returns:
        torch.Tensor, shape [total_length, num_joints, 3]
    """
    device = sequence.device
    dtype = sequence.dtype
    if downsample_rate >= 1.0 or np.isclose(downsample_rate, 1.0):
        # Downsample: interpolate with anti-aliasing
        src_span = int(round(total_length * downsample_rate))
        src_end = min(start_idx + src_span, sequence.shape[0])
        src_seq = sequence[start_idx:src_end]
        # Anti-aliasing
        if antialias and downsample_rate > 1.0 and src_seq.shape[0] > 8:
            cutoff_frac = min(1.0 / downsample_rate, 0.9)
            src_seq = hann_lowpass_1d_gcnext(src_seq, cutoff_frac)
        # Interpolate
        t_src = torch.arange(src_seq.shape[0], device=device, dtype=dtype)
        t_target = torch.linspace(src_seq.shape[0] - 1, 0, total_length, device=device, dtype=dtype)
        t_target = torch.clamp(t_target, 0, src_seq.shape[0] - 1)
        t0 = torch.floor(t_target).long()
        t1 = torch.clamp(t0 + 1, max=src_seq.shape[0] - 1)
        w = (t_target - t0.to(dtype)).view(-1, 1, 1)
        x0 = src_seq[t0]
        x1 = src_seq[t1]
        y = (1 - w) * x0 + w * x1
        # Reverse to chronological order
        y = torch.flip(y, dims=[0])
        return y
    else:
        # Upsample: interpolate to more frames
        upsample_factor = 1.0 / downsample_rate
        seq_slice = sequence[start_idx : start_idx + total_length]
        orig_time_steps = seq_slice.shape[0]
        new_time_steps = int(np.round(orig_time_steps * upsample_factor))
        seq_perm = seq_slice.permute(1, 2, 0)  # [num_joints, 3, time]
        seq_upsampled = F.interpolate(seq_perm, size=new_time_steps, mode='linear', align_corners=True)
        seq_upsampled = seq_upsampled.permute(2, 0, 1)  # [time, num_joints, 3]
        return seq_upsampled[:total_length]


def hann_lowpass_1d_physmop(x, cutoff_frac, k_min=5):
    """
    x: [B, T, F...] (time=dim1). cutoff_frac in (0,1], relative to Nyquist (1.0 == Nyquist).
    Simple Hann-windowed low-pass via depthwise 1D conv along time.
    """
    B, T = x.shape[0], x.shape[1]
    if T < 3:
        return x

    # Kernel length ~ 8/cutoff; keep odd
    k = max(k_min, int(math.ceil(8 / max(1e-6, cutoff_frac)))) | 1
    t = torch.arange(k, device=x.device, dtype=x.dtype)
    n = t - (k - 1) / 2

    # cutoff_frac is relative to Nyquist=0.5 cycles/sample -> normalized cutoff fc (cycles/sample)
    fc = 0.5 * cutoff_frac  # so cutoff_frac=1.0 means fc=0.5 (Nyquist)
    # sinc kernel (torch.sinc uses π-normalization: sinc(x)=sin(πx)/(πx))
    h = torch.sinc((2 * fc) * n)
    # Hann window
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * (t / (k - 1)))
    h = (h * w).to(x.dtype)
    h = h / h.sum()  # DC gain = 1

    pad = k // 2
    x_flat = x.reshape(B, T, -1).transpose(1, 2)  # [B, F, T]
    h = h.view(1, 1, k).to(x.device, x.dtype)
    y = F.conv1d(F.pad(x_flat, (pad, pad), mode='replicate'),
                 h.expand(x_flat.shape[1], 1, k),
                 groups=x_flat.shape[1])
    return y.transpose(1, 2).reshape_as(x)

def resample_with_history_boundary_anchored_physmop(x_hist, N, alpha, antialias = True):
    """
    Resample a sequence x_hist of length L to a new length N, keeping the last frame fixed.
    The resampling is anchored at the history boundary (last frame of x_hist).
    Args:
        x_hist: Input sequence of shape (L, D)
        N: Desired output length
        alpha: Resampling factor (N / L)
        antialias: Whether to apply anti-aliasing filter when downsampling
    """
    B, T_h = x_hist.shape[0], x_hist.shape[1]

    # Anti-alias if downsampling
    if antialias and alpha > 1.0 and T_h > 8:
        # cutoff relative to Nyquist: 1/alpha (<=1). Use a small safety margin.
        cutoff_frac = min(1.0 / alpha, 0.9)
        x_hist = hann_lowpass_1d_physmop(x_hist, cutoff_frac)
    
    # Build target times anchored at the last frame
    t_last = T_h - 1
    t = torch.arange(N, device=x_hist.device, dtype=x_hist.dtype)
    t_target = t_last - t * float(alpha)                # newest->oldest
    # print("Resampling with alpha =", alpha)
    # print("t_target before clamp:", t_target)
    t_target = torch.clamp(t_target, 0, T_h - 1)

    t0 = torch.floor(t_target).long()
    t1 = torch.clamp(t0 + 1, max=T_h - 1)
    w = (t_target - t0.to(t_target.dtype)).view(1, N, *([1] * (x_hist.dim() - 2)))

    x0 = x_hist[:, t0, ...]
    x1 = x_hist[:, t1, ...]
    y  = (1 - w) * x0 + w * x1
    # reverse to chronological order (oldest -> newest)
    return torch.flip(y, dims=[1])


def compute_confidence_interval(mean, std, n, confidence=0.95):
    """
    mean: array of means
    std: array of standard deviations
    n: sample size (number of runs)
    confidence: confidence level (default 0.95)
    Returns lower and upper bounds for the confidence interval
    """
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)  # z-score for 95% CI ≈ 1.96
    errors = []
    for i, std_per_resample_rate in enumerate(std):
        error = z * std_per_resample_rate / np.sqrt(n[i])
        errors.append(error)
    return mean, np.array(errors)

def export_mean_error_resample_rate_csv(
    mean, error, filename,
    resample_rates=None, time_labels=None
):
    """
    Exports mean±error to a CSV file.
    Each row: Number of input frames, mean±error at 80ms, ..., mean±error at 1000ms
    """
    import csv
    num_resample_rates = mean.shape[0]
    num_timesteps = mean.shape[1]
    if resample_rates is None:
        resample_rates = list(range(num_resample_rates))
    if time_labels is None:
        time_labels = ["80ms", "400ms", "560ms", "1000ms"]

    with open(filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = (
            ["Resample rate"] +
            [f"Mean\pmerror at {label}" for label in time_labels]
        )
        writer.writerow(header)
        for idx in range(num_resample_rates):
            row = [f"{resample_rates[idx]:.2f}"]
            for t in range(num_timesteps):
                val = f"{mean[idx, t]:.2f}\pm {error[idx, t]:.2f}"
                row.append(val)
            writer.writerow(row)


