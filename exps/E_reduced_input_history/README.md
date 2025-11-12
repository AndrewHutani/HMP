# E_reduced_input_history

Evaluate the trade-off between **input history length**, **model latency**, and **prediction accuracy**.  
This experiment analyzes the retrained GCNext and PhysMoP models with shorter input windows and compares their real-time performance to the original configurations.

---

## Run from the repository root

All commands must be executed from the root of the repository:

    (HMP) C:\path\to\HMP> python -m exps.E_reduced_input_history.<script_name_without_py>

---

## Files

- *gcnext_short_history_eval.py*  
  Evaluates **GCNext** with a shorter input history.  
  Writes MPJPE results to a `.txt` file inside `performance_logs/`.

- *physmop_short_history_eval.py*  
  Evaluates **PhysMoP** with a shorter input history.  
  Writes MPJPE results to a `.txt` file inside `performance_logs/`.

- *gcnext_retrained_realtime_performance.py*  
  Prints **real-time performance metrics** (latency, throughput, per-frame time) for the retrained GCNext models.  
  Metrics are **printed only** to the console and not stored.

- *physmop_retrained_realtime_performance.py*  
  Prints **real-time performance metrics** for the retrained PhysMoP models.  
  Metrics are **printed only** to the console and not stored.

- *eval_different_lengths.py*  
  Combines the results from both the MPJPE logs and the real-time metrics.  
  The user must **manually fill in the real-time metrics** in the script.  
  The script then produces comparative plots showing accuracy–latency trade-offs.

- *figures/*  
  Contains generated figures comparing accuracy and real-time performance across different input lengths.

- *performance_logs/*  
  Contains the `.txt` MPJPE logs produced by the evaluation scripts.

---
## Model configurations

### GCNext model configurations

| Model name | Input length (`INPUT_LENGTH`) | Single-pass prediction length (`SINGLE_PASS_PREDICTION_LENGTH`)| Checkpoint path (`MODEL_CHECKPOINT`) | Output log (`OUTPUT_LOG_FILENAME`)|
|-------------|------------------------------|-----------------------------------------------------------------|--------------------------------------|------------------------------------|
| **Baseline** | 50 | 10 | `ckpt/baseline/hist_length_50.pth` | `gcnext_performance_front_to_back.txt` |
| | 25 | 10 | `ckpt/baseline/hist_length_25.pth` | `gcnext_hist_length_25.txt` |
| | 20 | 10 | `ckpt/baseline/hist_length_20.pth` | `gcnext_hist_length_20.txt` |
|**Best short-history** | 16 | 10 | `ckpt/baseline/hist_length_16.pth` | `gcnext_hist_length_16.txt` |
|  | 12 | 10 | `ckpt/baseline/hist_length_12.pth` | `gcnext_hist_length_12.txt` | 
|  | 8 | 8 | `ckpt/baseline/hist_length_8.pth` | `gcnext_hist_length_8.txt` | 
| **2-pass variant** | 16 | 13 | `ckpt/baseline/hist_length_16_pred_length_13.pth` | `gcnext_hist_length_16_pred_length_13.txt` | 
| **Single-pass variant** | 25 | 25 | `ckpt/baseline/hist_length_25_pred_length_25.pth` | `gcnext_hist_length_25_pred_length_25.txt` |


### PhysMoP model configurations

| Model name | Input length (`INPUT_LENGTH`) | Checkpoint path (`MODEL_CHECKPOINT`) | Output log (`OUTPUT_LOG_FILENAME`) |
|-------------|------------------------------|--------------------------------------|------------------------------------|
| **Baseline** | 12 | `ckpt/PhysMoP/2023_12_21-17_09_24_20364.pt` | `.txt` |
| **PhysMoP (20 frames)** | 20 | `ckpt/PhysMoP/hist_length_20.pt` | `hist_length_20.txt` |
| **PhysMoP (16 frames)** | 16 | `ckpt/PhysMoP/hist_length_16.pt` | `hist_length_16.txt` |
| **PhysMoP (12 frames)** | 12 | `ckpt/PhysMoP/hist_length_12.pt` | `hist_length_12.txt` |
| **PhysMoP (8 frames)**  | 8  | `ckpt/PhysMoP/hist_length_8.pt`  | `hist_length_8.txt`  |


---

## How to Run

### 1. Evaluate GCNext and PhysMoP models
Run the evaluation scripts to generate MPJPE logs.

    python -m exps.E_reduced_input_history.gcnext_short_history_eval
    python -m exps.E_reduced_input_history.physmop_short_history_eval

Logs are saved automatically in:

    exps/E_reduced_input_history/performance_logs/

### 2. Measure real-time performance
Run these scripts to print latency and throughput metrics to the console.

    python -m exps.E_reduced_input_history.gcnext_retrained_realtime_performance
    python -m exps.E_reduced_input_history.physmop_retrained_realtime_performance


### 3. Generate plots
Once you’ve recorded the printed metrics inside *eval_different_lengths.py*, run:

    python -m exps.E_reduced_input_history.eval_different_lengths

This will create all accuracy–latency plots and export them to:

    exps/E_reduced_input_history/figures/

---

## Expected Outputs

### Logs (.txt)
Written to:
`exps/E_reduced_input_history/performance_logs/`

Examples:
