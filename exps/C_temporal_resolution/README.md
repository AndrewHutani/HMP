# C_temporal_resolution

Evaluate the effect of **temporal resampling** on model performance.  
This experiment studies how changing the input frame rate affects GCNext and PhysMoP under two evaluation modes:

- **Matching output:** The model output is evaluated at the same (resampled) frame rate as the input.  
- **Consistent output:** The model output is evaluated at the original frame rate (fixed time interval).

Each runner writes raw `.txt` logs to `performance_logs/`.  
The plotting script reads those logs and produces figures in `figures/`.

---

## Run from the repository root

All commands must be executed from the root of the repository:

    (HMP) C:\path\to\HMP> python -m exps.C_temporal_resolution.<script_name_without_py>

---

## Files

- *gcnext_matching_output.py*  
  Evaluates **GCNext** at different resampling rates where the output frame rate matches the resampled input.  
  Writes results to `performance_logs/`.

- *gcnext_consistent_output.py*  
  Evaluates **GCNext** with consistent (fixed) output frame rate across different resampling factors.  
  Writes results to `performance_logs/`.

- *physmop_matching_output.py*  
  Evaluates **PhysMoP** (data and physics branches) with matching output frequency.  
  Writes results to `performance_logs/`.

- *physmop_consistent_output.py*  
  Evaluates **PhysMoP** (data and physics branches) with consistent output frequency.  
  Writes results to `performance_logs/`.

- *eval_temporal_resolution.py*  
  Aggregates results from the six `.csv` logs and generates the plots for model comparison under varying resample factors.

- *utils_resample_eval.py*  
  Contains shared utility functions for resampling, confidence interval computation, and CSV/text export.

- *figures/*  
  Directory for all generated visualizations.

- *performance_logs/*  
  Directory for raw `.csv` logs produced by the runner scripts.

---

## How to Run

### 1. Generate raw logs

    python -m exps.C_temporal_resolution.gcnext_matching_output
    python -m exps.C_temporal_resolution.gcnext_consistent_output
    python -m exps.C_temporal_resolution.physmop_matching_output
    python -m exps.C_temporal_resolution.physmop_consistent_output

### 2. Create the plots

    python -m exps.C_temporal_resolution.eval_temporal_resolution

