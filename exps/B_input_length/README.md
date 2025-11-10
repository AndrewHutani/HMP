# B_input_length

Evaluate model performance for different input sequence lengths.  
This experiment compares GCNext and PhysMoP across multiple input window sizes.  
Each runner writes raw .txt log files to performance_logs/.  
The plotting script reads those logs and generates figures in figures/.

---

## Run from the repository root

All commands must be executed from the root of the repository:

    (HMP) C:\path\to\HMP> python -m exps.B_input_length.<script_name_without_py>

---

## Files
- *convert_physmop_to_gcn.py*  
  Converts the evaluation set used in the PhysMoP model to a format used by the GCNext model.

- *gcnext_different_lengths.py*  
  Evaluates GCNext on the Human3.6M dataset for various input lengths.  
  Logs MPJPE results to performance_logs/gcnext_different_lengths.txt.

- *gcnext_on_amass_different_lengths.py*  
  Evaluates GCNext on the AMASS dataset for the same input lengths.  
  Logs results to performance_logs/gcnext_on_amass_different_lengths.txt.

- *physmop_different_lengths.py*  
  Evaluates PhysMoP (both branches) on the AMASS dataset.  
  Logs results to performance_logs/physmop_different_lengths.txt.

- *eval_different_lengths.py*  
  Loads the .txt performance logs from performance_logs/ and produces the aggregated plots in figures/.

- *figures/*  
  Contains generated figures that visualize the input-length sensitivity for each model.

- *performance_logs/*  
  Contains .txt output logs from each of the runner scripts.

---

## How to Run
### 0. Convert AMASS to GCNext format

    python -m exps.B_input_length.convert_physmop_to_gcn

### 1. Generate raw logs

    python -m exps.B_input_length.gcnext_different_lengths
    python -m exps.B_input_length.gcnext_on_amass_different_lengths
    python -m exps.B_input_length.physmop_different_lengths

### 2. Create the plots

    python -m exps.B_input_length.eval_different_lengths

---
