# D_realtime_metrics

Evaluate **real-time performance** of GCNext and PhysMoP.  
This experiment focuses on measuring runtime characteristics such as latency, throughput, and processing time per prediction step.

Unlike previous experiments, these scripts **do not save logs or figures** — all results are printed directly to the console.

---

## Run from the repository root

All commands must be executed from the root of the repository:

    (HMP) C:\path\to\HMP> python -m exps.D_realtime_metrics.<script_name_without_py>

---

## Files

- *gcnext_realtime_performance.py*  
  Measures the real-time inference performance of **GCNext**.  
  Prints realtime performance metrics average latency, average prediction time, average single pass time, and jitter.

- *physmop_realtime_performance.py*  
  Measures the real-time inference performance of **PhysMoP** (data and physics branches).  
  Prints realtime performance metrics average latency, average prediction time, and jitter.

---

## How to Run

Run each script individually to print the performance metrics to the console:

    python -m exps.D_realtime_metrics.gcnext_realtime_performance
    python -m exps.D_realtime_metrics.physmop_realtime_performance
