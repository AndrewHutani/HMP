### *gcnext_short_history_eval.py*

Evaluate **GCNext** using a **shorter input history** (set inside the script) and write a performance log as `.txt`.

**Edit these constants at the top of the file:**
```python
INPUT_LENGTH = 25
SINGLE_PASS_PREDICTION_LENGTH = 10
MODEL_CHECKPOINT = "ckpt/baseline/hist_length_25.pth"
OUTPUT_LOG_FILENAME = "gcnext_hist_length_25.txt"

```

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


