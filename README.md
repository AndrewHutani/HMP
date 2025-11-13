# Human Motion Prediction Experiments (Thesis Repository)

This repository contains all experiments conducted for the **Master’s Thesis** on **Human Motion Prediction (HMP)**.  
The project investigates how input history, temporal sampling, and model architecture affect accuracy and real-time feasibility of human motion prediction.

Each experiment corresponds directly to a section in the thesis (Sections IV – V).  
All experiments can be reproduced from pre-trained model checkpoints.

## Overview

Evaluate and compare **GCNext** and **PhysMoP** models under different experimental conditions:
- Feeding direction (front-to-back vs back-to-front)  
- Input sequence length  
- Temporal resampling  
- Real-time performance metrics  
- Retrained short-history models

## Environment Setup

This project uses a Conda environment defined in `environment.yml`.

### Creating the environment

```bash
conda env create -f environment.yml
conda activate HMP
```

## Repository Structure
The repository has the following structure where the important directories are outlined


```bash
HMP/
├─ ckpt/ # Model checkpoints
├─ data/ # Raw and processed data
├─ dataset/ # Modules for loading in processed data
├─ exps/
│ ├─ A_feeding_strategy/ # Section IV-A
│ ├─ B_input_length/ # Section IV-B
│ ├─ C_temporal_resolution/ # Section IV-C
│ ├─ D_realtime_metrics/ # Section IV-D
│ └─ E_reduced_input_history/ # Section V – V-A
│
├─ models/ # Model definitions
├─ motion_examples/
└─ README.md # This file
```

Each experiment folder includes:
- A dedicated `README.md` with detailed instructions  
- Python scripts to run the evaluations  
- `figures/` with generated plots  
- `performance_logs/` with `.txt` evaluation logs  

All scripts must be run from the repository root using:
```bash
python -m exps.<experiment_folder>.<script_name_without_py>
```
## Experiment Summary

| Experiment | Folder | Description | Thesis Section |
|-------------|---------|-------------|----------------|
| **A** | `A_feeding_strategy/` | Tests front-to-back vs back-to-front feeding for GCNext and PhysMoP. | IV-A |
| **B** | `B_input_length/` | Evaluates how input window size affects prediction accuracy. | IV-B |
| **C** | `C_temporal_resolution/` | Studies temporal resampling and consistent vs matching output frequencies. | IV-C |
| **D** | `D_realtime_metrics/` | Benchmarks real-time metrics. | IV-D |
| **E** | `E_reduced_input_history/` | Compares retrained short-history models and analyzes the accuracy–latency trade-off. | V – V-A |

---

## Training and Original Repositories

Training and testing scripts are **not included** here.  
To retrain or reproduce the checkpoints, refer to the official implementations:

| Model | Original Repository |
|--------|--------------------|
| **GCNext** | [Original GCNext Repository](https://github.com/BradleyWang0416/GCNext) |
| **PhysMoP** | [Original PhysMoP Repository](https://github.com/zhangy76/PhysMoP) |

In this repository, only inference and evaluation scripts are provided.  
Checkpoints from those repositories should be placed under `ckpt/` as shown above.

---