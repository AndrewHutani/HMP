# A_feeding_strategy

Evaluate **front-to-back** vs **back-to-front** feeding for **GCNext** and **PhysMoP**.  
Each runner writes raw txt logs to `performance_logs/`.  
The plotting script reads those logs and creates the figures in `figures/`.

---

## Run from the repository root

All commands must be executed from the root of the repository:


(HMP) C:\path\to\HMP> python -m exps.A_feeding_strategy.<script_name_without_py>

## Files

* *gcnext_front_to_back.py*  
Runs GCNext with front-to-back feeding. Writes raw MPJPE logs.

* *gcnext_back_to_front.py*  
Runs GCNext with back-to-front feeding. Writes raw MPJPE logs.

* *physmop_front_to_back.py*  
Runs PhysMoP (data and physics branches) with front-to-back feeding. Writes raw MPJPE logs.

* *physmop_back_to_front.py*  
Runs PhysMoP (data and physics branches) with back-to-front feeding. Writes raw MPJPE logs.

* *eval_ftb_btf.py*  
Loads the raw txt logs from performance_logs/ and generates comparison plots in figures/.

* *figures/*  
Output directory for generated plots. Also contains static legend assets used by the plotting script.

* *performance_logs/*  
Output directory for txt logs produced by the four runner scripts.

## How to Run
1. Generate raw logs
```bash
# GCNext
python -m exps.A_feeding_strategy.gcnext_front_to_back
python -m exps.A_feeding_strategy.gcnext_back_to_front

# PhysMoP (both branches logged by each script)
python -m exps.A_feeding_strategy.physmop_front_to_back
python -m exps.A_feeding_strategy.physmop_back_to_front
```

2. Create the plots
```bash
python -m exps.A_feeding_strategy.eval_ftb_btf
```