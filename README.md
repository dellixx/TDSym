
# Learning Continuous Temporal Dynamics on Symplectic Manifolds for Temporal Knowledge Graph Embedding

<p align="center">
  <b>Anonymous Submission for Code Reproducibility</b>
</p>






## 📂 Repository Structure

The project is organized as follows:

```text
.
├── models/               # Model implementations
│   ├── TDSym.py          # [Ours] SymplecticTemporal Model
│   ├── TeAST.py          # Baseline: TeAST
│   ├── TeLM.py           # Baseline: TeLM
│   ├── TeRDy.py          # Baseline: TeRDy
│   ├── TComplEx.py       # Baseline: TComplEx
│   ├── TCompoundE.py     # Baseline: TCompoundE
│   └── ...
├── src_data/             # Raw data storage
├── datasets.py           # Dataset loading and batching utilities
├── learner.py            # Main training and evaluation script
├── optimizers.py         # Custom optimizers
├── regularizers.py       # Regularization terms (N3, Time-Regularization)
├── process_icews.py      # Preprocessing for standard interpolation tasks
├── process_icews_time.py # [Important] Preprocessing for  Future Prediction
├── run_base.sh           # Automation script for running Baseline models
├── run_TDSym.sh          # Automation script for running TDSym (Ours)
└── requirements.txt      # Python dependencies

```

## 🛠️ Environment Setup

We recommend using `conda` to manage the environment to ensure reproducibility.

```bash
# 1. Create a new environment
conda create --name tdsym_env python=3.10
conda activate tdsym_env

# 2. Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset Preparation




```bash
python process_icews.py
```


```bash
python process_icews_time.py
```


## 🚀 Reproducing Experiments

We provide shell scripts to automate the training loop for different datasets and ranks.

### 1. Train TDSym (Ours)

To run the proposed model with the configurations reported in the paper:

```bash
bash run_TDSym.sh
```

*Configuration Note: You can modify `GPU_ID`, `DATASETS` (e.g., ICEWS14), and `RANKS` (e.g., 64) directly inside this script.*

### 2. Train Baselines

To run the comparison models (TeLM, TeAST, TComplEx, etc.):

```bash
bash run_base.sh
```

### 3. Manual Execution

You can also execute `learner.py` directly for debugging or single-run experiments:

```bash
# Example: Running TDSym on ICEWS14 with Rank 64
python learner.py \
    --dataset ICEWS14 \
    --model TDSym \
    --rank 64 \
    --learning_rate 0.1 \
    --max_epochs 100
```


## ⚠️ Note on Anonymity

This repository is submitted for a **double-blind review**. Please do not distribute this code.
