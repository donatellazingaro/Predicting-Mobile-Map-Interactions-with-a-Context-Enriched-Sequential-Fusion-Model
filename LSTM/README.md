# LSTM Augmentation and Evaluation Pipeline  
*Predicting Mobile Map Interactions with a Context-Enriched Sequential Fusion Model*

---

## Overview

This module contains the training and evaluation workflow for the **LSTM-based sequential model** used to predict mobile map interactions from tappigraphy data.  
It performs a full **augmentation sweep**, training multiple models under different noise, masking, and feature dropout configurations, and then evaluates all trained checkpoints on a held-out test set.

---

## Workflow

###  Train Models (Augmentation Sweep)
Run:
```bash
python augmentation_sweep.py
```
This will:
- Train models across combinations of mask_prob, noise_std, and feature_drop_prob.

- Save the best checkpoint for each configuration in the models/ folder.

- - Write a summary table to augmentation_sweep_results.json.

Each model uses early stopping and class-balanced sampling for robust performance.

### Evaluate Trained Models
After training completes, run:

bash
Copy code
python evaluate_augmentation_models.py
This will:

Load all .pt checkpoints from models/.

Evaluate them on test_sessions_processed.pkl.

Save:

all_test_results.json — metrics per model (accuracy, precision, recall, F1, ROC-AUC, confusion counts).

threshold_analysis.txt — precision/recall/F1 across thresholds (0.1–0.9).

# Data Requirements

The following files are located in `projects/computational/LSTM/data/`:

| File | Description |
|------|--------------|
| `training_config_clean.pkl` | Model configuration (e.g., feature index mapping) |
| `train_sessions_processed.pkl` | Preprocessed training sessions |
| `dev_sessions_processed.pkl` | Validation sessions for early stopping |
| `test_sessions_processed.pkl` | Test sessions for final evaluation |

### Folder Structure
projects/computational/LSTM/
├── augmentation_sweep.py
├── evaluate_augmentation_models.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│ ├── training_config_clean.pkl
│ ├── train_sessions_processed.pkl
│ ├── dev_sessions_processed.pkl
│ └── test_sessions_processed.pkl
│
├── models/
│ ├── best_0.1_0.15_0.0.pt
│ ├── best_0.2_0.3_0.1.pt
│ └── ...
│
├── augmentation_sweep_results.json
├── all_test_results.json
└── threshold_analysis.txt

# Dependencies
Install all dependencies from requirements.txt

### Output Files

| File | Description |
|------|--------------|
| `augmentation_sweep_results.json` | Training loss per configuration |
| `all_test_results.json` | Model performance metrics |
| `threshold_analysis.txt` | F1, precision, recall across thresholds |

