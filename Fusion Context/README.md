# Fusion Context Module

This folder contains the computational workflow for **context–sequence fusion modeling** in the MapOnTap project.  
It integrates temporal interaction embeddings from an LSTM with contextual representations (urban form and spatial traits), both **with** and **without** a trained autoencoder (AE).  
The module implements model training, evaluation, explainability, and ablation analyses.

---

##  Directory Overview

Fusion Context/
├── data/ # Preprocessed session + context pickles (aligned_, fusion_context_)
├── models/ # Saved checkpoints (.pt)
├── outputs/ # Evaluation results, metrics, SHAP values, plots
│
├── fusion_run.py # Main experiment script (AE vs Non-AE)
├── fusion_explainability.py # SHAP explainability analysis
├── fusion_shap_posneg.py # Positive vs negative SHAP contribution plots
├── fusion_ablation_study.py # Optional ablation analyses
└── README.md

yaml
Copy code

---

## Purpose

The **Fusion Context** module tests how contextual information improves the classification of **map-app sessions** when fused with sequential tappigraphy-based features.

Two pipelines are implemented:

1. **Non-AE Fusion** — uses raw 8-D context vectors  
   *(betweenness, closeness, degree, orientation entropy, circuity, Sense of Direction, Spatial Anxiety, Gender)*

2. **AE Fusion** — compresses context vectors via a pretrained autoencoder encoder  
   *(stored in `models/context_encoder.pt`)*

---

## ⚙️ Core Scripts

### `fusion_run.py`
Runs the full fusion experiment (AE vs Non-AE).

- Loads aligned session and context data.  
- Optionally loads the AE encoder.  
- Trains a fusion head combining LSTM and context embeddings.  
- Saves:
  - Model checkpoints: `fusion_head_ae.pt`, `fusion_head_noae.pt`  
  - Performance metrics: `ae_vs_noae_test_overall.csv`, `ae_vs_noae_test_perclass.csv`  
  - SHAP caches: `shap_cache_ae_dev.npz`, `shap_cache_noae_dev.npz`

**Usage**
```bash
python fusion_run.py
fusion_explainability.py
```

Computes SHAP explainability for both AE and Non-AE fusion models.
Loads SHAP caches from fusion_run.py or recomputes embeddings if missing.

### Outputs:

- SHAP summary plots (With_AE_Overall.png, Without_AE_Overall.png)

- Mean absolute SHAP tables (*_values.csv)

- Optional per-class plots (MAP vs NON-MAP)

### Usage

```bash
python fusion_explainability.py 
fusion_shap_posneg.py  
```

Visualizes positive vs negative SHAP contributions per context feature.

Reads the CSV summaries produced by fusion_explainability.py.

Generates stacked bar plots comparing AE and Non-AE contexts.

Produces:

- shap_ae_posneg_overall.png

- shap_noae_posneg_overall.png


## Usage

``` bash
python fusion_shap_posneg.py
fusion_ablation_study.py
```

Tests model sensitivity by removing or shuffling selected context dimensions.

Usage

```bash
python fusion_ablation_study.py
```

All pickles must be stored in Fusion Context/data/:

File	Description
aligned_train_sessions.pkl	Training session tensors
aligned_dev_sessions.pkl	Validation sessions
aligned_test_sessions.pkl	Test sessions
fusion_context_train.pkl	Context vectors for training
fusion_context_dev.pkl	Context vectors for validation
fusion_context_test.pkl	Context vectors for testing

## Outputs
Folder	Key Files	Description
models/	fusion_head_ae.pt, fusion_head_noae.pt, context_encoder.pt	Trained models
outputs/	ae_vs_noae_test_overall.csv, shap_cache_*.npz, *_values.csv, shap_*_posneg_*.png	Metrics, SHAP results, and figures

### Typical Workflow

### 1. Train or reuse autoencoder
python context_autoencoder_train.py

### 2. Run fusion experiment
python fusion_run.py

### 3. Compute SHAP explainability
python fusion_explainability.py

### 4. Plot positive vs negative SHAP contributions
python fusion_shap_posneg.py
