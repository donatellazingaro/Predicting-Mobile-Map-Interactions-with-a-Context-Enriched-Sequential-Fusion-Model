# Predicting Mobile Map Interactions with a Context-Enriched Sequential Fusion Model

This repository contains the experimental and computational workflow developed in the **MapOnTap** project to investigate how *real-world mobile map interactions* can be predicted from a combination of:

1. **Sequential behavior** — tappigraphy time series modeled with an LSTM backbone  
2. **Context** — urban-form metrics and individual spatial traits (Sense of Direction, Spatial Anxiety, Gender)

The project unites **Geographic Information Science**, **behavioral modeling**, and **machine learning** to answer a central question:

**Does context help us anticipate when a person will interact with a mobile map app?**

---

## Repository Overview

```
projects/
└── computational/
    ├── LSTM/                     # Sequence modeling (tappigraphy)
    │   └── models/
    │       └── best_00_15.pt     # Best LSTM backbone (renamed to best_lstm.pt)
    │
    └── Fusion Context/           # Context–sequence fusion and explainability
        ├── data/                 # Aligned sessions and context pickles
        ├── models/               # AE encoder + fusion heads
        ├── outputs/              # Metrics, SHAP results, ablation plots
        ├── fusion_run.py         # AE vs Non-AE experiment
        ├── fusion_explainability.py
        ├── fusion_shap_posneg.py
        ├── fusion_ablation_study.py
        └── README.md
```

---

## Quickstart

```bash
# 1) Clone repository
git clone https://github.com/donatellazingaro/Predicting-Mobile-Map-Interactions-with-a-Context-Enriched-Sequential-Fusion-Model.git
cd Predicting-Mobile-Map-Interactions-with-a-Context-Enriched-Sequential-Fusion-Model/projects/computational/"Fusion Context"

# 2) (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 3) Verify required data files
ls data

# 4) Run the main experiment (AE vs Non-AE)
python fusion_run.py

# 5) Compute SHAP explainability
python fusion_explainability.py
python fusion_shap_posneg.py

# 6) Optional: run ablation variants
python fusion_ablation_study.py
```

Deactivate with:

```bash
deactivate
```

---

## Required Data

Each `.pkl` file contains preprocessed, anonymized data derived from MapOnTap sessions.

| File | Description |
|------|--------------|
| `aligned_train_sessions.pkl` | Training session tensors `(session_id, T×F, label)` |
| `aligned_dev_sessions.pkl` | Development/validation set |
| `aligned_test_sessions.pkl` | Held-out test set |
| `fusion_context_train.pkl` | Context vectors (8-D) for training |
| `fusion_context_dev.pkl` | Context vectors for validation |
| `fusion_context_test.pkl` | Context vectors for testing |

**Context vector (8 features):**

```
[ betweenness, closeness, degree, orientation_entropy, circuity,
  Sense_of_Direction, Spatial_Anxiety, Gender ]
```

---

## Model Architecture

| Component | Role | Notes |
|------------|------|-------|
| **LSTM Backbone** | Encodes tappigraphy sequences | Trained separately (best checkpoint: `best_00_15.pt` → `best_lstm.pt`) |
| **Context Branch (Non-AE)** | Projects raw 8-D context | Simple MLP |
| **Context Branch (AE)** | Compresses context 8→3 | Pretrained autoencoder encoder (`context_encoder.pt`) |
| **Fusion Head** | Combines LSTM + Context | `[LSTM_repr ; Context_repr] → MLP → logit → sigmoid` |

Typical AE bottleneck: **8 → 16 → 3**, with symmetrical decoder during training.

---

## Fusion Context Module

This module tests how contextual features improve classification of **map-app sessions** when fused with temporal LSTM embeddings.

### Output Artifacts

| File | Description |
|------|--------------|
| `models/fusion_head_ae.pt` | Best fusion model (AE branch) |
| `models/fusion_head_noae.pt` | Baseline fusion model (Non-AE) |
| `outputs/ae_vs_noae_test_overall.csv` | Summary metrics |
| `outputs/ae_vs_noae_test_perclass.csv` | Per-class precision/recall/F1 |
| `outputs/shap_cache_*.npz` | Cached SHAP values |
| `outputs/shap_*_posneg_*.png` | Positive/negative contribution plots |
| `outputs/ablation_overall.csv` | Ablation results |
| `outputs/ablation_map_recall_compare.png` | MAP recall comparison |

---

## Explainability (SHAP)

```bash
python fusion_explainability.py
python fusion_shap_posneg.py
```

* `fusion_explainability.py` generates mean |SHAP| summaries per feature.  
* `fusion_shap_posneg.py` separates positive vs negative contributions for interpretability.  
* All results saved under `outputs/`.

---

## Ablation Studies

```bash
python fusion_ablation_study.py
```

Three ablation variants mirror the main fusion design:

| Variant | Features Used | Description |
|----------|----------------|-------------|
| **Full** | All 8 | Complete context |
| **Urban-only** | First 5 | Street-network metrics |
| **Traits-only** | Last 3 | Spatial abilities + Gender |

Each is tested for **AE** and **Non-AE** branches with metrics at fixed τ = 0.80 and tuned τ (F1-based).

---

## Reproducibility

* Random seeds fixed for NumPy and PyTorch.  
* CuDNN deterministic mode disabled for exact reproducibility.  
* Library versions pinned in `requirements.txt` / `environment.yml`.  
* All runs log checkpoints and device info automatically.

```bash
python -m venv venv
source venv/bin/activate   # enter
deactivate                 # exit
```

---

## Troubleshooting

**Q 1. “Size mismatch for encoder weights (expected 3 or 8)”**  
→ Check that the autoencoder was trained with 8-D inputs and that `input_dim=8, latent_dim=3` in `_load_context_encoder_strict`.

**Q 2. “My code changes don’t take effect.”**  
→ Run using absolute paths, or clear `.pyc` caches:

```bash
find . -name "*.pyc" -delete
```

**Q 3. “How to exit Python REPL or venv?”**  
→ `exit()` or `Ctrl-D`; for venv: `deactivate`.

---

## Contributing

1. Keep functions modular and reproducible.  
2. Maintain consistent variable naming and dimensionality.  
3. Extend both AE and Non-AE branches when adding new context features.  
4. Document every change in shape or preprocessing.

Pull requests should include:
* Summary of the modification  
* Validation metrics (dev + test)  
* Notes on updated artifact paths  

---

## Citation

If you use this code or its datasets, please cite:

```bibtex
@misc{zingaro2025fusion,
  title  = {Predicting Mobile Map Interactions with a Context-Enriched Sequential Fusion Model},
  author = {Zingaro, Donatella and collaborators},
  year   = {2025},
  note   = {GitHub repository},
  howpublished = {\url{https://github.com/donatellazingaro/Predicting-Mobile-Map-Interactions-with-a-Context-Enriched-Sequential-Fusion-Model}}
}
```


---
