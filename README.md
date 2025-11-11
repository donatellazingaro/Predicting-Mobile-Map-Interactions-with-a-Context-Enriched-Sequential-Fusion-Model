# Predicting Mobile Map Interactions with a Context-Enriched Sequential Fusion Model

This repository contains the implementation of a **context-enriched sequential fusion model** that predicts user interaction patterns with **GPS-embedded mobile map applications**.  
The model combines **tappigraphy data** (touch interaction dynamics) with **contextual features** derived from environmental, temporal, and behavioral signals to model engagement patterns during real-world navigation tasks.

---

## Overview

The framework integrates:

- **Sequential modeling:** Long Short-Term Memory (LSTM) networks capture temporal dependencies in user behavior.  
- **Context fusion:** A parallel autoencoder branch encodes contextual information (e.g., spatial, temporal, and environmental metrics) and fuses it with behavioral representations.  
- **Explainability:** Post-hoc analyses (e.g., SHAP) identify key contextual factors influencing interaction behavior.  

The project supports:
- Training and evaluation of the fusion model (with and without the context autoencoder).  
- Ablation studies and augmentation sweeps.  
- Reproducible data preprocessing and explainability workflows.  

---

## Repository Layout

├── data/ # Input pickles or parquet files (train/dev/test)
├── models/ # Saved PyTorch models (encoder, AE, full fusion)
├── outputs/ # Loss logs, metrics, explainability results
├── notebooks/ # Analysis notebooks (aligned with TSAS paper)
├── src/ # Core training and evaluation scripts
│ ├── fusion_train_autoencoder.py
│ ├── fusion_train_fusion_model.py
│ ├── evaluate_fusion_model.py
│ └── explain_fusion_model.py
├── environment.yml # Conda environment for local development
├── environment-ci.yml # CI-safe version for GitHub Actions
└── .github/workflows/ # Continuous integration setup

yaml
Copy code

---

## Getting Started

### 1. Create the environment
```bash
conda env create -f environment.yml
conda activate ci-environment
2. Train the context autoencoder
bash
Copy code
python src/fusion_train_autoencoder.py
3. Train and evaluate the full fusion model
bash
Copy code
python src/fusion_train_fusion_model.py
python src/evaluate_fusion_model.py
4. Run explainability analyses
bash
Copy code
python src/explain_fusion_model.py
