#!filepath projects/computational/'Fusion Context'/fusion_ablation_study.py
"""
fusion_ablation_study.py (aligned with fusion_run.py)
----------------------------------------------------
Performs ablation studies for the LSTM–Context Fusion model
with and without AE compression of context features.

The script:
1. Loads pretrained LSTM backbone and (optionally) AE encoder.
2. Runs ablation variants:
      - Full context (8 features)
      - Urban-only (first 5)
      - Traits-only (last 3)
3. For each variant:
      - Non-AE branch: raw masked context → fusion head training
      - AE branch: masked context encoded with pretrained AE → fusion head training
4. Evaluates both at τ=0.80 and tuned τ (F1-based).
5. Saves overall and per-class metrics and a MAP recall comparison plot.

Outputs:
---------
- outputs/ablation_overall_ae.csv
- outputs/ablation_overall_raw.csv
- outputs/ablation_perclass_ae.csv
- outputs/ablation_perclass_raw.csv
- outputs/ablation_map_recall_compare.png
"""
import os
import json
import pickle
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import DataLoader, TensorDataset  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_fscore_support
)

# === Import models and utilities from fusion_run.py ===
from fusion_run import (
    _load_context_encoder_strict,
    LstmModel,
    LSTMContextFusionHead,
    extract_sliding_windows_with_context
)

# ================================================================
# CONFIG
# ================================================================
SEED = 42
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 1000
PATIENCE = 20
WINDOW_LENGTH = 10
FIXED_TAU = 0.80

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_LSTM_PATH = os.path.join(BASE_DIR, "..", "LSTM", "models", "best_lstm.pt")
AE_ENCODER_PATH = os.path.join(MODEL_DIR, "context_encoder.pt")

# feature group indices (must match order in fusion_context_*.pkl)
URBAN_IDX = [0, 1, 2, 3, 4]
TRAITS_IDX = [5, 6, 7]

# ================================================================
# UTILITIES
# ================================================================
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_fusion_context(path: str):
    data = load_pickle(path)
    ids = [x[0] for x in data]
    ctx = np.array([x[1] for x in data], dtype=np.float32)
    labels = np.array([x[2] for x in data], dtype=np.float32)
    return ids, ctx, labels


# ================================================================
# METRICS
# ================================================================
def tune_threshold(y_true, y_prob):
    best_tau, best_val = 0.5, -1
    for tau in np.linspace(0.05, 0.95, 19):
        y_bin = (y_prob >= tau).astype(int)
        val = f1_score(y_true, y_bin, zero_division=0)
        if val > best_val:
            best_val, best_tau = val, tau
    return float(best_tau)


def compute_metrics(y_true, y_prob, threshold):
    y_bin = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_bin),
        "Precision": precision_score(y_true, y_bin, zero_division=0),
        "Recall": recall_score(y_true, y_bin, zero_division=0),
        "F1": f1_score(y_true, y_bin, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    }


def per_class_metrics(y_true, y_prob, threshold):
    y_bin = (y_prob >= threshold).astype(int)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_bin, labels=[0, 1], zero_division=0)
    return pd.DataFrame({
        "Class": ["NON-MAP", "MAP"],
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "Support": support
    })


# ================================================================
# MAIN TRAINING LOOP
# ================================================================
def train_and_eval_variant(mask_indices: List[int], use_ae: bool, variant_name: str) -> Tuple[Dict, pd.DataFrame]:
    """Train and evaluate one ablation variant (with or without AE)."""
    print(f"\n Variant: {variant_name} | {'AE' if use_ae else 'Non-AE'}")

    # --- Load aligned sessions and context ---
    train_sess = load_pickle(os.path.join(DATA_DIR, "aligned_train_sessions.pkl"))
    dev_sess = load_pickle(os.path.join(DATA_DIR, "aligned_dev_sessions.pkl"))

    train_ids, train_ctx_all, _ = load_fusion_context(os.path.join(DATA_DIR, "fusion_context_train.pkl"))
    dev_ids, dev_ctx_all, _ = load_fusion_context(os.path.join(DATA_DIR, "fusion_context_dev.pkl"))

    train_ctx = train_ctx_all[:, mask_indices]
    dev_ctx = dev_ctx_all[:, mask_indices]

    # --- AE encoding (if applicable) ---
    if use_ae:
        ae = _load_context_encoder_strict(AE_ENCODER_PATH, input_dim=8, latent_dim=3, device=DEVICE)
        ae.eval()
        with torch.no_grad():
            train_ctx = ae(torch.tensor(train_ctx_all, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            dev_ctx = ae(torch.tensor(dev_ctx_all, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        print(f"Encoded context shape: {train_ctx.shape}")

    # --- Prepare LSTM backbone ---
    lstm = LstmModel(in_channels=train_sess[0][1].shape[1], hidden_channels=16, num_layers=3).to(DEVICE)
    state = torch.load(BEST_LSTM_PATH, map_location=DEVICE)
    lstm.load_state_dict({k: v for k, v in state.items() if not k.startswith("head.")}, strict=False)
    for p in lstm.parameters():
        p.requires_grad_(False)
    lstm.eval()

    # --- Extract windows ---
    ctx_lookup_train = dict(zip(train_ids, train_ctx))
    ctx_lookup_dev = dict(zip(dev_ids, dev_ctx))

    X_train, ctx_train, y_train = extract_sliding_windows_with_context(train_sess, ctx_lookup_train, WINDOW_LENGTH)
    X_dev, ctx_dev, y_dev = extract_sliding_windows_with_context(dev_sess, ctx_lookup_dev, WINDOW_LENGTH)

    # --- Fusion head ---
    fusion = LSTMContextFusionHead(lstm_hidden_size=16, context_dim=ctx_train.shape[1]).to(DEVICE)
    opt = optim.Adam(fusion.parameters(), lr=LR)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Data loaders ---
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train), torch.tensor(ctx_train), torch.tensor(y_train)
    ), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TensorDataset(
        torch.tensor(X_dev), torch.tensor(ctx_dev), torch.tensor(y_dev)
    ), batch_size=BATCH_SIZE, shuffle=False)

    # --- Train fusion head ---
    best_dev = float("inf")
    patience = PATIENCE
    for epoch in range(EPOCHS):
        fusion.train()
        for xb, cb, yb in train_loader:
            xb, cb, yb = xb.to(DEVICE), cb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                seq_repr = lstm(xb)
            logits = fusion(seq_repr, cb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        # Evaluate on dev
        fusion.eval()
        dev_loss = 0.0
        probs_dev, y_true = [], []
        with torch.no_grad():
            for xb, cb, yb in dev_loader:
                xb, cb, yb = xb.to(DEVICE), cb.to(DEVICE), yb.to(DEVICE)
                seq_repr = lstm(xb)
                logits = fusion(seq_repr, cb)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                probs_dev.extend(probs)
                y_true.extend(yb.cpu().numpy().ravel())
                dev_loss += loss_fn(logits, yb).item()
        if dev_loss < best_dev - 1e-6:
            best_dev = dev_loss
            best_probs, best_y = np.array(probs_dev), np.array(y_true)
            patience = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                break

    # --- Metrics ---
    tuned_tau = tune_threshold(best_y, best_probs)
    metrics_fixed = compute_metrics(best_y, best_probs, FIXED_TAU)
    metrics_tuned = compute_metrics(best_y, best_probs, tuned_tau)
    perclass = per_class_metrics(best_y, best_probs, FIXED_TAU)

    overall = {
        "Variant": variant_name,
        "Branch": "AE" if use_ae else "Non-AE",
        "DevLoss": float(best_dev),
        "Threshold_fixed": float(FIXED_TAU),
        "Threshold_tuned": float(tuned_tau),
        **{f"{k}@fixed": v for k, v in metrics_fixed.items()},
        **{f"{k}@tuned": v for k, v in metrics_tuned.items()},
    }

    return overall, perclass


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    set_seed()

    variants = {
        "Full": list(range(8)),
        "Urban-only": URBAN_IDX,
        "Traits-only": TRAITS_IDX
    }

    all_overall, all_perclass = [], []

    for use_ae in [False, True]:
        for name, mask in variants.items():
            overall, perclass = train_and_eval_variant(mask, use_ae, name)
            all_overall.append(overall)
            perclass["Variant"] = name
            perclass["Branch"] = "AE" if use_ae else "Non-AE"
            all_perclass.append(perclass)

    df_overall = pd.DataFrame(all_overall)
    df_perclass = pd.concat(all_perclass, ignore_index=True)

    df_overall.to_csv(os.path.join(OUTPUT_DIR, "ablation_overall.csv"), index=False)
    df_perclass.to_csv(os.path.join(OUTPUT_DIR, "ablation_perclass.csv"), index=False)

    # Plot MAP recall comparison
    plt.figure(figsize=(8, 5))
    tidy = df_perclass[df_perclass["Class"] == "MAP"]
    variants_list = ["Full", "Urban-only", "Traits-only"]
    x = np.arange(len(variants_list))
    w = 0.35

    for i, branch in enumerate(["Non-AE", "AE"]):
        vals = [float(tidy[(tidy.Variant == v) & (tidy.Branch == branch)]["Recall"].iloc[0]) for v in variants_list]
        plt.bar(x + (i - 0.5) * w, vals, width=w, label=branch)
        for xi, val in zip(x + (i - 0.5) * w, vals):
            plt.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=9)

    plt.xticks(x, variants_list)
    plt.ylim(0, 1)
    plt.ylabel("MAP Recall (dev)")
    plt.title("Ablation Study: MAP Recall@τ=0.80 (AE vs Non-AE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_map_recall_compare.png"), dpi=600)
    plt.close()

    print("\nSaved:")
    print(" - ablation_overall.csv")
    print(" - ablation_perclass.csv")
    print(" - ablation_map_recall_compare.png")
