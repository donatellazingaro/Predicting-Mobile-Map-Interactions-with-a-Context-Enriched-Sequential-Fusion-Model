#!filepath projects/computational/Fusion Context/fusion_run.py
"""
fusion_run.py
--------------------------------------------------------
Trains and evaluates LSTM–Context Fusion models with and without
Autoencoder (AE) compression on context features. Produces:

  - Overall & per-class evaluation metrics
  - Stored fusion heads (AE / Non-AE)
  - SHAP-ready caches (for dev set)
  - JSON manifest for reproducibility

Usage:
    python fusion_run.py
"""

import os
import json
import random
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
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

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_LSTM_PATH = os.path.join(BASE_DIR, "..", "LSTM", "models", "best_lstm.pt")
AE_ENCODER_PATH = os.path.join(MODEL_DIR, "context_encoder.pt")
AE_FULL_PATH = os.path.join(MODEL_DIR, "context_autoencoder_full.pt")
FUSION_RAW_CKPT = os.path.join(MODEL_DIR, "fusion_head_noae.pt")
FUSION_AE_CKPT  = os.path.join(MODEL_DIR, "fusion_head_ae.pt")

CSV_OVERALL   = os.path.join(OUTPUT_DIR, "ae_vs_noae_test_overall.csv")
CSV_PERCLASS  = os.path.join(OUTPUT_DIR, "ae_vs_noae_test_perclass.csv")
PKL_PREDS     = os.path.join(OUTPUT_DIR, "ae_vs_noae_test_preds.pkl")
MANIFEST_JSON = os.path.join(OUTPUT_DIR, "manifest.json")

SHAP_RAW_NPZ  = os.path.join(OUTPUT_DIR, "shap_cache_noae_dev.npz")
SHAP_AE_NPZ   = os.path.join(OUTPUT_DIR, "shap_cache_ae_dev.npz")

RAW_FEATURE_NAMES = [
    "betweenness", "closeness", "degree", "orientation_entropy",
    "circuity", "Sense of Direction", "Spatial Anxiety", "Gender"
]

# ================================================================
# UTILITIES
# ================================================================
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fusion_context(pkl_path: str) -> Tuple[list, np.ndarray, np.ndarray]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    ids = [x[0] for x in data]
    ctx = np.array([x[1] for x in data], dtype=np.float32)
    labels = np.array([x[2] for x in data], dtype=np.float32)
    return ids, ctx, labels


def load_aligned_sessions(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def extract_sliding_windows_with_context(aligned_sessions, context_lookup: Dict[Any, np.ndarray], window_length: int = 10):
    seq_windows, ctx_windows, labels = [], [], []
    for sid, session_tensor, label in aligned_sessions:
        if sid not in context_lookup:
            continue
        n = session_tensor.shape[0]
        if n < window_length:
            continue
        for start in range(n - window_length + 1):
            window = session_tensor[start:start + window_length].clone()
            seq_windows.append(window.numpy())
            ctx_windows.append(context_lookup[sid])
            labels.append(label)
    return np.stack(seq_windows), np.stack(ctx_windows), np.array(labels, dtype=np.float32)


def _require_file(path: str, hint: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file missing: {path}\nHint: {hint}")


# ================================================================
# DATASET
# ================================================================
class SessionFusionDataset(Dataset):
    def __init__(self, seq_data: np.ndarray, ctx_data: np.ndarray, labels: np.ndarray):
        self.seq_data = seq_data
        self.ctx_data = ctx_data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "sequence": torch.tensor(self.seq_data[idx], dtype=torch.float32),
            "context": torch.tensor(self.ctx_data[idx], dtype=torch.float32),
            "label": torch.tensor([self.labels[idx]], dtype=torch.float32)
        }


# ================================================================
# MODELS
# ================================================================
class LstmModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, lstm_dropout=0.3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_channels, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_channels, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]


class ContextEncoder(nn.Module):
    def __init__(self, input_dim=8, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
    def forward(self, x):
        return self.encoder(x)


class LSTMContextFusionHead(nn.Module):
    def __init__(self, lstm_hidden_size, context_dim, fusion_hidden=32, dropout=0.3):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(lstm_hidden_size + fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )
    def forward(self, seq_repr, context_vec):
        ctx = self.context_proj(context_vec)
        fused = torch.cat([seq_repr, ctx], dim=1)
        return self.fusion_head(fused)


# ================================================================
# ENCODER LOADER (ROBUST)
# ================================================================
def _load_context_encoder_infer(ckpt_path: str, device: torch.device) -> tuple[nn.Module, int, int]:
    """Loads AE encoder, inferring input and latent dims automatically."""
    state = torch.load(ckpt_path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {ckpt_path}")

    # Handle full AE checkpoints by filtering only encoder weights
    if any(k.startswith("decoder.") for k in state.keys()):
        state = {k: v for k, v in state.items() if k.startswith("encoder.")}
    # Normalize keys: remove prefix if needed
    if all(k.startswith("encoder.") for k in state.keys()):
        state = {k.replace("encoder.", ""): v for k, v in state.items()}

    # Infer dimensions
    w0, w2 = state.get("0.weight"), state.get("2.weight")
    if w0 is None or w2 is None:
        raise RuntimeError(f"Incomplete encoder checkpoint: {list(state.keys())}")
    in_dim = w0.shape[1]
    latent_dim = w2.shape[0]

    enc = ContextEncoder(input_dim=in_dim, latent_dim=latent_dim).to(device)
    enc.load_state_dict({f"encoder.{k}": v for k, v in state.items()}, strict=True)
    enc.eval()
    print(f"Loaded Context Encoder from {ckpt_path} (in_dim={in_dim}, latent_dim={latent_dim})")
    return enc, in_dim, latent_dim


# ================================================================
# TRAIN / EVAL HELPERS
# ================================================================
def train_fusion(lstm_model, fusion_head, loader, optimizer, criterion):
    lstm_model.eval()
    fusion_head.train()
    total_loss = 0.0
    for batch in loader:
        seq = batch["sequence"].to(DEVICE)
        ctx = batch["context"].to(DEVICE)
        lbl = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        with torch.no_grad():
            seq_repr = lstm_model(seq)
        logits = fusion_head(seq_repr, ctx)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seq.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_fusion(lstm_model, fusion_head, loader, criterion):
    lstm_model.eval()
    fusion_head.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []
    for batch in loader:
        seq = batch["sequence"].to(DEVICE)
        ctx = batch["context"].to(DEVICE)
        lbl = batch["label"].to(DEVICE)
        seq_repr = lstm_model(seq)
        logits = fusion_head(seq_repr, ctx)
        loss = criterion(logits, lbl)
        total_loss += loss.item() * seq.size(0)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.extend(probs)
        all_targets.extend(lbl.cpu().numpy().ravel())
    return total_loss / len(loader.dataset), np.array(all_probs), np.array(all_targets)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_bin = (y_prob >= threshold).astype(int)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")
    acc  = accuracy_score(y_true, y_bin)
    prec = precision_score(y_true, y_bin, zero_division=0)
    rec  = recall_score(y_true, y_bin, zero_division=0)
    f1   = f1_score(y_true, y_bin, zero_division=0)
    pr   = average_precision_score(y_true, y_prob)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "PR-AUC": pr, "ROC-AUC": roc_auc}


def per_class_metrics(y_true, y_prob, threshold):
    y_bin = (y_prob >= threshold).astype(int)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_bin, labels=[0,1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0,1]).ravel()
    return pd.DataFrame({
        "Class": ["NON-MAP", "MAP"],
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "Support": support
    }), dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def tune_threshold(y_true, y_prob, metric="F1"):
    best_tau, best_val = 0.5, -1
    for tau in np.linspace(0.05, 0.95, 19):
        y_bin = (y_prob >= tau).astype(int)
        val = f1_score(y_true, y_bin, zero_division=0) if metric.upper() == "F1" else accuracy_score(y_true, y_bin)
        if val > best_val:
            best_val, best_tau = val, tau
    return float(best_tau)


# ================================================================
# SHAP CACHING
# ================================================================
@torch.no_grad()
def _compute_seq_repr(lstm: LstmModel, X: np.ndarray) -> np.ndarray:
    seq_repr = []
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    bs = 2048
    for i in range(0, len(X_tensor), bs):
        seq_repr.append(lstm(X_tensor[i:i+bs]).cpu().numpy())
    return np.vstack(seq_repr)


def _save_shap_cache(path_npz: str, seq_repr: np.ndarray, ctx: np.ndarray, y: np.ndarray,
                     feature_names: list | None, mode_label: str) -> None:
    np.savez_compressed(
        path_npz,
        seq_repr=seq_repr,
        ctx=ctx,
        y=y.astype(np.float32),
        mode=np.array([mode_label]),
        feature_names=np.array(feature_names if feature_names else [], dtype=object),
    )


# ================================================================
# RUN FUSION EXPERIMENT
# ================================================================
def run_fusion(use_ae: bool):
    label_name = "AE" if use_ae else "Non-AE"
    print(f"\nRunning Fusion Experiment | {label_name}")

    _require_file(BEST_LSTM_PATH, "Train LSTM backbone first.")
    if use_ae:
        if os.path.exists(AE_ENCODER_PATH):
            ckpt = AE_ENCODER_PATH
        elif os.path.exists(AE_FULL_PATH):
            ckpt = AE_FULL_PATH
        else:
            raise FileNotFoundError("No AE checkpoint found.")
        ae, ae_in_dim, ae_latent_dim = _load_context_encoder_infer(ckpt, device=DEVICE)

    # --- Load data ---
    train_sess = load_aligned_sessions(os.path.join(DATA_DIR, "aligned_train_sessions.pkl"))
    dev_sess   = load_aligned_sessions(os.path.join(DATA_DIR, "aligned_dev_sessions.pkl"))
    test_sess  = load_aligned_sessions(os.path.join(DATA_DIR, "aligned_test_sessions.pkl"))

    train_ids, train_ctx, _ = load_fusion_context(os.path.join(DATA_DIR, "fusion_context_train.pkl"))
    dev_ids,   dev_ctx,   _ = load_fusion_context(os.path.join(DATA_DIR, "fusion_context_dev.pkl"))
    test_ids,  test_ctx,  _ = load_fusion_context(os.path.join(DATA_DIR, "fusion_context_test.pkl"))

    train_ctx_lookup = dict(zip(train_ids, train_ctx))
    dev_ctx_lookup   = dict(zip(dev_ids, dev_ctx))
    test_ctx_lookup  = dict(zip(test_ids, test_ctx))

    X_train, ctx_train, y_train = extract_sliding_windows_with_context(train_sess, train_ctx_lookup, WINDOW_LENGTH)
    X_dev,   ctx_dev,   y_dev   = extract_sliding_windows_with_context(dev_sess,   dev_ctx_lookup,   WINDOW_LENGTH)
    X_test,  ctx_test,  y_test  = extract_sliding_windows_with_context(test_sess,  test_ctx_lookup,  WINDOW_LENGTH)

    # --- AE encoding ---
    if use_ae:
        if ctx_train.shape[1] != ae_in_dim:
            raise RuntimeError(f"Context dim mismatch: context={ctx_train.shape[1]}, encoder expects {ae_in_dim}")
        with torch.no_grad():
            ctx_train = ae(torch.tensor(ctx_train, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            ctx_dev   = ae(torch.tensor(ctx_dev,   dtype=torch.float32, device=DEVICE)).cpu().numpy()
            ctx_test  = ae(torch.tensor(ctx_test,  dtype=torch.float32, device=DEVICE)).cpu().numpy()
        print(f"Context dims after encoding: {ctx_train.shape[1]} (latent={ae_latent_dim})")

    # --- Load frozen LSTM backbone ---
    lstm = LstmModel(in_channels=X_train.shape[2], hidden_channels=16, num_layers=3).to(DEVICE)
    state_dict = torch.load(BEST_LSTM_PATH, map_location=DEVICE)
    filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    lstm.load_state_dict(filtered_state, strict=False)
    for p in lstm.parameters():
        p.requires_grad_(False)
    lstm.eval()

    train_loader = DataLoader(SessionFusionDataset(X_train, ctx_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(SessionFusionDataset(X_dev,   ctx_dev,   y_dev),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(SessionFusionDataset(X_test,  ctx_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

    fusion_head = LSTMContextFusionHead(lstm_hidden_size=16, context_dim=ctx_train.shape[1]).to(DEVICE)
    optimizer = optim.Adam(fusion_head.parameters(), lr=LR)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_dev = float("inf")
    patience_left = PATIENCE
    ckpt_path = FUSION_AE_CKPT if use_ae else FUSION_RAW_CKPT

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_fusion(lstm, fusion_head, train_loader, optimizer, criterion)
        dev_loss, _, _ = eval_fusion(lstm, fusion_head, dev_loader, criterion)
        print(f"{label_name} | Epoch {epoch:03d} | Train {train_loss:.5f} | Dev {dev_loss:.5f}")
        if dev_loss < best_dev - 1e-6:
            best_dev = dev_loss
            patience_left = PATIENCE
            torch.save(fusion_head.state_dict(), ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"{label_name} | Early stopping.")
                break

    fusion_head.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    dev_loss, dev_probs, dev_targets    = eval_fusion(lstm, fusion_head, dev_loader,  criterion)
    test_loss, test_probs, test_targets = eval_fusion(lstm, fusion_head, test_loader, criterion)
    tuned_tau = tune_threshold(dev_targets, dev_probs, metric="F1")

    metrics_tau   = compute_metrics(test_targets, test_probs, threshold=FIXED_TAU)
    metrics_tuned = compute_metrics(test_targets, test_probs, threshold=tuned_tau)
    perclass_df, cm_dict = per_class_metrics(test_targets, test_probs, threshold=FIXED_TAU)

    seq_repr_dev = _compute_seq_repr(lstm, X_dev)
    if use_ae:
        ae_feature_names = [f"AE_latent_{i+1}" for i in range(ctx_dev.shape[1])]
        _save_shap_cache(SHAP_AE_NPZ, seq_repr_dev, ctx_dev, y_dev, ae_feature_names, "AE")
    else:
        _save_shap_cache(SHAP_RAW_NPZ, seq_repr_dev, ctx_dev, y_dev, RAW_FEATURE_NAMES, "Non-AE")

    out = {
        "DevLoss": float(dev_loss),
        "TestLoss": float(test_loss),
        "Threshold_fixed": float(FIXED_TAU),
        "Threshold_tuned": float(tuned_tau),
        **{f"{k}@fixed": float(v) for k, v in metrics_tau.items()},
        **{f"{k}@tuned": float(v) for k, v in metrics_tuned.items()},
        **cm_dict
    }
    preds = {"probs": test_probs, "targets": test_targets}

    return out, perclass_df.assign(Setting=label_name), preds, label_name, ckpt_path


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    set_global_seed(SEED)

    results = {}
    perclass_rows = []
    preds = {}
    artifacts = {}

    for use_ae in [False, True]:
        metrics, perclass_df, out_preds, label, ckpt = run_fusion(use_ae)
        results[label] = metrics
        perclass_rows.append(perclass_df)
        preds[label] = out_preds
        artifacts[label] = {
            "fusion_head_ckpt": ckpt,
            "shap_cache_npz": SHAP_AE_NPZ if label == "AE" else SHAP_RAW_NPZ
        }

    df_results = pd.DataFrame(results).T
    print("\nTest Results (rows = Non-AE vs AE):")
    print(df_results)
    df_results.to_csv(CSV_OVERALL, index=True)

    df_perclass = pd.concat(perclass_rows, ignore_index=True)
    df_perclass.to_csv(CSV_PERCLASS, index=False)

    with open(PKL_PREDS, "wb") as f:
        pickle.dump(preds, f)

    manifest = {
        "device": str(DEVICE),
        "best_lstm_path": os.path.abspath(BEST_LSTM_PATH),
        "context_encoder_path": os.path.abspath(AE_ENCODER_PATH) if os.path.exists(AE_ENCODER_PATH) else None,
        "fusion_heads": artifacts,
        "outputs": {
            "overall_csv": os.path.abspath(CSV_OVERALL),
            "perclass_csv": os.path.abspath(CSV_PERCLASS),
            "preds_pkl": os.path.abspath(PKL_PREDS),
            "shap_cache_noae_npz": os.path.abspath(SHAP_RAW_NPZ),
            "shap_cache_ae_npz": os.path.abspath(SHAP_AE_NPZ),
        }
    }
    with open(MANIFEST_JSON, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved overall metrics → {CSV_OVERALL}")
    print(f"Saved per-class metrics → {CSV_PERCLASS}")
    print(f"Saved predictions → {PKL_PREDS}")
    print(f"Saved fusion heads → {FUSION_RAW_CKPT}, {FUSION_AE_CKPT}")
    print(f"Saved SHAP caches → {SHAP_RAW_NPZ}, {SHAP_AE_NPZ}")
    print(f"Saved manifest → {MANIFEST_JSON}")
