#!filepath projects/computational/Fusion Context/fusion_explainability.py
"""
fusion_explainability.py
---------------------------------------
Computes SHAP explanations for the LSTM–Context Fusion models (AE vs Non-AE).

Outputs:
  • SHAP summary plots (.png)
  • Mean absolute SHAP value tables (.csv)
  • Separate per-class analyses (MAP vs NON-MAP)
  • Automatically reuses SHAP caches saved by fusion_run.py if available
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch  # type: ignore
import shap   # type: ignore

# ================== CONFIG ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_LENGTH = 10

# Subsampling for SHAP background (keeps runtime/memory under control)
BACKGROUND_MAX = 4096     # size of background reference set
EVAL_MAX = None           # optionally limit total points evaluated (None = all)

# Base directories (relative to this script)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LSTM_DIR   = os.path.join(BASE_DIR, "..", "LSTM", "models")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_LOCAL = os.path.join(BASE_DIR, "data")
DATA_HPC   = "/data/dzinga"
DATA_DIR   = DATA_HPC if os.path.exists(os.path.join(DATA_HPC, "aligned_dev_sessions.pkl")) else DATA_LOCAL

# Checkpoints (relative, reproducible paths)
LSTM_CKPT      = os.path.join(LSTM_DIR,  "best_lstm.pt")
AE_ENCODER_CKPT= os.path.join(MODEL_DIR, "context_encoder.pt")
FUSION_AE_CKPT = os.path.join(MODEL_DIR, "fusion_head_ae.pt")
FUSION_RAW_CKPT= os.path.join(MODEL_DIR, "fusion_head_noae.pt")

# SHAP caches (produced by fusion_run.py)
SHAP_RAW_NPZ   = os.path.join(OUTPUT_DIR, "shap_cache_noae_dev.npz")
SHAP_AE_NPZ    = os.path.join(OUTPUT_DIR, "shap_cache_ae_dev.npz")
MANIFEST_JSON  = os.path.join(OUTPUT_DIR, "manifest.json")

OUT_DIR = OUTPUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Using DATA_DIR = {DATA_DIR}")

# ================== IO HELPERS ==================
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def require(path: str, hint: str = ""):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}\n{hint}")

# ================== DATA LOADING (fallback path) ==================
def load_fusion_context(pkl_path):
    data = load_pickle(pkl_path)
    ids = [x[0] for x in data]
    ctx = np.array([x[1] for x in data], dtype=np.float32)
    labels = np.array([x[2] for x in data], dtype=np.float32)
    return ids, ctx, labels

def load_aligned_sessions(pkl_path):
    return load_pickle(pkl_path)

def extract_sliding_windows_with_context(aligned_sessions, context_lookup, window_length=10):
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

# ================== MODELS ==================
class LstmModel(torch.nn.Module):
    """Pretrained LSTM feature extractor"""
    def __init__(self, in_channels, hidden_channels, num_layers, lstm_dropout=0.3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
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

class LSTMContextFusionHead(torch.nn.Module):
    """Fusion layer combining LSTM and context features"""
    def __init__(self, lstm_hidden_size, context_dim, fusion_hidden=32, dropout=0.3):
        super().__init__()
        self.context_proj = torch.nn.Sequential(
            torch.nn.Linear(context_dim, fusion_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.fusion_head = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden_size + fusion_hidden, fusion_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fusion_hidden, 1),
        )

    def forward(self, seq_repr, context_vec):
        ctx = self.context_proj(context_vec)
        fused = torch.cat([seq_repr, ctx], dim=1)
        return self.fusion_head(fused).squeeze(-1)

class ContextEncoder(torch.nn.Module):
    """Autoencoder encoder branch"""
    def __init__(self, input_dim=8, latent_dim=3):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)

# ================== CACHE LOADING ==================
def load_shap_cache(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    seq_repr = data["seq_repr"]
    ctx = data["ctx"]
    y = data["y"]
    mode = str(data["mode"][0]) if "mode" in data.files else "Unknown"
    feature_names = (data["feature_names"].tolist()
                     if "feature_names" in data.files and len(data["feature_names"]) > 0
                     else None)
    return seq_repr, ctx, y, feature_names, mode

def maybe_load_manifest():
    if os.path.exists(MANIFEST_JSON):
        with open(MANIFEST_JSON, "r") as f:
            return json.load(f)
    return None

# ================== Fallback: compute dev embeddings if no cache ==================
def compute_dev_embeddings_and_context():
    print("Caches not found — computing dev seq_repr + contexts from pickles...")
    # Dev pickles
    DEV_SESS_PATH = os.path.join(DATA_DIR, "aligned_dev_sessions.pkl")
    DEV_CTX_PATH  = os.path.join(DATA_DIR, "fusion_context_dev.pkl")
    require(DEV_SESS_PATH, "Export dev sessions before running SHAP.")
    require(DEV_CTX_PATH,  "Export dev context before running SHAP.")

    dev_sess = load_aligned_sessions(DEV_SESS_PATH)
    dev_ids, dev_ctx_all, y_dev_seq = load_fusion_context(DEV_CTX_PATH)
    dev_ctx_lookup = dict(zip(dev_ids, dev_ctx_all))

    X_dev_seq, dev_ctx, y_dev_seq = extract_sliding_windows_with_context(
        dev_sess, dev_ctx_lookup, WINDOW_LENGTH
    )

    # AE compression
    require(AE_ENCODER_CKPT, "Train/export context encoder to models/context_encoder.pt")
    ae_encoder = ContextEncoder(input_dim=dev_ctx.shape[1], latent_dim=3).to(DEVICE)
    state_dict = torch.load(AE_ENCODER_CKPT, map_location=DEVICE)
    # Accept either plain encoder keys or prefixed with "encoder."
    if all(k in state_dict for k in ["0.weight", "0.bias", "2.weight", "2.bias"]):
        ae_encoder.encoder.load_state_dict(state_dict, strict=True)
    else:
        encoder_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
        ae_encoder.encoder.load_state_dict(encoder_state, strict=True)
    ae_encoder.eval()

    with torch.no_grad():
        dev_ctx_ae = ae_encoder(torch.tensor(dev_ctx, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    # LSTM seq representation
    require(LSTM_CKPT, "Point LSTM_CKPT to the canonical best_lstm.pt.")
    lstm_model = LstmModel(in_channels=X_dev_seq.shape[2], hidden_channels=16, num_layers=3).to(DEVICE)
    lstm_state = torch.load(LSTM_CKPT, map_location=DEVICE)
    # tolerate possible head params in ckpt
    lstm_model.load_state_dict({k: v for k, v in lstm_state.items() if "linear" not in k}, strict=False)
    lstm_model.eval()

    # Batch to avoid OOM
    seq_repr_dev = []
    X = torch.tensor(X_dev_seq, dtype=torch.float32, device=DEVICE)
    bs = 2048
    with torch.no_grad():
        for i in range(0, len(X), bs):
            seq_repr_dev.append(lstm_model(X[i:i+bs]).cpu().numpy())
    seq_repr_dev = np.vstack(seq_repr_dev)

    # Feature names
    ae_feature_names = [f"AE_latent_{i+1}" for i in range(dev_ctx_ae.shape[1])]
    raw_feature_names = [
        "betweenness", "closeness", "degree", "orientation_entropy",
        "circuity", "Sense of Direction", "Spatial Anxiety", "Gender"
    ]

    return (
        (seq_repr_dev, dev_ctx_ae, y_dev_seq, ae_feature_names, "AE"),
        (seq_repr_dev, dev_ctx,    y_dev_seq, raw_feature_names, "Non-AE"),
    )

# ================== SHAP CORE ==================
def choose_background(X: np.ndarray, k: int | None) -> np.ndarray:
    if k is None or X.shape[0] <= k:
        return X
    idx = np.random.RandomState(0).choice(X.shape[0], size=k, replace=False)
    return X[idx]

def run_shap_one_mode(seq_repr: np.ndarray,
                      ctx_data: np.ndarray,
                      fusion_ckpt: str,
                      title: str,
                      feature_names: list[str]):
    """Compute SHAP on context part (holding seq_repr fixed as part of model input)."""
    # Load fusion head
    require(fusion_ckpt, f"Missing fusion head: {fusion_ckpt}")
    fusion_head = LSTMContextFusionHead(lstm_hidden_size=16,
                                        context_dim=ctx_data.shape[1],
                                        fusion_hidden=32).to(DEVICE)
    fusion_head.load_state_dict(torch.load(fusion_ckpt, map_location=DEVICE))
    fusion_head.eval()

    # Combined input = [seq_repr | ctx]
    combined = np.hstack([seq_repr, ctx_data])

    # Optional subsampling for evaluation set
    if EVAL_MAX is not None and combined.shape[0] > EVAL_MAX:
        rs = np.random.RandomState(1)
        keep = rs.choice(combined.shape[0], size=EVAL_MAX, replace=False)
        combined = combined[keep]
        ctx_view = ctx_data[keep]
    else:
        ctx_view = ctx_data

    # Background set for SHAP (Kernel/Explainer auto-select)
    background = choose_background(combined, BACKGROUND_MAX)

    def combined_forward(arr):
        arr = np.asarray(arr, dtype=np.float32)
        s = arr[:, :seq_repr.shape[1]]
        c = arr[:, seq_repr.shape[1]:]
        s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        c_t = torch.tensor(c, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits = fusion_head(s_t, c_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    # SHAP: we explain the **context** slice by passing the concatenation
    explainer = shap.Explainer(combined_forward, background)
    sv = explainer(combined)

    # Slice out SHAP values corresponding to the context fields
    # (features aligned to combined columns)
    ctx_start = seq_repr.shape[1]
    ctx_sv = sv[:, ctx_start:]

    # Plot
    plt.figure()
    shap.summary_plot(
        ctx_sv,
        features=ctx_view,
        feature_names=feature_names,
        show=False,
        plot_size=(8, 5)
    )
    out_png = os.path.join(OUT_DIR, f"{title.replace(' ', '_')}.png")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()

    # Table
    mean_abs = np.abs(ctx_sv.values).mean(axis=0)
    df = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs})
    out_csv = os.path.join(OUT_DIR, f"{title.replace(' ', '_')}_values.csv")
    df.to_csv(out_csv, index=False)

    print(f"Saved SHAP plot: {out_png}")
    print(f"Saved SHAP values: {out_csv}")

# ================== MAIN FLOW ==================
def main():
    print("=== SHAP Explainability ===")

    # 1) Try caches from fusion_run.py
    have_raw = os.path.exists(SHAP_RAW_NPZ)
    have_ae  = os.path.exists(SHAP_AE_NPZ)

    if have_raw and have_ae:
        print("Found SHAP caches. Loading NPZ files…")
        seq_repr_ae,  ctx_ae,  y_dev_ae,  ae_names,  mode_ae  = load_shap_cache(SHAP_AE_NPZ)
        seq_repr_raw, ctx_raw, y_dev_raw, raw_names, mode_raw = load_shap_cache(SHAP_RAW_NPZ)
        # Sanity: seq_repr from both caches should match (they come from the same dev windows)
        # If not, we still proceed independently.
    else:
        # 2) No caches → compute dev embeddings/contexts (fallback)
        (seq_repr_ae, ctx_ae, y_dev_ae, ae_names, mode_ae), \
        (seq_repr_raw, ctx_raw, y_dev_raw, raw_names, mode_raw) = compute_dev_embeddings_and_context()

    # 3) Run SHAP (overall)
    print("\nRunning SHAP (overall): AE vs Non-AE")
    run_shap_one_mode(seq_repr_ae,  ctx_ae,  FUSION_AE_CKPT,  "With_AE_Overall",    ae_names)
    run_shap_one_mode(seq_repr_raw, ctx_raw, FUSION_RAW_CKPT, "Without_AE_Overall", raw_names)

    # 4) Per-class (MAP=1, NON-MAP=0), if labels available
    # We prefer y from AE cache path; both should be identical in order
    if y_dev_ae is not None:
        print("\nRunning SHAP per class…")
        for label_value, label_name in [(1, "MAP"), (0, "NON-MAP")]:
            mask = (y_dev_ae == label_value)
            if mask.sum() == 0:
                continue
            run_shap_one_mode(seq_repr_ae[mask],  ctx_ae[mask],  FUSION_AE_CKPT,  f"AE_{label_name}",    ae_names)
            run_shap_one_mode(seq_repr_raw[mask], ctx_raw[mask], FUSION_RAW_CKPT, f"NoAE_{label_name}",  raw_names)

    print("\nExplainability analysis complete.")
    print(f"All outputs saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()
