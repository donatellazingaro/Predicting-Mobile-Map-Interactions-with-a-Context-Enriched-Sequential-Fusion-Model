#!filepath projects/computational/Fusion Context/fusion_shap_posneg.py
"""
fusion_shap_posneg.py
---------------------------------------
Computes and visualizes relative positive vs negative SHAP contributions
for AE and Non-AE context features (overall and per class).

Inputs:
  - outputs/shap_cache_ae_dev.npz
  - outputs/shap_cache_noae_dev.npz

Each cache includes:
  seq_repr, ctx, y, feature_names, mode

Outputs:
  - shap_ae_posneg_overall.png
  - shap_noae_posneg_overall.png
  - shap_ae_posneg_map.png
  - shap_noae_posneg_map.png
  - shap_ae_posneg_nonmap.png
  - shap_noae_posneg_nonmap.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import shap  # type: ignore
import torch  # type: ignore

# ===================== CONFIG =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHAP_AE_NPZ = os.path.join(OUTPUT_DIR, "shap_cache_ae_dev.npz")
SHAP_NOAE_NPZ = os.path.join(OUTPUT_DIR, "shap_cache_noae_dev.npz")

FUSION_AE_CKPT = os.path.join(MODEL_DIR, "fusion_head_ae.pt")
FUSION_RAW_CKPT = os.path.join(MODEL_DIR, "fusion_head_noae.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== MODEL =====================
class LSTMContextFusionHead(torch.nn.Module):
    """Fusion layer combining LSTM and context features."""
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


# ===================== LOADERS =====================
def load_shap_cache(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing SHAP cache: {path}")
    data = np.load(path, allow_pickle=True)
    return {
        "seq_repr": data["seq_repr"],
        "ctx": data["ctx"],
        "y": data["y"],
        "feature_names": data["feature_names"].tolist(),
        "mode": str(data["mode"][0]),
    }


# ===================== SHAP COMPUTATION =====================
def compute_shap_values(seq_repr, ctx, ckpt_path, feature_names, title):
    """Compute signed SHAP values for context features."""
    fusion_head = LSTMContextFusionHead(lstm_hidden_size=16,
                                        context_dim=ctx.shape[1]).to(DEVICE)
    fusion_head.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    fusion_head.eval()

    combined = np.hstack([seq_repr, ctx])
    bg_size = min(4096, len(combined))
    background = combined[np.random.RandomState(0).choice(len(combined), size=bg_size, replace=False)]

    def model_forward(arr):
        arr = np.asarray(arr, dtype=np.float32)
        s = arr[:, :seq_repr.shape[1]]
        c = arr[:, seq_repr.shape[1]:]
        s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        c_t = torch.tensor(c, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits = fusion_head(s_t, c_t)
            return torch.sigmoid(logits).cpu().numpy()

    explainer = shap.Explainer(model_forward, background)
    shap_values = explainer(combined)
    ctx_shap = shap_values.values[:, seq_repr.shape[1]:]

    print(f"SHAP computed for {title}: {ctx_shap.shape}")
    return ctx_shap


# ===================== PLOTTING =====================
def plot_shap_pos_neg(ctx_shap, feature_names, title, save_path):
    """Plot relative positive vs negative SHAP impact per feature (mean-based)."""
    pos_impacts = np.array([
        np.mean(v[v > 0]) if np.any(v > 0) else 0.0 for v in ctx_shap.T
    ])
    neg_impacts = np.array([
        np.mean(v[v < 0]) if np.any(v < 0) else 0.0 for v in ctx_shap.T
    ])

    pos_abs = np.abs(pos_impacts)
    neg_abs = np.abs(neg_impacts)
    total_sum = pos_abs.sum() + neg_abs.sum()
    if total_sum == 0:
        print(f"[skip] Zero total SHAP magnitude for {title}")
        return

    pos_pct = (pos_abs / total_sum) * 100
    neg_pct = (neg_abs / total_sum) * 100

    order = np.argsort(pos_abs + neg_abs)
    ylabels = np.array(feature_names)[order]

    plt.figure(figsize=(9, 6))
    plt.barh(ylabels, neg_pct[order],
             color="royalblue", alpha=0.85, label="Negative impact (toward NON-MAP)")
    plt.barh(ylabels, pos_pct[order], left=neg_pct[order],
             color="crimson", alpha=0.85, label="Positive impact (toward MAP)")

    for i, idx in enumerate(order):
        if neg_pct[idx] > 0:
            plt.text(neg_pct[idx] / 2, i, f"{neg_pct[idx]:.1f}%",
                     va="center", ha="center", color="white", fontsize=9)
        if pos_pct[idx] > 0:
            plt.text(neg_pct[idx] + pos_pct[idx] / 2, i, f"{pos_pct[idx]:.1f}%",
                     va="center", ha="center", color="white", fontsize=9)

    plt.xlabel("Share of total mean absolute SHAP impact (%)", fontsize=12)
    plt.title(title, fontsize=14, weight="bold")
    plt.legend(loc="lower right", fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")

# ===================== MAIN =====================
if __name__ == "__main__":
    print("=== SHAP Polarity Visualization (AE + Non-AE) ===")

    # --- Load caches ---
    ae_data = load_shap_cache(SHAP_AE_NPZ)
    noae_data = load_shap_cache(SHAP_NOAE_NPZ)

    # --- Compute SHAP values ---
    ae_shap = compute_shap_values(ae_data["seq_repr"], ae_data["ctx"],
                                  FUSION_AE_CKPT, ae_data["feature_names"], "AE")
    noae_shap = compute_shap_values(noae_data["seq_repr"], noae_data["ctx"],
                                    FUSION_RAW_CKPT, noae_data["feature_names"], "Non-AE")

    # --- Overall polarity plots ---
    plot_shap_pos_neg(ae_shap, ae_data["feature_names"],
                      "Latent Context Features (AE) — Positive vs Negative Impact (Overall)",
                      os.path.join(OUTPUT_DIR, "shap_ae_posneg_overall.png"))

    plot_shap_pos_neg(noae_shap, noae_data["feature_names"],
                      "8-D Context Features — Positive vs Negative Impact (Overall)",
                      os.path.join(OUTPUT_DIR, "shap_noae_posneg_overall.png"))

    # --- Per-class polarity plots ---
    for label_value, label_name in [(1, "MAP"), (0, "NON-MAP")]:
        mask_ae = ae_data["y"] == label_value
        mask_noae = noae_data["y"] == label_value

        if mask_ae.sum() > 0:
            plot_shap_pos_neg(ae_shap[mask_ae], ae_data["feature_names"],
                              f"Latent Context Features (AE) — Positive vs Negative Impact ({label_name} only)",
                              os.path.join(OUTPUT_DIR, f"shap_ae_posneg_{label_name.lower()}.png"))

        if mask_noae.sum() > 0:
            plot_shap_pos_neg(noae_shap[mask_noae], noae_data["feature_names"],
                              f"8-D Context Features — Positive vs Negative Impact ({label_name} only)",
                              os.path.join(OUTPUT_DIR, f"shap_noae_posneg_{label_name.lower()}.png"))

    print("\n All SHAP polarity plots (overall and per class) generated successfully.")
