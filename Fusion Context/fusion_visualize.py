#!filepath projects/computational/'Fusion Context'/fusion_visualize.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix  # type: ignore

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PKL_PREDS = os.path.join(OUTPUT_DIR, "ae_vs_noae_test_preds.pkl")
CSV_OVERALL = os.path.join(OUTPUT_DIR, "ae_vs_noae_test_overall.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Matplotlib & Seaborn aesthetics ===
sns.set(style="whitegrid", font="DejaVu Sans", font_scale=1.2)
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "savefig.bbox": "tight"
})

# === Load saved predictions ===
with open(PKL_PREDS, "rb") as f:
    data = pickle.load(f)

preds_dict = {k: v["probs"] for k, v in data.items()}
targets_dict = {k: v["targets"] for k, v in data.items()}

# === Load operating thresholds (fallback to 0.80) ===
tau_default = 0.80
taus = {"AE": tau_default, "Non-AE": tau_default}
if os.path.exists(CSV_OVERALL):
    df_overall = pd.read_csv(CSV_OVERALL, index_col=0)
    # rows are "AE" and "Non-AE" from fusion_run.py
    for tag in ["AE", "Non-AE"]:
        col = "Threshold_fixed"
        if tag in df_overall.index and col in df_overall.columns:
            try:
                taus[tag] = float(df_overall.loc[tag, col])
            except Exception:
                taus[tag] = tau_default

# === Colors ===
palette = {
    "AE": "#1f77b4",       # deep blue
    "Non-AE": "#ff7f0e",   # amber orange
}

def _closest_point_on_pr(y_true, y_score, tau):
    """Return (recall, precision) at the threshold nearest to tau."""
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thr)=len(precision)-1
    if thr.size == 0:
        # degenerate case; return last point
        return float(recall[-1]), float(precision[-1])
    idx = int(np.argmin(np.abs(thr - tau)))
    # Align to PR arrays: idx maps to precision[idx+1], recall[idx+1]
    return float(recall[idx + 1]), float(precision[idx + 1])

def _closest_point_on_roc(y_true, y_score, tau):
    """Return (fpr, tpr) at the threshold nearest to tau."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    if thr.size == 0:
        return float(fpr[-1]), float(tpr[-1])
    idx = int(np.argmin(np.abs(thr - tau)))
    # roc_curve thresholds align one-to-one with fpr/tpr
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx]), float(tpr[idx])

# === Precision–Recall Curve (with operating-point markers) ===
plt.figure(figsize=(6, 5))
for tag in ["AE", "Non-AE"]:
    preds = preds_dict[tag]
    targets = targets_dict[tag]
    precision, recall, _ = precision_recall_curve(targets, preds)
    plt.plot(recall, precision, lw=2.2, color=palette[tag], label=tag)
    # Marker at τ
    r_tau, p_tau = _closest_point_on_pr(targets, preds, taus[tag])
    plt.plot([r_tau], [p_tau], marker="o", ms=7, color=palette[tag])
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision–Recall Curve (markers at operating τ)", fontsize=14, pad=12)
plt.legend(frameon=True, facecolor="white", edgecolor="lightgray", title="Model")
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve_compare.png"))
plt.close()

# === ROC Curve (with operating-point markers) ===
plt.figure(figsize=(6, 5))
for tag in ["AE", "Non-AE"]:
    preds = preds_dict[tag]
    targets = targets_dict[tag]
    fpr, tpr, _ = roc_curve(targets, preds)
    plt.plot(fpr, tpr, lw=2.2, color=palette[tag], label=tag)
    # Marker at τ
    fpr_tau, tpr_tau = _closest_point_on_roc(targets, preds, taus[tag])
    plt.plot([fpr_tau], [tpr_tau], marker="o", ms=7, color=palette[tag])
plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate (Recall)", fontsize=12)
plt.title("ROC Curve (markers at operating τ)", fontsize=14, pad=12)
plt.legend(frameon=True, facecolor="white", edgecolor="lightgray", title="Model")
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve_compare.png"))
plt.close()

# === Confusion Matrices at operating τ ===
for tag in ["AE", "Non-AE"]:
    preds = preds_dict[tag]
    targets = targets_dict[tag]
    tau = taus[tag]
    y_pred = (preds >= tau).astype(int)
    cm = confusion_matrix(targets, y_pred, labels=[0, 1], normalize="true")

    plt.figure(figsize=(4.8, 4.4))
    sns.heatmap(
        cm,
        annot=True, fmt=".2f",
        cmap="Blues", cbar=False, square=True,
        linewidths=0.6, linecolor="white",
        xticklabels=["Non-MAP", "MAP"],
        yticklabels=["Non-MAP", "MAP"],
    )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel(f"Predicted Label (τ = {tau:.2f})", fontsize=12)
    plt.title(f"Confusion Matrix — {tag}", fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"conf_matrix_{tag}.png"))
    plt.close()

print("Visualization complete.")
print(f"Saved plots in: {OUTPUT_DIR}")
