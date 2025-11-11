#!filepath projects/computational/'Fusion Context'/fusion_results_summary.py
"""
fusion_results_summary.py
---------------------------------------
Visualizes and summarizes AE vs Non-AE fusion model performance.

Inputs:
  - outputs/ae_vs_noae_test_overall.csv
  - outputs/ae_vs_noae_test_perclass.csv

Outputs:
  - outputs/ae_vs_noae_overall.png
  - outputs/ae_vs_noae_perclass.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OVERALL = os.path.join(OUTPUT_DIR, "ae_vs_noae_test_overall.csv")
CSV_PERCLASS = os.path.join(OUTPUT_DIR, "ae_vs_noae_test_perclass.csv")

# ================= LOAD =================
df_overall = pd.read_csv(CSV_OVERALL, index_col=0)
df_per_class = pd.read_csv(CSV_PERCLASS)

print("\n Overall Performance:")
print(df_overall.round(3))
print("\n Per-Class Performance (MAP vs NON-MAP):")
print(df_per_class.round(3))

# ================= FIGURE 1: Overall Metrics =================
metrics = ["Accuracy@fixed", "Precision@fixed", "Recall@fixed", "F1@fixed", "ROC-AUC@fixed"]
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, df_overall.loc["Non-AE", metrics], width, label="Without AE", alpha=0.85)
ax.bar(x + width/2, df_overall.loc["AE", metrics], width, label="With AE", alpha=0.85)

ax.set_ylabel("Score")
ax.set_title("Overall Performance: AE vs Non-AE")
ax.set_xticks(x)
ax.set_xticklabels([m.replace("@fixed", "") for m in metrics])
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "ae_vs_noae_overall.png")
plt.savefig(out1, dpi=300)
plt.close()
print(f" Saved {out1}")

# ================= FIGURE 2: Per-Class Metrics =================
per_class_metrics = ["Precision", "Recall", "F1"]
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for i, cls in enumerate(["MAP", "NON-MAP"]):
    df_cls = df_per_class[df_per_class["Class"] == cls]
    ae_vals = df_cls[df_cls["Setting"] == "AE"][per_class_metrics].mean()
    noae_vals = df_cls[df_cls["Setting"] == "Non-AE"][per_class_metrics].mean()

    x = np.arange(len(per_class_metrics))
    axes[i].bar(x - width/2, noae_vals, width, label="Without AE", alpha=0.85)
    axes[i].bar(x + width/2, ae_vals, width, label="With AE", alpha=0.85)
    axes[i].set_title(cls)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(per_class_metrics)
    axes[i].set_ylim(0, 1)
    axes[i].legend()

fig.suptitle("Per-Class Performance Comparison", fontsize=14)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "ae_vs_noae_perclass.png")
plt.savefig(out2, dpi=600)
plt.close()
print(f" Saved {out2}")
