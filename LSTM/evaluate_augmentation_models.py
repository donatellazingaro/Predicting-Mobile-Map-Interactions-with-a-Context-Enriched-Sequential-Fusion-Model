#!filepath projects/computational/LSTM/evaluate_augmentation_models.py
from __future__ import annotations
import glob
import json
import logging
import pickle
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from pydantic import BaseSettings, PositiveInt, validator  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset  # type: ignore


# ---------------------------------------------------------------------
# Directories and configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Evaluate saved augmentation models at a fixed operating point."""

    # Paths are relative to the script location
    data_dir: str = str(BASE_DIR / "data")
    models_dir: str = str(BASE_DIR / "models")
    results_json: str = str(BASE_DIR / "all_test_results.json")
    thresholds_txt: str = str(BASE_DIR / "threshold_analysis.txt")

    cfg_pickle: str = "training_config_clean.pkl"
    test_pickle: str = "test_sessions_processed.pkl"

    window_length: PositiveInt = 10
    batch_size: PositiveInt = 128

    # filenames like best_{mask}_{noise}_{drop}.pt
    filename_pattern: str = r"best_(\d*\.?\d+)_(\d*\.?\d+)_(\d*\.?\d+)\.pt"

    # your operating threshold τ
    fixed_threshold: float = 0.80

    seed: int = 42
    log_level: str = "INFO"

    @validator("log_level")
    def _upper(cls, v: str) -> str:
        return v.upper()

    class Config:
        env_file = ".env"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_load_pickle(path: str):
    """Load a pickle safely, raising FileNotFoundError if missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.exception("Failed to unpickle %s", path)
        raise RuntimeError(f"Could not load pickle: {path}") from e


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Windows:
    """[N, T, F] windows and [N] integer labels."""
    X: torch.Tensor
    y: torch.Tensor


def extract_sliding_windows(
    session_tuples: Sequence[Tuple[torch.Tensor, int]],
    window_length: int,
) -> Windows:
    X_list: List[torch.Tensor] = []
    y_list: List[int] = []
    for sess, label in session_tuples:
        if not isinstance(sess, torch.Tensor) or sess.dim() != 2:
            raise ValueError("Each session must be a 2D torch.Tensor [T, F].")
        T = int(sess.shape[0])
        if T < window_length:
            continue
        for s in range(T - window_length + 1):
            X_list.append(sess[s : s + window_length].clone())
            y_list.append(int(label))
    if not X_list:
        raise ValueError("No windows extracted for test set.")
    X = torch.stack(X_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return Windows(X=X, y=y)


# ---------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------
class LstmModel(nn.Module):
    """LSTM binary classifier outputting a single logit per window."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        num_layers: int = 3,
        lstm_dropout: float = 0.3,
    ) -> None:
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
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h0 = torch.zeros(self.num_layers, b, self.hidden_channels, device=x.device)
        c0 = torch.zeros(self.num_layers, b, self.hidden_channels, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.head(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            probs = torch.sigmoid(model(xb))
            y_prob.extend(probs.detach().cpu().numpy().tolist())
            y_true.extend(yb.detach().cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_prob)


def _safe_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        tn = int(cm[0, 0]) if cm.size > 0 else 0
        return tn, 0, 0, 0
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    """Compute metrics at threshold τ plus threshold-free scores."""
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    tn, fp, fn, tp = _safe_confusion_matrix(y_true, y_pred)
    return {
        "accuracy_at_thr": acc,
        "precision_at_thr": prec,
        "recall_at_thr": rec,
        "f1_at_thr": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float],
) -> List[Tuple[float, float, float, float, float, float]]:
    rows: List[Tuple[float, float, float, float, float, float]] = []
    for thr in thresholds:
        y_bin = (y_prob >= thr).astype(int)
        precision = precision_score(y_true, y_bin, zero_division=0)
        recall = recall_score(y_true, y_bin, zero_division=0)
        f1_pos = f1_score(y_true, y_bin, zero_division=0)
        f1_macro = f1_score(y_true, y_bin, average="macro", zero_division=0)
        negatives = (y_true == 0)
        specificity = float((y_bin[negatives] == 0).mean()) if negatives.any() else float("nan")
        youden_j = (recall if np.isfinite(recall) else 0.0) + (specificity if np.isfinite(specificity) else 0.0) - 1
        rows.append((float(thr), float(precision), float(recall), float(f1_pos), float(f1_macro), float(youden_j)))
    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    """Evaluate all 'best_*.pt' models on the TEST set at τ and sort by F1@τ."""
    cfg = Settings()
    setup_logging(cfg.log_level)
    set_seed(cfg.seed)
    device = get_device()
    logging.info("Device: %s", device)

    cfg_path = Path(cfg.data_dir) / cfg.cfg_pickle
    test_path = Path(cfg.data_dir) / cfg.test_pickle
    config = safe_load_pickle(str(cfg_path))
    test_sess = safe_load_pickle(str(test_path))

    if not isinstance(config, dict) or "stoi" not in config:
        raise KeyError("Config pickle must contain the key 'stoi'.")

    win_test = extract_sliding_windows(test_sess, cfg.window_length)
    pin_mem = torch.cuda.is_available()
    test_loader = DataLoader(
        TensorDataset(win_test.X, win_test.y),
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_mem,
    )
    logging.info(
        "Test windows: %s | label dist: %s",
        tuple(win_test.X.shape),
        dict(Counter(win_test.y.tolist())),
    )

    pattern = re.compile(cfg.filename_pattern)
    model_files = sorted(Path(cfg.models_dir).glob("best_*.pt"))
    logging.info("Found %d model files in %s", len(model_files), cfg.models_dir)

    in_channels = len(config["stoi"])
    results: List[Dict[str, float]] = []
    last_probs: np.ndarray = np.array([])
    last_true: np.ndarray = np.array([])

    for path in model_files:
        m = pattern.search(path.name)
        if not m:
            logging.warning("Skipping unexpected file: %s", path.name)
            continue
        mask_prob, noise_std, feat_drop = map(float, m.groups())
        logging.info(
            "Evaluating %s (mask=%.2f noise=%.2f drop=%.2f) at τ=%.2f",
            path.name, mask_prob, noise_std, feat_drop, cfg.fixed_threshold,
        )

        model = LstmModel(
            in_channels=in_channels,
            hidden_channels=16,
            num_layers=3,
            lstm_dropout=0.3,
        ).to(device)

        try:
            state = torch.load(path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            elif isinstance(state, dict):
                model.load_state_dict(state, strict=False)
            else:
                raise RuntimeError("Checkpoint format not recognized.")
        except Exception:
            logging.exception("Failed to load checkpoint: %s", path)
            continue

        y_true, y_prob = evaluate_model(model, test_loader, device)
        metrics = metrics_from_probs(y_true, y_prob, thr=cfg.fixed_threshold)
        metrics.update(
            {
                "mask_prob": float(mask_prob),
                "noise_std": float(noise_std),
                "feature_drop_prob": float(feat_drop),
                "ckpt": str(path),
                "threshold": float(cfg.fixed_threshold),
            }
        )
        results.append(metrics)
        last_probs = y_prob
        last_true = y_true

    results_sorted = sorted(results, key=lambda r: r["f1_at_thr"], reverse=True)
    Path(cfg.results_json).write_text(json.dumps(results_sorted, indent=2))
    logging.info("Saved metrics sorted by F1@τ=%.2f → %s", cfg.fixed_threshold, cfg.results_json)

    if last_probs.size > 0:
        lo = max(0.10, cfg.fixed_threshold - 0.20)
        hi = min(0.95, cfg.fixed_threshold + 0.20)
        thresholds = [round(t, 2) for t in np.linspace(lo, hi, 9)]
        rows = threshold_sensitivity(last_true, last_probs, thresholds)
        Path(cfg.thresholds_txt).write_text(
            "thr\tprecision\trecall\tf1_pos\tf1_macro\tyouden_j\n" +
            "\n".join(f"{thr:.2f}\t{p:.3f}\t{r:.3f}\t{f1p:.3f}\t{f1m:.3f}\t{j:.3f}" 
                      for thr, p, r, f1p, f1m, j in rows)
        )
        logging.info("Saved threshold analysis to %s (range %.2f–%.2f)", cfg.thresholds_txt, thresholds[0], thresholds[-1])
    else:
        logging.warning("No models evaluated; threshold analysis skipped.")


if __name__ == "__main__":
    main()
