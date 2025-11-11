#!filepath projects/computational/LSTM/augmentation_sweep.py
from __future__ import annotations
import itertools
import json
import logging
import os
import pickle
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from pydantic import BaseSettings, PositiveFloat, PositiveInt, validator  # type: ignore
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler  # type: ignore

from sklearn.metrics import (  # type: ignore
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)


class Settings(BaseSettings):
    """Configuration for augmentation sweep."""

    # Paths
    data_dir: str = "projects/computational/LSTM/data"
    models_dir: str = "projects/computational/LSTM/models"
    sweep_summary_path: str = "projects/computational/LSTM/augmentation_sweep_results.json"

    # Data files
    cfg_pickle: str = "training_config_clean.pkl"
    train_pickle: str = "train_sessions_processed.pkl"
    dev_pickle: str = "dev_sessions_processed.pkl"

    # Training
    seed: int = 42
    batch_size: PositiveInt = 32
    lr: PositiveFloat = 1e-4
    max_epochs: PositiveInt = 16
    patience: PositiveInt = 6
    window_length: PositiveInt = 10
    sampler_multiplier: PositiveFloat = 1.25
    sampler_cap: PositiveInt = 8_000_000

    # Model
    in_channels_from_config_key: str = "stoi"
    hidden_channels: PositiveInt = 16
    num_layers: PositiveInt = 3
    lstm_dropout: float = 0.3

    # Augmentation grids
    mask_probs: Tuple[float, ...] = (0.05, 0.1, 0.15, 0.2)
    noise_stds: Tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3)
    feat_drop_probs: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.15)

    # Selection
    fixed_threshold: float = 0.80  # operating point τ

    log_level: str = "INFO"

    @validator("log_level")
    def _upper(cls, v: str) -> str:
        return v.upper()

    class Config:
        env_file = ".env"


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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.exception("Failed to unpickle %s", path)
        raise RuntimeError(f"Could not load pickle: {path}") from e


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def augment_window(
    window: torch.Tensor,
    mask_prob: float,
    noise_std: float,
    feature_drop_prob: float,
) -> torch.Tensor:
    """Apply Gaussian noise, single time-step masking, and single feature-channel dropout per window."""
    if noise_std > 0:
        window = window + torch.randn_like(window) * float(noise_std)
    if mask_prob > 0 and np.random.rand() < float(mask_prob):
        t = np.random.randint(0, window.shape[0])
        window[t] = 0
    if feature_drop_prob > 0 and np.random.rand() < float(feature_drop_prob):
        f = np.random.randint(0, window.shape[1])
        window[:, f] = 0
    return window


@dataclass(frozen=True)
class Windows:
    X: torch.Tensor
    y: torch.Tensor


def extract_sliding_windows(
    session_tuples: Sequence[Tuple[torch.Tensor, int]],
    window_length: int,
    augment: bool = False,
    mask_prob: float = 0.0,
    noise_std: float = 0.0,
    feature_drop_prob: float = 0.0,
) -> Windows:
    """Create fixed-length overlapping windows from variable-length sessions."""
    X_list: List[torch.Tensor] = []
    y_list: List[int] = []
    for sess, label in session_tuples:
        if not isinstance(sess, torch.Tensor) or sess.dim() != 2:
            raise ValueError("Each session must be a 2D torch.Tensor [T, F].")
        T = int(sess.shape[0])
        if T < window_length:
            continue
        for s in range(T - window_length + 1):
            w = sess[s : s + window_length].clone()
            if augment:
                w = augment_window(w, mask_prob, noise_std, feature_drop_prob)
            X_list.append(w)
            y_list.append(int(label))
    if not X_list:
        raise ValueError("No windows extracted. Check inputs or window_length.")
    X = torch.stack(X_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return Windows(X=X, y=y)


class LstmModel(nn.Module):
    """LSTM binary classifier producing a single logit per window."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, lstm_dropout: float) -> None:
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
        logits = self.head(out[:, -1, :]).squeeze(-1)
        return logits


def make_loaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_dev: torch.Tensor,
    y_dev: torch.Tensor,
    batch_size: int,
    sampler_multiplier: float,
    sampler_cap: int,
) -> Tuple[DataLoader, DataLoader, Counter]:
    counts = Counter(y_train.tolist())
    n_pos, n_neg = counts.get(1, 0), counts.get(0, 0)
    majority = max(n_pos, n_neg, 1)
    target_samples = int(min(sampler_multiplier * majority, sampler_cap))
    weights = torch.ones(len(y_train), dtype=torch.double)
    if n_pos > 0:
        weights[y_train == 1] = target_samples / float(max(n_pos, 1))
    if n_neg > 0:
        weights[y_train == 0] = target_samples / float(max(n_neg, 1))
    sampler = WeightedRandomSampler(weights=weights, num_samples=target_samples, replacement=True)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, sampler=sampler)
    dev_loader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader, counts


@torch.no_grad()
def dev_metrics_at_threshold(model: nn.Module, dev_loader: DataLoader, device: torch.device, thr: float) -> dict:
    """Compute dev metrics at fixed τ plus threshold-free companions."""
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    for xb, yb in dev_loader:
        xb = xb.to(device)
        y_true.extend(yb.cpu().numpy().tolist())
        y_prob.extend(torch.sigmoid(model(xb)).cpu().numpy().tolist())
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_bin = (y_prob >= thr).astype(int)

    metrics = {
        "f1_at_thr": float(f1_score(y_true, y_bin, zero_division=0)),
        "precision_at_thr": float(precision_score(y_true, y_bin, zero_division=0)),
        "recall_at_thr": float(recall_score(y_true, y_bin, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    return metrics


def train_one_config(
    cfg: Settings,
    device: torch.device,
    train_sess: Sequence[Tuple[torch.Tensor, int]],
    dev_sess: Sequence[Tuple[torch.Tensor, int]],
    mask_prob: float,
    noise_std: float,
    feature_drop_prob: float,
    in_channels: int,
) -> Tuple[float, str, float, dict]:
    """Train one augmentation triple; return selector, ckpt path, best dev loss, and dev metrics."""
    win_train = extract_sliding_windows(
        train_sess, cfg.window_length, augment=True, mask_prob=mask_prob, noise_std=noise_std, feature_drop_prob=feature_drop_prob
    )
    win_dev = extract_sliding_windows(dev_sess, cfg.window_length, augment=False)

    train_loader, dev_loader, counts = make_loaders(
        X_train=win_train.X,
        y_train=win_train.y,
        X_dev=win_dev.X,
        y_dev=win_dev.y,
        batch_size=cfg.batch_size,
        sampler_multiplier=cfg.sampler_multiplier,
        sampler_cap=cfg.sampler_cap,
    )
    n_pos, n_neg = counts.get(1, 0), counts.get(0, 0)

    model = LstmModel(
        in_channels=in_channels,
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
        lstm_dropout=cfg.lstm_dropout,
    ).to(device)

    pos_weight = torch.tensor([float(n_neg) / float(max(n_pos, 1))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_dev, patience_left = float("inf"), int(cfg.patience)
    ckpt_path = os.path.join(cfg.models_dir, f"best_{mask_prob}_{noise_std}_{feature_drop_prob}.pt")

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb = xb.to(device)
                yb = yb.float().to(device)
                dev_loss += float(loss_fn(model(xb), yb).item())
        dev_loss /= max(len(dev_loader), 1)
        logging.info(
            "mask=%.2f noise=%.2f drop=%.2f | epoch=%02d dev_loss=%.4f",
            mask_prob, noise_std, feature_drop_prob, epoch, dev_loss
        )

        if dev_loss < best_dev:
            best_dev = dev_loss
            patience_left = cfg.patience
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logging.info("Early stopping: mask=%.2f noise=%.2f drop=%.2f", mask_prob, noise_std, feature_drop_prob)
                break

    # Reload best and compute metrics at fixed τ
    best_model = LstmModel(
        in_channels=in_channels,
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
        lstm_dropout=cfg.lstm_dropout,
    ).to(device)
    best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    devm = dev_metrics_at_threshold(best_model, dev_loader, device, thr=cfg.fixed_threshold)

    selection_score = devm["f1_at_thr"]  # choose by F1 at τ; swap to precision/recall if desired
    return selection_score, ckpt_path, best_dev, devm


def main() -> None:
    cfg = Settings()
    setup_logging(cfg.log_level)
    set_seed(cfg.seed)
    ensure_dirs(cfg.data_dir, cfg.models_dir)
    device = get_device()
    logging.info("Device: %s", device)

    cfg_path = os.path.join(cfg.data_dir, cfg.cfg_pickle)
    train_path = os.path.join(cfg.data_dir, cfg.train_pickle)
    dev_path = os.path.join(cfg.data_dir, cfg.dev_pickle)

    config = safe_load_pickle(cfg_path)
    train_sess = safe_load_pickle(train_path)
    dev_sess = safe_load_pickle(dev_path)

    in_key = cfg.in_channels_from_config_key
    if in_key not in config:
        raise KeyError(f"Config missing key '{in_key}' for in_channels.")
    in_channels = len(config[in_key])
    logging.info("Detected input feature size (F): %d", in_channels)

    results: List[dict] = []
    combos = list(itertools.product(cfg.mask_probs, cfg.noise_stds, cfg.feat_drop_probs))
    logging.info("Total augmentation combinations: %d", len(combos))

    for mp, ns, fd in combos:
        sel, ckpt, best_dev, devm = train_one_config(cfg, device, train_sess, dev_sess, mp, ns, fd, in_channels)
        row = {
            "mask_prob": float(mp),
            "noise_std": float(ns),
            "feature_drop_prob": float(fd),
            "selection_score": round(float(sel), 6),  # F1@τ
            "best_dev_loss": round(float(best_dev), 6),
            "ckpt": ckpt,
            "thr": float(cfg.fixed_threshold),
            # diagnostics
            "dev_f1_at_thr": float(devm["f1_at_thr"]),
            "dev_precision_at_thr": float(devm["precision_at_thr"]),
            "dev_recall_at_thr": float(devm["recall_at_thr"]),
            "dev_pr_auc": float(devm["pr_auc"]),
            "dev_roc_auc": float(devm["roc_auc"]),
        }
        results.append(row)

    # Sort by performance at the operating point (higher is better)
    results_sorted = sorted(results, key=lambda r: r["selection_score"], reverse=True)

    # Persist sweep summary
    with open(cfg.sweep_summary_path, "w") as f:
        json.dump(results_sorted, f, indent=2)

    # Create a canonical pointer to the best model for downstream stages
    best = results_sorted[0]
    best_ckpt = best["ckpt"]
    best_link = os.path.join(cfg.models_dir, "best_lstm.pt")
    try:
        if os.path.islink(best_link) or os.path.exists(best_link):
            os.remove(best_link)
        os.symlink(os.path.abspath(best_ckpt), best_link)
    except OSError:
        import shutil
        shutil.copy2(best_ckpt, best_link)

    logging.info(
        "Sweep complete. Best by F1@τ=%.2f: %.4f (mask=%.2f, noise=%.2f, drop=%.2f).",
        cfg.fixed_threshold, best["selection_score"], best["mask_prob"], best["noise_std"], best["feature_drop_prob"]
    )
    logging.info("Canonical best LSTM at: %s -> %s", best_link, best_ckpt)


if __name__ == "__main__":
    main()
