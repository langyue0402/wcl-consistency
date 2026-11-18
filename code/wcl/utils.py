"""Shared utilities: config loading, seeds, metrics, logging."""

from __future__ import annotations

import os
import random
import sys
import json
from pathlib import Path

import numpy as np
import yaml
import torch


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal error rate for bonafide scores (higher = more bonafide).

    FAR/FRR are computed on a dense threshold grid and the EER is the value
    at the crossing point (score > t => bonafide).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)  # 1 = bonafide, 0 = spoof
    assert labels.max() <= 1 and labels.min() >= 0
    thr = np.linspace(scores.min() - 1e-6, scores.max() + 1e-6, 10001)
    far = np.array([np.mean(scores[labels == 0] > t) for t in thr])
    frr = np.array([np.mean(scores[labels == 1] <= t) for t in thr])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2)


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold (on bonafide scores) minimizing FAR+FRR on a labeled set."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    thr = np.sort(scores)
    thr = np.concatenate([[thr[0] - 1e-6], thr, [thr[-1] + 1e-6]])
    far = np.array([np.mean(scores[labels == 0] < t) for t in thr])
    frr = np.array([np.mean(scores[labels == 1] >= t) for t in thr])
    return float(thr[np.argmin(far + frr)])


def afe_trim_cells(afe_cfg: dict, stride: int = 1, sr: int = 16000,
                   hop: int = 320) -> float | None:
    """Mean support shift the VAD applies, in grid cells (XLS-R: hop = 320).

    ``afe.vad.extra_trim_s`` plus half the jitter is the mean head trim; this is
    the displacement the ground metric has to price (see
    ``wcl.losses.calibrate_geometry``).  ``hop=320`` is the 20 ms frame rate of
    wav2vec2/XLS-R at 16 kHz.
    """
    vad = (afe_cfg or {}).get("vad") or {}
    trim = (float(vad.get("extra_trim_s", 0.0))
            + 0.5 * float(vad.get("extra_trim_jitter_s", 0.0)))
    if trim <= 0:
        return None
    return float(trim) * float(sr) / float(hop) / max(int(stride), 1)


class Logger:
    """Minimal logger: console + optional csv/json dump."""

    def __init__(self, out_dir: str | None = None):
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        print(msg, flush=True)

    def save_json(self, name: str, obj) -> None:
        if self.out_dir:
            with open(self.out_dir / name, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
