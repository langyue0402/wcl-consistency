"""ASVspoof 2019 LA dataset and feature-cache access.

Directory layout expected under `data_root` (see README for download info):

    ASVspoof2019_LA_train/flac/LA_T_*.flac
    ASVspoof2019_LA_dev/flac/LA_D_*.flac
    ASVspoof2019_LA_eval/flac/LA_E_*.flac
    ASVspoof2019_LA_cm_protocols/LA.cm.train.trn.txt
    ASVspoof2019_LA_cm_protocols/LA.cm.dev.trl.txt
    ASVspoof2019_LA_cm_protocols/LA.cm.eval.trl.txt

Protocol lines:  SPKID FILE - - BONAFIDE|SPOOF   (train)
                 SPKID FILE - ATTACK BONAFIDE|SPOOF (dev/eval)

Features are pre-extracted by scripts/preprocess.py into `cache_dir`:

    cache_dir/{split}/{phase}/{utt_id}.npy     # [T, F] log-mel, padded + normalized
    cache_dir/{split}/meta.json                # list of {utt, path, label, attack}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .afe import STAGES, cascade_prefix

LABEL_BONAFIDE = 1
LABEL_SPOOF = 0


def parse_protocol(proto_path: Path, flac_dir: Path) -> list[dict]:
    """Parse one ASVspoof 2019 LA protocol file."""
    entries = []
    for line in open(proto_path, encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        spk, utt = parts[0], parts[1]
        label = 1 if parts[-1].lower() == "bonafide" else 0
        attack = parts[-2] if len(parts) >= 5 and parts[-2] != "-" else "bonafide"
        wav = flac_dir / f"{utt}.flac"
        if not wav.exists():
            continue
        entries.append({"utt": utt, "path": str(wav), "label": label,
                        "attack": attack})
    return entries


def load_meta(cache_dir: Path, split: str) -> list[dict]:
    with open(cache_dir / split / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def phases() -> list[str]:
    return ["clean"] + STAGES  # vad = full cascade prefix


def phase_npy(cache_dir: Path, split: str, phase: str, utt: str) -> np.ndarray:
    p = cache_dir / split / phase / f"{utt}.npy"
    return np.load(p)


class WCLTrainDataset(Dataset):
    """Dual-view training set.

    Each sample returns (clean_view, distorted_view, label). The distorted
    view is drawn from the AFE stage cache (random stage by default, or the
    full cascade), both views cropped at the SAME time offset.
    """

    def __init__(self, meta: list[dict], cache_dir: Path, split: str = "train",
                 view_schedule: str = "random_stage",
                 stage_probs: list[float] | None = None,
                 random_crop: bool = False, crop_frames: int | None = None,
                 seed: int = 0, feature_type: str = "logmel"):
        self.meta = meta
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.view_schedule = view_schedule
        # the clean view is always the first return value, so putting mass on
        # "clean" as the *distorted* view only wastes a dual-view pair
        self.stage_probs = np.array(stage_probs or [0.0, 0.25, 0.25, 0.25, 0.25],
                                    dtype=np.float64)
        self.stage_probs /= self.stage_probs.sum()
        self.random_crop = random_crop
        self.crop_frames = crop_frames
        self.feature_type = feature_type
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._worker_rng: np.random.Generator | None = None

    def _rng(self) -> np.random.Generator:
        """Per-worker RNG.

        ``__init__`` runs in the parent process, so every DataLoader worker
        would inherit an identical copy of ``self.rng`` and emit the same
        stage/crop sequence. Reseeding from the worker id restores the
        diversity the sampling is supposed to provide.
        """
        if self._worker_rng is None:
            try:
                from torch.utils.data import get_worker_info
                info = get_worker_info()
            except Exception:                     # pragma: no cover
                info = None
            wid = int(info.id) if info is not None else 0
            self._worker_rng = np.random.default_rng(self.seed * 1009 + wid)
        return self._worker_rng

    def __len__(self) -> int:
        return len(self.meta)

    def _read(self, phase: str, utt: str) -> np.ndarray:
        return phase_npy(self.cache_dir, self.split, phase, utt)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        item = self.meta[idx]
        utt, label = item["utt"], item["label"]

        clean = self._read("clean", utt)
        T = clean.shape[0]

        if self.view_schedule == "full_cascade":
            dist_phase = "vad"
        else:
            dist_phase = self._rng().choice(phases(), p=self.stage_probs)
        distorted = self._read(dist_phase, utt)

        if self.feature_type == "raw":
            # [N] waveform: no time cropping here (the cache is already fixed
            # length and both views share the same samples)
            return (torch.from_numpy(clean).unsqueeze(0),
                    torch.from_numpy(distorted).unsqueeze(0), label)

        if self.random_crop and self.crop_frames and self.crop_frames < T:
            lo = self._rng().integers(0, T - self.crop_frames + 1)
            clean = clean[lo: lo + self.crop_frames]
            distorted = distorted[lo: lo + self.crop_frames]

        clean = torch.from_numpy(clean).unsqueeze(0)        # [1, T, F]
        distorted = torch.from_numpy(distorted).unsqueeze(0)  # [1, T, F]
        return clean, distorted, label


class WCLPhaseDataset(Dataset):
    """Single-phase evaluation set (used per AFE stage prefix)."""

    def __init__(self, meta: list[dict], cache_dir: Path, split: str, phase: str,
                 feature_type: str = "logmel"):
        self.meta = meta
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.phase = phase
        self.feature_type = feature_type

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        item = self.meta[idx]
        feat = phase_npy(self.cache_dir, self.split, self.phase, item["utt"])
        if self.feature_type == "raw":
            return torch.from_numpy(feat).unsqueeze(0), item["label"]
        x = torch.from_numpy(feat).unsqueeze(0)  # [1, T, F]
        return x, item["label"]


def build_train_loader(cfg: dict, split: str = "train", shuffle: bool = True):
    from torch.utils.data import DataLoader
    cache_dir = Path(cfg["data"]["cache_dir"])
    meta = load_meta(cache_dir, split)
    ftype = cfg["feature"].get("type", "logmel")
    ds = WCLTrainDataset(
        meta, cache_dir, split=split,
        view_schedule=cfg["train"].get("view_schedule", "random_stage"),
        stage_probs=cfg["train"].get("stage_probs"),
        random_crop=cfg["feature"].get("random_crop", False),
        crop_frames=cfg["feature"].get("crop_frames"),
        seed=cfg["train"].get("seed", 0),
        feature_type=ftype,
    )
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                      shuffle=shuffle, num_workers=cfg["train"].get("num_workers", 2),
                      pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle)


def build_eval_loader(cfg: dict, split: str, phase: str, shuffle: bool = False):
    from torch.utils.data import DataLoader
    cache_dir = Path(cfg["data"]["cache_dir"])
    meta = load_meta(cache_dir, split)
    ds = WCLPhaseDataset(meta, cache_dir, split, phase,
                         feature_type=cfg["feature"].get("type", "logmel"))
    return DataLoader(ds, batch_size=cfg["eval"].get("batch_size", 32),
                      shuffle=shuffle,
                      num_workers=cfg["eval"].get("num_workers", 2),
                      pin_memory=torch.cuda.is_available())
