#!/usr/bin/env python3
"""Pre-extract feature caches for all AFE phase prefixes.

Two feature types are supported (``feature.type`` in the config):

  * ``logmel`` — [T, F] log-mel spectrogram, input of the 2D-CNN encoder;
  * ``raw``    — [N] standardized waveform (N = sr * duration), input of the
                 XLS-R + AASIST backbone used by TFCL.

For every utterance in a split, up to five arrays are produced:
    clean, aec, ns (AEC+NS), agc (AEC+NS+AGC), vad (AEC+NS+AGC+VAD)
stored as {cache_dir}/{split}/{phase}/{utt_id}.npy, plus {split}/meta.json.

The work is CPU-bound (FLAC decode + four AFE stages per utterance), so it runs
over a process pool: serially this is ~14 h for the full LA corpus, which is
billed time on a rented GPU box.

Usage:
    python scripts/preprocess.py --config configs/three_arm_phys.yaml --splits train dev
    python scripts/preprocess.py --config configs/three_arm_phys.yaml --splits eval --workers 16
    python scripts/preprocess.py --config configs/three_arm_phys.yaml --splits eval \\
        --afe-seed 7 --cache-dir cache/phys_v4_holdout      # unseen distortion draw
    python scripts/preprocess.py --config configs/smoke.yaml --splits train --limit 32
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from wcl.afe import apply_cascade, cascade_prefix, STAGES
from wcl.dataset import parse_protocol
from wcl.features import (log_mel, pad_or_crop, normalize, pad_or_crop_wave,
                          normalize_wave)
from wcl.utils import load_config

_G: dict = {}          # per-process globals (config is inherited via fork/spawn)


def _init(cfg: dict, split: str, phases: list[str], afe_seed: int,
          cache_dir: str, dtype: str) -> None:
    _G.clear()
    _G.update(cfg=cfg, split=split, phases=phases, afe_seed=afe_seed,
              cache_dir=cache_dir, dtype=dtype)
    for p in phases:
        (Path(cache_dir) / split / p).mkdir(parents=True, exist_ok=True)


def _work(args) -> dict:
    idx, item = args
    import librosa

    cfg = _G["cfg"]
    split, phases = _G["split"], _G["phases"]
    afe_cfg, fcfg = cfg["afe"], cfg["feature"]
    sr = int(cfg["data"]["sr"])
    ftype = fcfg.get("type", "logmel")
    cache_dir = Path(_G["cache_dir"])
    np_dtype = np.float16 if _G["dtype"] == "float16" else np.float32

    x, _ = librosa.load(item["path"], sr=sr, mono=True)
    x = x.astype(np.float32)
    # per-utterance seed: with a fixed seed every utterance receives the *same*
    # random draw, which silently turns the distortion family into a point
    utt_seed = (int(_G["afe_seed"]) * 1_000_003 + idx * 7919) % (2 ** 31 - 1)
    for phase in phases:
        prefix = cascade_prefix(STAGES, phase) if phase != "clean" else []
        y = apply_cascade(x, sr, afe_cfg, seed=utt_seed, stages=prefix) if prefix else x
        if ftype == "raw":
            feat = pad_or_crop_wave(y, int(round(sr * float(fcfg.get("duration", 4.0)))))
            feat = normalize_wave(feat)
        else:
            feat = log_mel(y, sr, fcfg["n_fft"], fcfg["hop"], fcfg["n_mels"])
            feat = pad_or_crop(feat, fcfg["max_frames"])
            feat = normalize(feat, fcfg.get("normalize", "meanstd"))
        np.save(cache_dir / split / phase / f"{item['utt']}.npy",
                feat.astype(np_dtype))
    return {"_idx": idx, "utt": item["utt"], "path": item["path"],
            "label": item["label"], "attack": item["attack"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None,
                    help="override data.root (e.g. synthetic data dir)")
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--phases", nargs="+", default=None,
                    help="subset of {clean,aec,ns,agc,vad}; default = all five. "
                         "Training needs every phase in train.stage_probs; the "
                         "eval table needs all five.")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N utterances per split (smoke)")
    ap.add_argument("--afe-seed", type=int, default=0,
                    help="base seed for the AFE distortion draw. Different values "
                         "give different members of the distortion family, which "
                         "is how a held-out cache is made")
    ap.add_argument("--cache-dir", default=None,
                    help="override data.cache_dir (e.g. cache/holdout)")
    ap.add_argument("--workers", type=int, default=min(16, mp.cpu_count()),
                    help="process-pool size; the job is CPU-bound")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="storage precision; float16 halves a 156 GB cache")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute utterances whose files already exist")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.root:
        cfg["data"]["root"] = args.root
    if args.cache_dir:
        cfg["data"]["cache_dir"] = args.cache_dir
    data_root = Path(cfg["data"]["root"])
    cache_dir = Path(cfg["data"]["cache_dir"])
    fcfg = cfg["feature"]
    ftype = fcfg.get("type", "logmel")
    phases = args.phases or (["clean"] + STAGES)

    split_map = {
        "train": ("ASVspoof2019_LA_train", "ASVspoof2019_LA_cm_protocols/LA.cm.train.trn.txt"),
        "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019_LA_cm_protocols/LA.cm.dev.trl.txt"),
        "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019_LA_cm_protocols/LA.cm.eval.trl.txt"),
    }

    for split in args.splits:
        flac_dir_name, proto_rel = split_map[split]
        flac_dir = data_root / flac_dir_name / "flac"
        proto_path = data_root / proto_rel
        if not proto_path.exists():
            print(f"[skip] protocol not found: {proto_path} "
                  f"(download ASVspoof 2019 LA and place it under {data_root})")
            continue
        entries = parse_protocol(proto_path, flac_dir)
        if args.limit:
            entries = entries[: args.limit]
        missing = []
        for i, e in enumerate(entries):
            if args.overwrite or not all(
                    (cache_dir / split / p / f"{e['utt']}.npy").exists()
                    for p in phases):
                missing.append((i, e))
        print(f"[preprocess] {split}: {len(entries)} utterances x {len(phases)} "
              f"phases -> {len(missing)} to compute, workers={args.workers}, "
              f"dtype={args.dtype}")

        meta = [None] * len(entries)
        meta_path = cache_dir / split / "meta.json"
        if meta_path.exists() and not args.overwrite:
            try:
                old = {m["utt"]: m for m in json.loads(meta_path.read_text())}
                for i, e in enumerate(entries):
                    if e["utt"] in old:
                        meta[i] = old[e["utt"]]
            except Exception:
                pass

        if missing:
            if args.workers <= 1:
                _init(cfg, split, phases, args.afe_seed, str(cache_dir), args.dtype)
                for it in tqdm(missing, desc=split):
                    m = _work(it)
                    meta[m.pop("_idx")] = m
            else:
                ctx = mp.get_context("spawn" if sys.platform == "win32" else "fork")
                with ctx.Pool(args.workers, initializer=_init,
                              initargs=(cfg, split, phases, args.afe_seed,
                                        str(cache_dir), args.dtype)) as pool:
                    for m in tqdm(pool.imap_unordered(_work, missing,
                                                      chunksize=8), total=len(missing),
                                  desc=split):
                        meta[m.pop("_idx")] = m

        meta = [m for m in meta if m is not None]
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        size_gb = sum(f.stat().st_size for p in phases
                      for f in (cache_dir / split / p).glob("*.npy")) / 1e9
        print(f"[done] {split}: meta at {meta_path} (n={len(meta)}), "
              f"{size_gb:.1f} GB on disk")


if __name__ == "__main__":
    main()
