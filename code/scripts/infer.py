#!/usr/bin/env python3
"""Inference: score one file or a whole directory with a trained WCL model.

Optionally applies an AFE phase prefix first (e.g. --phase vad simulates the
full front-end cascade before scoring), which is useful to reproduce the
robustness setting of the paper.

Usage:
    python scripts/infer.py --config configs/smoke.yaml \
        --ckpt outputs/smoke/default/best.pt --input /path/to/audio.flac
    python scripts/infer.py --config configs/smoke.yaml \
        --ckpt outputs/smoke/default/best.pt --input /path/dir/ \
        --phase vad --threshold 0.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from wcl.afe import apply_cascade, cascade_prefix, STAGES
from wcl.features import (log_mel, pad_or_crop, normalize, pad_or_crop_wave,
                          normalize_wave)
from wcl.model import build_model
from wcl.utils import load_config, get_device


def score_file(model, path: str, cfg, device, phase: str,
               afe_seed: int = 0) -> float:
    sr = cfg["data"]["sr"]
    fcfg = cfg["feature"]
    ftype = fcfg.get("type", "logmel")
    import librosa
    x, _ = librosa.load(path, sr=sr, mono=True)
    x = x.astype(np.float32)
    if phase != "clean":
        prefix = cascade_prefix(STAGES, phase)
        # stable per-file seed: with a fixed seed every file would receive the
        # same draw from the distortion family
        import zlib
        seed = (int(afe_seed) * 1_000_003 + zlib.crc32(str(path).encode())) % (2 ** 31 - 1)
        x = apply_cascade(x, sr, cfg["afe"], seed=seed, stages=prefix)
    if ftype == "raw":
        n_samples = int(round(sr * float(fcfg.get("duration", 4.0))))
        feat = normalize_wave(pad_or_crop_wave(x, n_samples))
        xin = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(device)
    else:
        feat = log_mel(x, sr, fcfg["n_fft"], fcfg["hop"], fcfg["n_mels"])
        feat = pad_or_crop(feat, fcfg["max_frames"])
        feat = normalize(feat, fcfg.get("normalize", "meanstd"))
        xin = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(xin)
        p = torch.softmax(logits, dim=1)[:, 1].item()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None,
                    help="override data.root (e.g. synthetic data dir)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="audio file or directory")
    ap.add_argument("--phase", default="clean",
                    choices=["clean"] + STAGES,
                    help="AFE prefix to apply before scoring (default: clean)")
    ap.add_argument("--afe-seed", type=int, default=0,
                    help="base seed of the AFE distortion draw (see preprocess.py)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="score > threshold => bonafide (use the value printed "
                         "by evaluate.py for the best dev threshold)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.root:
        cfg["data"]["root"] = args.root
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    inp = Path(args.input)
    files = sorted(inp.rglob("*.flac")) if inp.is_dir() else [inp]
    if inp.is_dir():
        files += sorted(inp.rglob("*.wav"))
        files = sorted(set(files))
    if not files:
        print("no audio files found")
        return

    print(f"{'file':<40} {'score':>7}  verdict")
    print("-" * 60)
    for f in files:
        p = score_file(model, str(f), cfg, device, args.phase, args.afe_seed)
        verdict = "bonafide" if p > args.threshold else "spoof"
        print(f"{str(f)[-40:]:<40} {p:7.4f}  {verdict}")


if __name__ == "__main__":
    main()
