#!/usr/bin/env python3
"""Generate a SMALL synthetic stand-in dataset with the exact ASVspoof 2019 LA
directory/protocol layout, so the full pipeline can be smoke-tested before the
real database is downloaded.

Bonafide = formant-like harmonic speech; spoof = the same speech with TTS/VC
artifacts (phase modulation, spectral flattening, discontinuities).

Usage:
    python scripts/make_synthetic_data.py --root data/synth_asvspoof \
        --train 40 --dev 40 --eval 40 --dur 2.0
Then run the pipeline with --root data/synth_asvspoof (see README).
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def synth_audio(seed: int, dur: float, sr: int, spoof: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    f0 = rng.uniform(110, 220)
    x = (0.4 * np.sin(2 * np.pi * f0 * t)
         + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
         + 0.15 * np.sin(2 * np.pi * 3 * f0 * t))
    # formant-ish shaping with a slow amplitude envelope (syllable-like)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 4) * t)
    x = x * env + 0.01 * rng.standard_normal(len(t))
    if spoof:
        # Stronger TTS/VC artifacts so the log-mel features are clearly
        # separable in the smoke test: broadband noise, phase jitter,
        # formant flattening and dropouts.
        j = np.cumsum(rng.uniform(-0.15, 0.15, len(t)))
        x = x * np.cos(j) * 0.6
        x = np.convolve(x, np.ones(8) / 8, mode="same")
        x = x + 0.25 * rng.standard_normal(len(t))
        n = len(x)
        for _ in range(4):
            pos = rng.integers(500, n - 500)
            x[pos: pos + 300] *= 0.3
    x = x / (np.max(np.abs(x)) + 1e-8) * 0.8
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synth_asvspoof")
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--dev", type=int, default=40)
    ap.add_argument("--eval", type=int, default=40)
    ap.add_argument("--dur", type=float, default=2.0)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    root = Path(args.root)
    for split, n in [("train", args.train), ("dev", args.dev), ("eval", args.eval)]:
        flac_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
        proto_dir = root / "ASVspoof2019_LA_cm_protocols"
        flac_dir.mkdir(parents=True, exist_ok=True)
        proto_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        n_bon, n_spo = 0, 0
        for i in range(n):
            spoof = i % 2 == 1
            tag = "SP" if spoof else "BN"
            utt = f"LA_{'T' if split == 'train' else 'D' if split == 'dev' else 'E'}_{tag}_{i:04d}"
            x = synth_audio(seed=1000 + i, dur=args.dur, sr=args.sr, spoof=spoof)
            sf.write(flac_dir / f"{utt}.flac", x, args.sr, format="FLAC")
            label = "spoof" if spoof else "bonafide"
            attack = ("A01" if spoof else "-")
            if split == "train":
                lines.append(f"{utt.split('_')[0]}_{utt} {utt} - - {label}")
            else:
                lines.append(f"{utt.split('_')[0]}_{utt} {utt} - {attack} {label}")
            if spoof:
                n_spo += 1
            else:
                n_bon += 1
        name = {"train": "train.trn", "dev": "dev.trl", "eval": "eval.trl"}[split]
        (proto_dir / f"LA.cm.{name}.txt").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")
        print(f"[synth] {split}: {n} utterances ({n_bon} bonafide, {n_spo} spoof) -> {flac_dir}")


if __name__ == "__main__":
    main()
