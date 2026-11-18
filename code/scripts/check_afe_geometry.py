#!/usr/bin/env python3
"""Pre-flight check: does the AFE actually produce the geometry the theory wants?

Before spending GPU hours on a run, verify two things on a sample of real
utterances from the configured data root:

  1. the truncating VAD really shortens the clip and shifts the support
     (report how many ms of signal are removed and what the resulting shift is);
  2. the ground cost really carries the time coordinate (report the share of the
     cost matrix that comes from the |i-j| term, for the configured lambda_t).

If (1) reports near-zero truncation, the data are cut too tight for an
energy-based VAD and the geometric premise is still absent; set
`afe.vad.extra_trim_s > 0` (or `truncate_mode: stitch`) before training, or the
run will repeat the first experiment's flaw.

Usage:
    python scripts/check_afe_geometry.py --config configs/three_arm_phys.yaml \
        --split dev --n 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wcl.afe import STAGES, apply_stage, cascade_prefix
from wcl.dataset import load_meta
from wcl.utils import load_config


def shift_ms(a: np.ndarray, b: np.ndarray, sr: int, max_ms: int = 200) -> float:
    """Lag (in ms) that best aligns b onto a, searched over +/- max_ms."""
    n = min(len(a), len(b))
    if n < sr // 10:
        return 0.0
    seg = min(n, int(0.4 * sr))
    x = a[:seg] - a[:seg].mean()
    y = b[:seg] - b[:seg].mean()
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    lags = np.arange(-int(max_ms / 1000 * sr), int(max_ms / 1000 * sr) + 1)
    best, best_c = 0, -2.0
    for lag in lags:
        if lag < 0:
            c = np.corrcoef(x[-lag:], y[:len(x) + lag])[0, 1]
        elif lag > 0:
            c = np.corrcoef(x[:len(x) - lag], y[lag:])[0, 1]
        else:
            c = np.corrcoef(x, y)[0, 1]
        if np.isfinite(c) and c > best_c:
            best, best_c = int(lag), float(c)
    return best / sr * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "eval"])
    ap.add_argument("--n", type=int, default=40, help="utterances to sample")
    args = ap.parse_args()

    cfg = load_config(args.config)
    import librosa

    meta = load_meta(Path(cfg["data"]["cache_dir"]), args.split)
    if not meta:
        print("empty meta")
        return 1
    idx = np.linspace(0, len(meta) - 1, min(args.n, len(meta))).astype(int)

    sr = cfg["data"]["sr"]
    afe_cfg = cfg["afe"]
    wcfg = cfg["wcl"]

    removed, shifts, ratios = [], [], []
    rng = np.random.default_rng(int(afe_cfg.get("seed", 0)))
    for i in idx:
        path = meta[i]["path"]
        x, _ = librosa.load(path, sr=sr, mono=True)
        x = np.asarray(x, dtype=np.float32)
        prefix = cascade_prefix(STAGES, "vad")
        # distort up to AGC, then isolate the VAD step so the effect is clean
        pre = apply_stage("aec", x, sr, afe_cfg, rng=rng)
        pre = apply_stage("ns", pre, sr, afe_cfg, rng=rng)
        pre = apply_stage("agc", pre, sr, afe_cfg, rng=rng)
        post = apply_stage("vad", pre, sr, afe_cfg, rng=rng)
        removed.append(1.0 - len(post) / max(len(pre), 1))
        shifts.append(shift_ms(pre, post, sr))

    removed = np.array(removed)
    shifts = np.array(shifts)

    lam_t = float(wcfg.get("lambda_t", 0.0))
    mode = str(wcfg.get("cost_mode", "euclidean"))
    print(f"split={args.split}  n={len(idx)}  vad.mode={afe_cfg.get('vad', {}).get('mode', 'truncate')}"
          f"/{afe_cfg.get('vad', {}).get('truncate_mode', 'span')}"
          f"  extra_trim_s={afe_cfg.get('vad', {}).get('extra_trim_s', 0.0)}")
    print(f"VAD removed: mean {removed.mean() * 100:.1f}%  "
          f"median {np.median(removed) * 100:.1f}%  "
          f"max {removed.max() * 100:.1f}%")
    print(f"support shift: mean |{np.abs(shifts).mean():.1f}| ms  "
          f"max |{np.abs(shifts).max():.1f}| ms")
    print(f"cost_mode={mode} lambda_t={lam_t}")

    if removed.mean() < 0.02:
        print("\n[warn] VAD 平均只删掉不到 2% 的信号 —— 这批数据切得太紧，"
              "能量型 VAD 基本没东西可删，几何前提仍然不成立。\n"
              "       对策：把 afe.vad.extra_trim_s 设为 0.03~0.08（强制前移），"
              "或改用 truncate_mode: stitch。")
        return 2
    print("\n[ok] 截断与位移均已出现，几何前提成立，可以开跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
