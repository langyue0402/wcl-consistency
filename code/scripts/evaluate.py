#!/usr/bin/env python3
"""Stage-wise evaluation of a trained WCL model.

Reports EER/AUC per AFE phase prefix (clean / AEC / AEC+NS / AEC+NS+AGC /
full cascade), mirroring the reporting convention of the reference paper
(their Table 1).

Usage:
    python scripts/evaluate.py --config configs/smoke.yaml --ckpt outputs/smoke/default/best.pt
    python scripts/evaluate.py --config configs/train_full.yaml --ckpt ... --split eval
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from wcl.dataset import load_meta, WCLPhaseDataset
from wcl.model import build_model
from wcl.utils import (load_config, get_device, compute_eer, compute_auc,
                       best_threshold, Logger)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None,
                    help="override data.root (e.g. synthetic data dir)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "eval"])
    ap.add_argument("--cache-dir", default=None,
                    help="override data.cache_dir. Point this at an "
                         "--afe-seed != 0 cache to measure generalisation to "
                         "unseen members of the distortion family")
    ap.add_argument("--phases", nargs="+", default=None,
                    help="subset of {clean,aec,ns,agc,vad}; default = config")
    ap.add_argument("--limit", type=int, default=None,
                    help="only score the first N utterances (sanity check)")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the output files, e.g. _holdout, so that an "
                         "in-family and an unseen-distortion evaluation of the "
                         "same checkpoint do not overwrite each other")
    ap.add_argument("--save-scores", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.root:
        cfg["data"]["root"] = args.root
    if args.cache_dir:
        cfg["data"]["cache_dir"] = args.cache_dir
    ckpt = torch.load(args.ckpt, map_location="cpu")
    device = get_device()
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cache_dir = Path(cfg["data"]["cache_dir"])
    meta = load_meta(cache_dir, args.split)
    if args.limit:
        meta = meta[: args.limit]
    phases = args.phases or cfg["eval"]["phases"]
    logger = Logger(None)

    print(f"split={args.split}  n={len(meta)}  ckpt={args.ckpt}")
    print(f"{'phase':<8} {'EER(%)':>8} {'AUC':>8}   stage")
    print("-" * 42)
    results, all_scores = {}, {}
    with torch.no_grad():
        for phase in phases:
            ds = WCLPhaseDataset(meta, cache_dir, args.split, phase,
                                 feature_type=cfg["feature"].get("type", "logmel"))
            loader = DataLoader(ds, batch_size=cfg["eval"].get("batch_size", 32),
                                num_workers=cfg["eval"].get("num_workers", 2))
            scores, labels = [], []
            for x, y in loader:
                x = x.to(device)
                logits, _ = model(x)
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                scores.append(p)
                labels.append(y.numpy())
            scores = np.concatenate(scores)
            labels = np.concatenate(labels)
            eer = compute_eer(scores, labels)
            auc = compute_auc(scores, labels)
            results[phase] = {"eer": eer, "auc": auc, "n": len(labels)}
            all_scores[phase] = {"scores": scores.tolist(),
                                 "labels": labels.tolist()}
            print(f"{phase:<8} {eer * 100:8.2f} {auc:8.4f}")

    avg_eer = sum(v["eer"] for v in results.values()) / len(results)
    print(f"{'avg':<8} {avg_eer * 100:8.2f}")
    thr = best_threshold(all_scores["vad"]["scores"], all_scores["vad"]["labels"])
    print(f"[info] best threshold on '{args.split}/vad' (score>thr => bonafide): {thr:.4f}")

    if args.save_scores:
        out = Path(args.ckpt).parent / f"scores_{args.split}{args.out_suffix}.npz"
        np.savez_compressed(out, **{p: np.array(all_scores[p]["scores"])
                                    for p in phases},
                            labels=np.array(labels))
        print(f"[saved] {out}")

    # machine-readable summary, consumed by scripts/collect_arms.py
    summary = {"split": args.split, "n": len(meta),
               "ckpt": str(args.ckpt), "avg_eer": avg_eer,
               "cache_dir": str(cache_dir),
               "lambda_t": ckpt.get("lambda_t"), "lambda_f": ckpt.get("lambda_f"),
               "lambda_eff": ckpt.get("lambda_eff"),
               "phases": {p: {"eer": v["eer"], "auc": v["auc"]}
                          for p, v in results.items()}}
    jout = Path(args.ckpt).parent / f"eval_{args.split}{args.out_suffix}.json"
    with open(jout, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"[saved] {jout}")


if __name__ == "__main__":
    main()
