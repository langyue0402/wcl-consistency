#!/usr/bin/env python3
"""Collect the three-arm comparison into one table.

Reads, for every run directory under ``--root``:

    <root>/<tag>/history.json        # dev curve, best average EER
    <root>/<tag>/eval_eval.json      # five-stage eval summary (optional)

and prints a Markdown table of the numbers the paper's Table 1 needs, plus the
VAD marginal factor (cascade with VAD divided by the cascade without it), which
is the quantity the WCL claim is judged on.

Usage:
    python scripts/collect_arms.py --root outputs/phys_v1
    python scripts/collect_arms.py --root outputs/phys_v1 --split dev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PHASES = ["clean", "aec", "ns", "agc", "vad"]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def best_dev(hist) -> tuple[float, int]:
    if not hist:
        return float("nan"), -1
    records = hist if isinstance(hist, list) else hist.get("epochs", [])
    best, ep = float("inf"), -1
    for rec in records:
        eer = rec.get("dev_avg_eer", rec.get("avg_eer"))
        if eer is None:
            continue
        if eer < best:
            best, ep = float(eer), int(rec.get("epoch", -1))
    return best, ep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/phys_v1")
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        summary = load_json(run / f"eval_{args.split}.json")
        if summary is None:
            continue
        hist = load_json(run / "history.json")
        dev_best, dev_ep = best_dev(hist)
        ph = summary["phases"]
        rows.append({
            "arm": run.name,
            "dev_best": dev_best,
            "dev_ep": dev_ep,
            **{p: ph.get(p, {}).get("eer", float("nan")) for p in PHASES},
            "avg": summary.get("avg_eer", float("nan")),
        })

    if not rows:
        print(f"no eval_{args.split}.json found under {root}")
        return 1

    hdr = ("| arm | best dev avgEER (%) | clean | AEC | AEC+NS | AEC+NS+AGC "
           "| full cascade | avg | VAD marginal |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    print(hdr)
    print(sep)
    for r in rows:
        marginal = (r["vad"] / r["agc"]) if r["agc"] else float("nan")
        print("| {} | {:.2f} (ep{}) | {:.2f} | {:.2f} | {:.2f} | {:.2f} | "
              "{:.2f} | {:.2f} | {:.2f}x |".format(
                  r["arm"], r["dev_best"] * 100, r["dev_ep"],
                  r["clean"] * 100, r["aec"] * 100, r["ns"] * 100,
                  r["agc"] * 100, r["vad"] * 100, r["avg"] * 100, marginal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
