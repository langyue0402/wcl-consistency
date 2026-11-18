#!/usr/bin/env python3
"""Collect a whole campaign into the tables the paper needs.

For every run directory ``<root>/<name>`` it reads

    history.json        dev curve (best epoch / best average EER)
    eval_eval.json      five-stage results on the in-family distortion draw
    eval_eval_holdout.json   same checkpoint on an *unseen* draw (optional)

and prints three Markdown tables:

  1. per-run           every number, one row per training run
  2. grouped           mean over seeds, with the spread, one row per method
  3. main comparison   the five-stage EER table (+ the VAD marginal, which is
                       the quantity the WCL claim is judged on)

Usage:
    python scripts/collect_campaign.py --root outputs/camp_v4
    python scripts/collect_campaign.py --root outputs/camp_v4 --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

PHASES = ["clean", "aec", "ns", "agc", "vad"]


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def best_dev(hist) -> tuple[float, int]:
    if not hist:
        return float("nan"), -1
    best, ep = float("inf"), -1
    for rec in hist:
        eer = rec.get("avg_eer")
        if eer is None:
            continue
        if eer < best:
            best, ep = float(eer), int(rec.get("epoch", -1))
    return best, ep


def base_name(name: str) -> str:
    return re.sub(r"__s\d+$", "", name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/camp_v4")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[collect] {root} does not exist yet")
        return 1

    rows = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        summ = load(run / "eval_eval.json")
        dev = load(run / "dev.json")
        hist = load(run / "history.json")
        dev_best, dev_ep = best_dev(hist)
        row = {"run": run.name, "group": base_name(run.name),
               "dev_best": dev_best, "dev_ep": dev_ep,
               "done": bool(summ)}
        if summ:
            ph = summ.get("phases", {})
            for p in PHASES:
                row[p] = ph.get(p, {}).get("eer", float("nan"))
            row["avg"] = summ.get("avg_eer", float("nan"))
            row["lambda_eff"] = summ.get("lambda_eff")
            row["lambda_t"] = summ.get("lambda_t")
        hold = load(run / "eval_eval_holdout.json")
        if hold:
            row["hold_avg"] = hold.get("avg_eer", float("nan"))
            row["hold_vad"] = hold.get("phases", {}).get("vad", {}).get("eer")
        rows.append(row)
        _ = dev

    if not rows:
        print(f"[collect] no run directories under {root}")
        return 1

    def pct(v):
        return "—" if v is None or v != v else f"{v * 100:.2f}"

    print("\n### 1. per-run\n")
    hdr = ("| run | best dev avgEER | ep | clean | AEC | AEC+NS | +AGC | cascade "
           "| avg | VAD× | holdout avg | λ_eff |")
    print(hdr)
    print("|" + "---|" * 12)
    for r in rows:
        marg = (r["vad"] / r["agc"]) if r.get("agc") else None
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["run"], pct(r["dev_best"]), r["dev_ep"],
            pct(r.get("clean")), pct(r.get("aec")), pct(r.get("ns")),
            pct(r.get("agc")), pct(r.get("vad")), pct(r.get("avg")),
            "—" if marg is None else f"{marg:.2f}",
            pct(r.get("hold_avg")),
            "—" if r.get("lambda_eff") is None else f"{r['lambda_eff']:.3f}"))

    print("\n### 2. grouped over seeds\n")
    groups = defaultdict(list)
    for r in rows:
        if r["done"]:
            groups[r["group"]].append(r)
    print("| method | n | cascade EER (mean) | spread | avg EER | clean | holdout |")
    print("|---|---|---|---|---|---|---|")
    for g, rs in sorted(groups.items()):
        def ok(v):
            return v is not None and v == v

        vad = [r["vad"] for r in rs if ok(r.get("vad"))]
        avg = [r["avg"] for r in rs if ok(r.get("avg"))]
        cl = [r["clean"] for r in rs if ok(r.get("clean"))]
        hd = [r["hold_avg"] for r in rs if ok(r.get("hold_avg"))]
        if not vad:
            continue
        spread = f"{pct(min(vad))}–{pct(max(vad))}" if len(vad) > 1 else "—"
        print("| {} | {} | {} | {} | {} | {} | {} |".format(
            g, len(vad), pct(sum(vad) / len(vad)), spread,
            pct(sum(avg) / len(avg)) if avg else "—",
            pct(sum(cl) / len(cl)) if cl else "—",
            pct(sum(hd) / len(hd)) if hd else "—"))

    print("\n### 3. main five-stage table (single seed shown; use table 2 for the "
          "seed-averaged version)\n")
    print("| method | clean | AEC | AEC+NS | AEC+NS+AGC | full cascade | avg |")
    print("|---|---|---|---|---|---|---|")
    for g, rs in sorted(groups.items()):
        r = rs[0]
        print("| {} | {} | {} | {} | {} | {} | {} |".format(
            g, pct(r.get("clean")), pct(r.get("aec")), pct(r.get("ns")),
            pct(r.get("agc")), pct(r.get("vad")), pct(r.get("avg"))))

    if args.csv:
        keys = ["run", "group", "dev_best", "dev_ep", *PHASES, "avg",
                "hold_avg", "lambda_eff", "lambda_t"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[csv] {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
