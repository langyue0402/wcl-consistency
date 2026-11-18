#!/usr/bin/env python3
"""Pre-flight check for a long, unattended campaign run.

Every check here exists because failing it *after* the run starts costs machine
hours: a missing cache split wastes the setup, a 156 GB cache on a 120 GB disk
dies at hour 6, an un-raised file-handle limit kills the eval pass, a flat
consistency term wastes the whole run.  So the script is deliberately paranoid
and exits non-zero on the first blocker.

    python3 scripts/preflight.py --config configs/three_arm_phys.yaml
    python3 scripts/preflight.py --config configs/three_arm_phys.yaml --tiers "core seeds abl alpha"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, WARN, FAIL = "[OK  ]", "[WARN]", "[FAIL]"
problems: list[str] = []
warnings: list[str] = []


def say(tag: str, msg: str) -> None:
    print(f"{tag} {msg}")
    if tag == FAIL:
        problems.append(msg)
    elif tag == WARN:
        warnings.append(msg)


def human(gb: float) -> str:
    return f"{gb:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tiers", default="core seeds abl alpha",
                    help="campaign tiers that will actually be run")
    ap.add_argument("--manifest", default=None,
                    help="campaign manifest; used only for the hour estimate")
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="rental budget in hours; the estimate is compared to it")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="override the disk requirement (default: from the cache)")
    args = ap.parse_args()

    from wcl.utils import load_config
    cfg = load_config(args.config)
    dcfg, tcfg, wcfg = cfg["data"], cfg["train"], cfg["wcl"]
    cache = Path(dcfg["cache_dir"])

    print("=" * 60)
    print("WCL pre-flight")
    print("=" * 60)

    # 1. python / torch / device ------------------------------------------------
    print(f"\n--- 1. environment ---")
    print(f"       python {sys.version.split()[0]}")
    try:
        import torch
        print(f"       torch {torch.__version__}  cuda={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            say(FAIL, "no CUDA device: a full campaign cannot run on CPU")
        else:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"       gpu{i}: {props.name}  {props.total_memory / 2**30:.1f} GiB")
            if torch.cuda.get_device_properties(0).total_memory < 20 * 2**30:
                say(WARN, "GPU has < 20 GiB: batch 8 + product cost may OOM; "
                          "reduce train.batch_size or raise wcl.grid_stride")
        if "+cu" not in torch.__version__:
            say(WARN, "torch build looks CPU-only on a machine that has a GPU")
    except ImportError:
        say(FAIL, "torch is not installed")

    # 2. data presence ----------------------------------------------------------
    print(f"\n--- 2. dataset ---")
    root = Path(dcfg.get("root", ""))
    proto = Path(dcfg.get("protocol_dir", root / "ASVspoof2019_LA_cm_protocols"))
    print(f"       root     = {root}")
    print(f"       protocol = {proto}")
    if not root.exists():
        say(WARN, f"data root {root} not found (fine if the cache already exists); "
                  f"fetch it with: python scripts/fetch_data.py --config "
                  f"{args.config} --workers 16")
    else:
        for split in ("train", "dev", "eval"):
            key = {"train": "train_dir", "dev": "dev_dir", "eval": "eval_dir"}[split]
            p = Path(dcfg.get(key, root / f"ASVspoof2019_LA_{split}"))
            print(f"       {split:<5} = {p}  {'exists' if p.exists() else 'MISSING'}")

    # 3. cache completeness -----------------------------------------------------
    print(f"\n--- 3. cache ---")
    dtype_bytes = 2 if str(dcfg.get("cache_dtype", "float32")) == "float16" else 4
    missing = []
    total_gb = 0.0
    for split in ("train", "dev", "eval"):
        meta = cache / split / "meta.json"
        if meta.exists():
            n = len(os.listdir(cache / split)) - 1
            size = sum(f.stat().st_size for f in (cache / split).glob("*") if f.is_file())
            total_gb += size / 2**30
            print(f"       {split:<5}: meta.json OK, {n} files, {human(size / 2**30)}")
        else:
            missing.append(split)
            print(f"       {split:<5}: MISSING meta.json")
    if missing and cfg["data"].get("require_cache", True):
        say(FAIL, f"cache incomplete ({', '.join(missing)}) at {cache}: run "
                  f"scripts/preprocess.py --config {args.config} "
                  f"--workers {os.cpu_count()} --dtype float16 first")

    # 4. disk ------------------------------------------------------------------
    print(f"\n--- 4. disk ---")
    per_split = {"train": 32.5, "dev": 31.8, "eval": 91.2}
    need = args.min_free_gb
    if need is None:
        need = sum(per_split[s] for s in missing) * (dtype_bytes / 4) if missing else 12.0
        need = max(need, 12.0)
    free = shutil.disk_usage(cache.parent if cache.parent.exists() else Path(".")).free / 2**30
    print(f"       free = {human(free)}, required >= {human(need)}  (cache dtype="
          f"{dtype_bytes * 8}bit)")
    if free < need:
        say(FAIL, f"only {human(free)} free: the cache needs {human(need)}; "
                  f"use --dtype float16 (halves it) or a bigger volume")
    elif free < need * 1.5:
        say(WARN, "disk headroom < 1.5x: checkpoints + logs may fill the volume")

    # 5. file handles -----------------------------------------------------------
    print(f"\n--- 5. file handles ---")
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        nw = int(tcfg.get("num_workers", 4))
        need_fd = 1024 + nw * 256
        print(f"       RLIMIT_NOFILE soft={soft} hard={hard}, need ~{need_fd} "
              f"(num_workers={nw})")
        if soft < need_fd:
            say(FAIL, f"file-handle limit {soft} < {need_fd}: run `ulimit -n 65536` "
                      f"(or lower train.num_workers) before the eval pass, or it "
                      f"dies with 'Too many open files'")
    except Exception as exc:                      # pragma: no cover
        say(WARN, f"could not read RLIMIT_NOFILE: {exc}")

    # 6. objective configuration -------------------------------------------------
    print(f"\n--- 6. objective ---")
    print(f"       cost_mode={wcfg.get('cost_mode')} geo_mode={wcfg.get('geo_mode')} "
          f"grid_stride={wcfg.get('grid_stride')} eps={wcfg.get('eps_init')}"
          f"->{wcfg.get('eps_final')} lambda_couple={wcfg.get('lambda_couple')}")
    if str(wcfg.get("cost_mode")) != "product":
        say(WARN, "cost_mode != product: Theorem 2 bounds the product-space W_1, "
                  "so this run is an ablation, not the main result")
    if str(wcfg.get("geo_mode")) == "fixed" and float(wcfg.get("lambda_t", 1.0)) == 1.0:
        say(WARN, "geo_mode=fixed with lambda_t=1 leaves the geometric term at "
                  "~0.5% of a feature gap (a 100 ms trim): this is the run-1 failure "
                  "mode, only useful as the `lam1` ablation")
    if str(wcfg.get("lambda_couple")) not in ("none", "const_product"):
        say(FAIL, f"unknown lambda_couple={wcfg.get('lambda_couple')}")
    eps0, eps1 = float(wcfg["eps_init"]), float(wcfg["eps_final"])
    lam_ratio = eps0 / max(eps1, 1e-9)
    print(f"       with lambda fixed, the strength lambda*eps falls {lam_ratio:.1f}x "
          f"over the anneal; report lambda*eps, not lambda")

    # 7. throughput budget -------------------------------------------------------
    print(f"\n--- 7. time budget ---")
    epochs = int(tcfg.get("epochs", 100))
    eval_every = int(tcfg.get("eval_every", 1))
    dev_phases = len(tcfg.get("dev_phases") or ["clean", "aec", "ns", "agc", "vad"])
    # measured on the reference run (25,380 utt, batch 8 + accum 4)
    optim_s = 331.0 * (1.15 if str(wcfg.get("cost_mode")) == "product" else 1.0)
    dev_s = 660.0 * dev_phases / 5.0
    per_epoch = optim_s + dev_s / eval_every
    per_run_h = per_epoch * epochs / 3600.0
    runs = 0
    if args.manifest and Path(args.manifest).exists():
        tiers = set(args.tiers.split())
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip() in tiers:
                runs += 1
    else:
        runs = {"core": 4, "seeds": 8, "abl": 10, "alpha": 2}
        runs = sum(runs[t] for t in args.tiers.split() if t in runs)
    total_h = per_run_h * runs
    print(f"       ~{per_epoch / 60:.1f} min/epoch  ->  {per_run_h:.1f} h/run(x{epochs} ep)"
          f"  ->  {runs} runs = {total_h:.0f} h")
    if args.budget_hours and total_h > args.budget_hours:
        say(FAIL, f"estimated {total_h:.0f} h exceeds the {args.budget_hours:.0f} h "
                  f"budget: cut tiers, epochs, or dev_phases before renting")
    elif args.budget_hours:
        print(f"       within the {args.budget_hours:.0f} h rental budget "
              f"(margin {args.budget_hours - total_h:.0f} h)")

    print("\n" + "=" * 60)
    if problems:
        print(f"{len(problems)} blocker(s) -- do not start a long run yet:")
        for p in problems:
            print(f"  * {p}")
        return 1
    print(f"pre-flight clean ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
