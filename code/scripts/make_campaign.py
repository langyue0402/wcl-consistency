#!/usr/bin/env python3
"""Generate the experiment campaign: configs + a manifest the runner consumes.

Why a generator instead of hand-written configs: every run has to differ from
the base config in exactly one place, and a copy-pasted YAML drifts.  Here the
base file stays the single source of truth and each run is a named override.

Tiers (run in this order):

    core     the three arms the paper's Table 1 needs, one config per seed
    seeds    extra seeds for the same three arms (error bars)
    abl      WCL with one component switched off; this is what tests the theory
             (marginal vs product, geometry on/off, calibrated vs fixed lam,
             the frozen-frontend control)
    alpha    the consistency-weight sweep promised in Section 5.4
    afe      AFE-level ablations; each needs its OWN feature cache, so they are
             flagged and skipped by the runner unless the cache exists

Usage:
    python scripts/make_campaign.py --base configs/three_arm_phys.yaml --seeds 0 1 2
    python scripts/make_campaign.py --base configs/three_arm_phys.yaml --campaign camp_v4
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

ARMS = ["ce", "tfcl", "wcl"]

# name, arm, overrides, needs_own_cache
ABLATIONS = [
    # --- directly tests the theoretical claim -------------------------------
    ("wcl_marginal", "wcl", {"wcl.cost_mode": "marginal"}, False,
     "two 1-D marginal transports (the Section 5.1 form) instead of one "
     "product-space transport: Theorem 2 does not apply to this loss"),
    ("wcl_nogeo", "wcl", {"wcl.beta_t": 0.0, "wcl.beta_f": 0.0}, False,
     "geometry switched off entirely: cost = ||v - v'||, i.e. run 1's cost but "
     "now with a live gradient"),
    ("wcl_lam1", "wcl", {"wcl.geo_mode": "fixed", "wcl.lambda_t": 1.0,
                         "wcl.lambda_f": 0.5}, False,
     "the v3 setting: lam_t = 1 makes a 100 ms trim cost 0.5% of a feature "
     "gap, so the plan is decided by appearance alone"),
    ("wcl_frontend", "wcl", {"model.measure_from": "frontend"}, False,
     "the run-1 defect kept as a control: measures on the frozen XLS-R output "
     "(zero gradient). Its numbers must reproduce dual-view CE."),
    ("wcl_tonly", "wcl", {"wcl.beta_f": 0.0}, False, "temporal geometry only"),
    ("wcl_fonly", "wcl", {"wcl.beta_t": 0.0}, False, "band geometry only"),
    ("wcl_constprod", "wcl", {"wcl.lambda_couple": "const_product"}, False,
     "hold lambda*eps fixed while eps anneals (a pure temperature anneal). The "
     "draft's schedule keeps lambda fixed, which weakens the consistency term "
     "by eps_init/eps_final = 5x relative to CE over the run"),
    ("wcl_stride1", "wcl", {"wcl.grid_stride": 1}, False,
     "full time resolution (1600 cells); checks the discretisation of "
     "grid_stride=2 does not drive the result"),
    ("tfcl_ckabatch", "tfcl", {"wcl.tfcl_cka_dim": "batch"}, False,
     "the reference CKA convention (over the batch dimension): reported as a "
     "weak arm because the estimator saturates at ~0.89 when B = 8"),
    # --- AFE-level, each needs a fresh cache -------------------------------
    ("wcl_vadzero", "wcl", {"afe.vad.mode": "zero"}, True,
     "in-place VAD (length preserved) instead of truncation: does the support "
     "shift the theory models actually matter?"),
    ("wcl_nojitter", "wcl", {"afe.randomize": False}, True,
     "one fixed distortion instead of the family: tests whether the jitter is "
     "what makes the invariance claim meaningful"),
]

ALPHA_SWEEP = [0.1, 1.0]


def set_dotted(cfg: dict, path: str, value) -> None:
    keys = path.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/three_arm_phys.yaml")
    ap.add_argument("--campaign", default="camp_v4")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out-root", default=None,
                    help="outputs/<this>; default = outputs/<campaign>")
    ap.add_argument("--tiers", nargs="+", default=["core", "seeds", "abl",
                                                   "alpha", "afe"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="override train.epochs for every run (cost control)")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base, encoding="utf-8"))
    camp = Path("campaign") / args.campaign
    camp.mkdir(parents=True, exist_ok=True)
    out_root = args.out_root or f"outputs/{args.campaign}"

    rows: list[tuple[str, str, str, str, str]] = []   # name, tier, arm, cfg, cache

    def emit(name: str, tier: str, arm: str, overrides: dict,
             cache: str = "") -> None:
        cfg = copy.deepcopy(base)
        cfg["out_dir"] = out_root
        for k, v in overrides.items():
            set_dotted(cfg, k, v)
        if args.epochs:
            cfg["train"]["epochs"] = args.epochs
        if cache:
            cfg["data"]["cache_dir"] = cache
        path = camp / f"{name}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        rows.append((name, tier, arm, str(path), cache or cfg["data"]["cache_dir"]))
        print(f"  {tier:5s} {name:16s} <- {args.base}  (arm={arm})")

    print(f"[campaign] base={args.base} out_root={out_root} seeds={args.seeds}")
    if "core" in args.tiers:
        for arm in ARMS:
            for s in args.seeds:
                emit(f"{arm}__s{s}", "core", arm, {"train.seed": s})
        # the clean-only CE arm is part of the core comparison, not an extra
        # ablation: without it the augmentation effect cannot be separated from
        # the constraint effect, and the four-arm table would be three arms.
        for s in args.seeds:
            emit(f"ce_cleanonly__s{s}", "core", "ce",
                 {"train.stage_probs": [1.0, 0.0, 0.0, 0.0, 0.0],
                  "train.seed": s})
    if "seeds" in args.tiers and len(args.seeds) > 1:
        # identical to core for seed 0; the runner skips what already exists
        for arm in ARMS:
            for s in args.seeds[1:]:
                emit(f"{arm}__s{s}", "seeds", arm, {"train.seed": s})
        for s in args.seeds[1:]:
            emit(f"ce_cleanonly__s{s}", "seeds", "ce",
                 {"train.stage_probs": [1.0, 0.0, 0.0, 0.0, 0.0],
                  "train.seed": s})
    if "abl" in args.tiers:
        for name, arm, ov, own_cache, _desc in ABLATIONS:
            if own_cache:
                continue
            emit(f"{name}__s{args.seeds[0]}", "abl", arm,
                 {**ov, "train.seed": args.seeds[0]})
    if "alpha" in args.tiers:
        # alpha = 0.3 is already covered by the core wcl arm; only the two
        # off-centre points are separate runs.
        for a in ALPHA_SWEEP:
            emit(f"wcl_a{a}__s{args.seeds[0]}", "alpha", "wcl",
                 {"wcl.alpha": a, "train.seed": args.seeds[0]})
    if "afe" in args.tiers:
        for name, arm, ov, own_cache, _desc in ABLATIONS:
            if not own_cache:
                continue
            cache = f"cache/{args.campaign}_{name}"
            emit(f"{name}__s{args.seeds[0]}", "afe", arm,
                 {**ov, "train.seed": args.seeds[0]}, cache=cache)

    manifest = camp / "manifest.tsv"
    # de-duplicate: a name may be emitted by both "core" and "seeds"
    seen, uniq = set(), []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        uniq.append(r)
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        f.write("name\ttier\tarm\tconfig\tcache\n")
        for r in uniq:
            f.write("\t".join(r) + "\n")
    print(f"\n[campaign] {len(uniq)} runs -> {manifest}")

    n_core = sum(1 for r in uniq if r[1] == "core")
    print(f"[campaign] core {n_core} | abl "
          f"{sum(1 for r in uniq if r[1] == 'abl')} | alpha "
          f"{sum(1 for r in uniq if r[1] == 'alpha')} | afe "
          f"{sum(1 for r in uniq if r[1] == 'afe')}")
    print("[campaign] tier 'afe' needs its own cache; the runner skips those "
          "until preprocess has been run for them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
