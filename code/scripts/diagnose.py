#!/usr/bin/env python3
"""Pre-flight diagnostics: run this BEFORE renting GPUs / training.

Checks, in order of severity:

  1. [BLOCKING] Does the consistency term actually produce gradient on any
     trainable parameter?  With `model.frontend.freeze: true` the WCL(xlsr)
     features come from a frozen encoder and the classifier is not on that
     path, so `lam * W` can be a pure constant added to the loss.
     This script compares optimizer-facing gradients with lam=0 and lam>0.
  2. [BLOCKING] Is the `agc` evaluation phase distinguishable from the `ns`
     phase?  `agc` only rescales the whole waveform, and the front end
     standardises every utterance (zero mean / unit variance), which removes
     exactly that rescaling.
  3. [BLOCKING] Is the objective the quantity the theorem bounds?  A loss made
     of the two marginal transports W_t + W_f is *exactly* at its floor when a
     displacement preserves both profiles (diagonal blob -> anti-diagonal
     blob), while the product-space W_1 is not.  Reported as a number.
  4. [HIGH] Is the entropic regularisation still "soft"?  If
     mean(ground cost) / eps is in the hundreds, Sinkhorn degenerates into a
     hard assignment and eps-annealing stops meaning what the paper says.
  5. [BLOCKING] Is the jitter actually consumed?  The theory is about a
     *family* of distortions, so two draws of the same utterance with different
     seeds must differ; with `afe.randomize: false` (or an unread parameter)
     training sees a single fixed point and the invariance claim is empty.
  6. [BLOCKING] Does the truncating VAD remove a measurable fraction of the
     signal?  On material cut too tight for an energy detector it removes
     almost nothing, and the displacement the theory models never occurs.
     Reported as removed fraction, head-trim shift in ms, and a 2% floor.

Checks 7 and 8 are reports rather than gates: the geometric coefficient
calibration (what the paper's Section 5.1 derives) and the cache / GPU-hour
budget (preflight.py already gates disk and file handles).

Usage:
    python scripts/diagnose.py --config configs/three_arm_phys.yaml
    python scripts/diagnose.py --config configs/legacy_run1.yaml --skip-model
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from wcl.losses import (wcl_loss, product_cost, calibrate_geometry, ground_cost,
                        wasserstein_1_entropic)
from wcl.model import build_model
from wcl.utils import load_config, afe_trim_cells

TINY_XLSR = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                 intermediate_size=128, conv_dim=(32, 32), conv_stride=(10, 5),
                 conv_kernel=(10, 3), num_conv_pos_embeddings=16,
                 num_conv_pos_embedding_groups=2, feat_extract_norm="layer")

FAILED: list[str] = []


def say(status: str, msg: str) -> None:
    print(f"[{status:4s}] {msg}")
    if status == "FAIL":
        FAILED.append(msg)


def _dummy_grids(cfg: dict, device: str, batch: int = 2):
    """Feature grids with the shapes the real model produces."""
    n_sub = int(cfg["model"].get("n_subbands", 8))
    hidden = int(cfg["model"].get("backend", {}).get("hidden", 128))
    if cfg["model"].get("measure_from", "backend") == "backend":
        dim = hidden
    else:
        dim = 1024
    if cfg["feature"].get("type") == "raw":
        dur = float(cfg["feature"].get("duration", 4.0))
        t = int(round(dur * 16000)) // 320
    else:
        t = int(cfg["feature"].get("max_frames", 640)) // 16
    c = max(dim // n_sub, 1)
    a = torch.randn(batch, c, t, n_sub, device=device)
    b = a + torch.randn_like(a)
    return a, b


# ---------------------------------------------------------------- check 1
def check_consistency_gradient(cfg: dict, device: str) -> None:
    """Is the WCL term actually training anything?"""
    print("\n--- 1. consistency term -> gradient ---")
    mcfg = dict(cfg["model"])
    wcfg = cfg.get("wcl", {})
    eps = float(wcfg.get("eps_final", 0.1))
    n_iter = int(wcfg.get("sinkhorn_iters", 20))
    stride = int(wcfg.get("grid_stride", 1))

    def build(freeze: bool):
        c = {"model": dict(mcfg)}
        if c["model"].get("arch") != "xlsr_aasist":
            return build_model(c).to(device)
        c["model"]["frontend"] = {"pretrained": False, "freeze": freeze,
                                  "config_overrides": TINY_XLSR}
        c["model"]["backend"] = dict(c["model"].get("backend") or {})
        c["model"]["backend"]["hidden"] = 32
        c["model"]["backend"]["n_layers"] = 2
        c["model"]["n_subbands"] = 4
        return build_model(c).to(device)

    for freeze in (True, False):
        if mcfg.get("arch") != "xlsr_aasist" and freeze is False:
            continue
        torch.manual_seed(0)
        model = build(freeze)
        model.eval()                                # no dropout => deterministic
        is_raw = cfg["feature"].get("type") == "raw"
        if is_raw:
            xc = torch.randn(2, 6400, device=device)
            xd = torch.randn(2, 6400, device=device)
        else:
            xc = torch.randn(2, 1, 320, 80, device=device)
            xd = torch.randn(2, 1, 320, 80, device=device)
        labels = torch.tensor([1, 0], device=device)

        # use the geometry the real run would use, so the test is not weakened
        # by an arbitrary lam_t
        with torch.no_grad():
            _, fc0 = model(xc)
            _, fd0 = model(xd)
        if str(wcfg.get("geo_mode", "calibrate")).lower() == "calibrate":
            lam_t, lam_f, _ = calibrate_geometry(
                fc0.float(), fd0.float(),
                beta_t=float(wcfg.get("beta_t", 1.0)),
                beta_f=float(wcfg.get("beta_f", 1.0)), stride=stride,
                shift_cells=afe_trim_cells(cfg.get("afe", {}), stride),
                band_cells=float(wcfg.get("band_cells", 1.0)))
        else:
            lam_t = float(wcfg.get("lambda_t", 0.0))
            lam_f = float(wcfg.get("lambda_f", 0.0))

        def grads(lam: float):
            lc, fc = model(xc)
            ld, fd = model(xd)
            loss = wcl_loss(lc, ld, labels, fc, fd, lam=lam, eps=eps,
                            n_iter=n_iter,
                            use_temporal=wcfg.get("use_temporal", True),
                            use_frequency=wcfg.get("use_frequency", True),
                            cost_mode=wcfg.get("cost_mode", "product"),
                            lambda_t=lam_t, lambda_f=lam_f,
                            cost_normalize=wcfg.get("cost_normalize", "mean"),
                            grid_stride=stride)
            model.zero_grad(set_to_none=True)
            loss.backward()
            return torch.cat([p.grad.flatten()
                              for p in model.parameters() if p.grad is not None])

        g0, g1 = grads(0.0), grads(1.0)
        d = (g0 - g1).abs().max().item()
        tag = "freeze=True" if freeze else "freeze=False"
        n_train = sum(p.requires_grad for p in model.parameters())
        if d > 1e-6 * max(g0.abs().max().item(), 1e-12):
            say("OK", f"{tag}: consistency term changes gradients "
                      f"(max|dGrad|={d:.3e}, trainable tensors={n_train})")
        else:
            say("FAIL", f"{tag}: lam=0 and lam=1 give IDENTICAL gradients "
                        f"(max|dGrad|={d:.1e}) -> the WCL term is a constant "
                        f"and training is pure dual-view CE")
    if device == "cpu":
        print("       (tiny randomly-initialised XLS-R used on CPU; "
              "the freeze=True conclusion is architecture-level)")


# ---------------------------------------------------------------- check 2
def check_agc_phase(cfg: dict) -> None:
    """Does the AGC stage survive the per-utterance standardisation?"""
    print("\n--- 2. is the AGC phase distinguishable from AEC+NS? ---")
    from wcl.afe import apply_cascade, cascade_prefix, STAGES
    from wcl.features import (normalize_wave, pad_or_crop_wave, log_mel,
                              pad_or_crop, normalize)

    sr = int(cfg["data"]["sr"])
    afe = cfg["afe"]
    rng = np.random.default_rng(0)
    t = np.arange(int(2 * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * 180 * t)
         + 0.04 * rng.standard_normal(t.size)).astype(np.float32)

    y_ns = apply_cascade(x, sr, afe, seed=afe.get("seed", 0),
                         stages=cascade_prefix(STAGES, "ns"))
    y_agc = apply_cascade(x, sr, afe, seed=afe.get("seed", 0),
                          stages=cascade_prefix(STAGES, "agc"))

    ftype = cfg["feature"].get("type", "logmel")
    if ftype == "raw":
        n = int(round(sr * float(cfg["feature"].get("duration", 4.0))))
        a = normalize_wave(pad_or_crop_wave(y_ns, n))
        b = normalize_wave(pad_or_crop_wave(y_agc, n))
    else:
        f = cfg["feature"]
        a = normalize(pad_or_crop(log_mel(y_ns, sr, f["n_fft"], f["hop"], f["n_mels"]),
                                  f["max_frames"]), f.get("normalize", "meanstd"))
        b = normalize(pad_or_crop(log_mel(y_agc, sr, f["n_fft"], f["hop"], f["n_mels"]),
                                  f["max_frames"]), f.get("normalize", "meanstd"))
    rel = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12))
    if rel < 1e-3:
        say("FAIL", f"phases 'ns' (AEC+NS) and 'agc' (AEC+NS+AGC) give the same "
                    f"model input (rel.diff={rel:.2e}): stage agc only rescales "
                    f"globally and the front end normalises it away")
    else:
        say("OK", f"AGC is observable (rel.diff={rel:.2e})")


# ---------------------------------------------------------------- check 3
def check_objective_matches_theorem(cfg: dict, device: str) -> None:
    """Does the loss see a displacement that preserves both profiles?"""
    print("\n--- 3. does the objective control the product-space W_1? ---")
    wcfg = cfg["wcl"]
    eps = float(wcfg.get("eps_init", 0.5))
    n_iter = int(wcfg.get("sinkhorn_iters", 20))
    mode = str(wcfg.get("cost_mode", "product")).lower()

    t, f, c = 16, 8, 4
    a = torch.zeros(1, c, t, f, device=device)
    b = torch.zeros(1, c, t, f, device=device)
    for i in range(min(t, f)):
        a[0, :, i, i] = 1.0
        b[0, :, i, f - 1 - i] = 1.0

    def w_marg(x, y):
        wt = wasserstein_1_entropic(
            ground_cost(x.mean(3).transpose(1, 2), y.mean(3).transpose(1, 2),
                        "physical", 1.0, "mean"), eps, n_iter)
        wf = wasserstein_1_entropic(
            ground_cost(x.mean(2).transpose(1, 2), y.mean(2).transpose(1, 2),
                        "physical", 0.5, "mean"), eps, n_iter)
        return float((wt + wf).item())

    def w_prod(x, y):
        cost = product_cost(x, y, 1.0, 0.5, "mean", 1)
        return float(wasserstein_1_entropic(cost, eps, n_iter).item())

    excess_m = w_marg(a, b) - w_marg(a, a)
    excess_p = w_prod(a, b) - w_prod(a, a)
    print(f"       marginal objective: excess = {excess_m:.2e}")
    print(f"       product-space W_1 : excess = {excess_p:.2e}")
    if mode in ("product", "prod", "joint"):
        if excess_p <= 1e-6:
            say("FAIL", "the product-space transport does not react to the "
                        "displacement either; check lambda_t/lambda_f = 0")
        else:
            say("OK", f"cost_mode=product sees the displacement "
                      f"(marginal would have reported {excess_m:.1e})")
    else:
        say("FAIL", f"cost_mode={mode} is a sum of two 1-D marginal transports: "
                    f"it sits exactly at its floor ({excess_m:.1e}) while the "
                    f"quantity Theorem 2 bounds moves ({excess_p:.1e}). "
                    f"Use cost_mode: product, or state in the paper that the "
                    f"bound does not apply to the objective.")


# ---------------------------------------------------------------- check 4
def check_eps_scale(cfg: dict, device: str) -> None:
    """Is eps small compared with the ground cost?"""
    print("\n--- 4. eps vs ground-cost scale ---")
    wcfg = cfg["wcl"]
    mode = str(wcfg.get("cost_mode", "product")).lower()
    norm = wcfg.get("cost_normalize", "mean")
    stride = int(wcfg.get("grid_stride", 1))
    a, b = _dummy_grids(cfg, device, batch=1)
    if str(wcfg.get("geo_mode", "calibrate")).lower() == "calibrate":
        lam_t, lam_f, info = calibrate_geometry(
            a, b, float(wcfg.get("beta_t", 1.0)),
            float(wcfg.get("beta_f", 1.0)), stride,
            shift_cells=afe_trim_cells(cfg.get("afe", {}), stride),
            band_cells=float(wcfg.get("band_cells", 1.0)))
    else:
        lam_t = float(wcfg.get("lambda_t", 0.0))
        lam_f = float(wcfg.get("lambda_f", 0.0))
        info = {}
    if mode in ("product", "prod", "joint"):
        cost = product_cost(a, b, lam_t, lam_f, norm, stride)
        costs = [("product", cost)]
    else:
        costs = [("temporal", ground_cost(a.mean(3).transpose(1, 2),
                                          b.mean(3).transpose(1, 2),
                                          "physical", lam_t, norm)),
                 ("frequency", ground_cost(a.mean(2).transpose(1, 2),
                                           b.mean(2).transpose(1, 2),
                                           "physical", lam_f, norm))]
    e0, e1 = float(wcfg["eps_init"]), float(wcfg["eps_final"])
    for name, cost in costs:
        mc = cost.mean().item()
        r0, r1 = mc / e0, mc / e1
        flag = "FAIL" if r1 > 50 else "OK"
        say(flag, f"{name}: mean cost={mc:7.3f}  cost/eps={r0:7.2f} -> {r1:7.2f} "
                  f"(eps {e0} -> {e1}, mode={mode}, normalize={norm}, "
                  f"cells={cost.shape[-1]});  >50 means a hard assignment")
        if flag == "FAIL":
            print("       fix: wcl.cost_normalize: mean, or raise eps so that "
                  "mean_cost/eps stays in [2, 10]")


# ---------------------------------------------------------------- check 5
def _probe_wave(cfg: dict, seconds: float = 4.0) -> np.ndarray:
    """Synthetic voiced signal with silences: enough structure for an energy
    detector and for a trim to be measurable without the dataset present."""
    sr = int(cfg["data"]["sr"])
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(0)
    x = np.zeros_like(t)
    for k, amp in ((1, 0.5), (2, 0.25), (3, 0.12), (5, 0.06)):
        x += amp * np.sin(2 * np.pi * 120.0 * k * t)
    # amplitude envelope: two "words" separated by a 0.4 s pause
    env = np.zeros_like(t)
    env[(t > 0.3) & (t < 1.4)] = 1.0
    env[(t > 1.8) & (t < 3.1)] = 1.0
    x = x * env + 0.01 * rng.standard_normal(t.shape)
    return x.astype(np.float32)


def check_jitter(cfg: dict) -> None:
    from wcl.afe import apply_cascade
    print("\n--- 5. is the jitter actually consumed by the cascade? ---")
    x = _probe_wave(cfg)
    sr = int(cfg["data"]["sr"])
    a = apply_cascade(x, sr, cfg, seed=7)
    b = apply_cascade(x, sr, cfg, seed=8)
    # the truncating VAD changes the length, so compare on the common prefix and
    # report the length change separately
    L = min(len(a), len(b))
    rel = float(np.mean(np.abs(a[:L] - b[:L])) / (np.mean(np.abs(x)) + 1e-12))
    dlen = abs(len(a) - len(b)) / float(len(x))
    rand = bool(cfg["afe"].get("randomize", False))
    print(f"       seed 7 vs seed 8: rel.diff = {rel:.3e}, length change = "
          f"{100*dlen:.1f}% (afe.randomize = {rand}, jitter = "
          f"{cfg['afe'].get('jitter')})")
    if rand and rel < 1e-3:
        say("FAIL", "randomize=True but two seeds give the same distortion: "
                    "the parameter is not consumed, so training sees one fixed "
                    "point instead of the family the theory assumes")
    elif rand:
        say("OK  ", f"jitter is consumed (rel.diff={rel:.3e} > 1e-3)")
    else:
        say("INFO", "randomize=False: one fixed distortion by design (the "
                    "wcl_nojitter ablation); not a gate failure")


def check_vad_removal(cfg: dict) -> None:
    from wcl.afe import vad_truncate
    print("\n--- 6. does the truncating VAD shift the support? ---")
    x = _probe_wave(cfg)
    sr = int(cfg["data"]["sr"])
    p = cfg["afe"]["vad"]
    mode = str(p.get("mode", "truncate")).lower()
    if mode != "truncate":
        say("FAIL", f"afe.vad.mode = {mode}: the VAD preserves the length and "
                    f"the support is never truncated, so the displacement the "
                    f"theory models does not occur in training")
        return
    y = vad_truncate(x, sr, quantile=float(p.get("quantile", 0.25)),
                     pad=int(p.get("pad", 3)),
                     min_keep_s=float(p.get("min_keep_s", 0.25)),
                     extra_trim_s=float(p.get("extra_trim_s", 0.0)),
                     extra_trim_jitter_s=float(p.get("extra_trim_jitter_s", 0.0)),
                     rng=np.random.default_rng(7))
    removed = 1.0 - len(y) / len(x)
    trim_ms = 1000.0 * (len(x) - len(y)) / sr
    print(f"       removed {100*removed:.1f}% of the samples, "
          f"head+tail shift = {trim_ms:.0f} ms (synthetic probe)")
    if removed < 0.02:
        say("FAIL", f"VAD removes only {100*removed:.1f}% of the signal; on a "
                    f"corpus cut this tight the support never moves")
    else:
        say("OK  ", f"VAD removes {100*removed:.1f}% ({trim_ms:.0f} ms); "
                    f"the support shift is real")

def check_geometry_scale(cfg: dict, device: str) -> None:
    """Is |i-j| visible next to ||v_i - v_j||?"""
    print("\n--- 7. is the geometric term visible? (report, not a gate) ---")
    wcfg = cfg["wcl"]
    stride = int(cfg["wcl"].get("grid_stride", 1))
    a, b = _dummy_grids(cfg, device, batch=2)
    lam_t, lam_f, info = calibrate_geometry(
        a, b, float(wcfg.get("beta_t", 1.0)), float(wcfg.get("beta_f", 1.0)),
        stride, shift_cells=afe_trim_cells(cfg.get("afe", {}), stride),
        band_cells=float(wcfg.get("band_cells", 1.0)))
    fixed_t = float(wcfg.get("lambda_t", 1.0))
    m = info["mean_feature_cost"]
    # normalised displacement of a 100 ms trim, in grid units
    if cfg["feature"].get("type") == "raw":
        cells = int(round(float(cfg["feature"].get("duration", 4.0)) * 16000)) // 320
    else:
        cells = int(cfg["feature"].get("max_frames", 640)) // 16
    cells = max(cells // max(stride, 1), 2)
    trim_100ms = 0.1 * 16000 / 320 / max(stride, 1)      # frames, then / (T-1)
    frac = trim_100ms / (cells - 1)
    with_fixed = fixed_t * frac
    with_calib = lam_t * frac
    print(f"       mean feature cost = {m:.3f}; 100 ms trim = {frac:.4f} of the span")
    print(f"       lam_t=1 (paper default) -> geometric cost of that trim = "
          f"{with_fixed:.4f}  ({with_fixed / m * 100:.2f}% of a typical feature gap)")
    print(f"       lam_t={lam_t:.1f} (calibrated)  -> {with_calib:.4f}  "
          f"({with_calib / m * 100:.1f}% of a typical feature gap)")
    if str(wcfg.get("geo_mode", "calibrate")).lower() == "calibrate":
        if with_calib < 0.05 * m:
            say("FAIL", "even after calibration the geometric term is <5% of a "
                        "typical feature gap: OT will match by appearance only")
        else:
            say("OK", f"geometry carries {with_calib / m * 100:.0f}% of a typical "
                      f"feature gap for a 100 ms trim (lam_t={lam_t:.1f}, "
                      f"lam_f={lam_f:.1f})")
    else:
        flag = "FAIL" if with_fixed < 0.05 * m else "OK"
        say(flag, f"geo_mode=fixed, lam_t={fixed_t}: the trim weighs "
                  f"{with_fixed / m * 100:.2f}% of a feature gap")


# ---------------------------------------------------------------- check 6
def check_cache_budget(cfg: dict) -> None:
    print("\n--- 8. cache budget and throughput (report, not a gate) ---")
    counts = {"train": 25380, "dev": 24844, "eval": 71237}
    if cfg["feature"].get("type") == "raw":
        n = int(round(int(cfg["data"]["sr"]) * float(cfg["feature"].get("duration", 4.0))))
        per = n * 4 + 128
    else:
        f = cfg["feature"]
        per = int(f["max_frames"]) * int(f["n_mels"]) * 4 + 128
    tot_gb = 0.0
    for split, k in counts.items():
        gb = k * 5 * per / 1e9
        tot_gb += gb
        print(f"       {split:5s}: {k:6d} utt x 5 phases x {per/1024:6.0f} KiB "
              f"= {gb:6.1f} GB")
    say("INFO", f"total cache for train+dev+eval = {tot_gb:.0f} GB, "
                f"{sum(counts.values())*5:,} files")
    bs = int(cfg["train"].get("batch_size", 8))
    steps = counts["train"] // bs
    print(f"       train steps/epoch = {steps} (batch {bs}); "
          f"dev eval = {counts['dev'] * len(cfg['eval']['phases'])} forwards")
    print("       measured on the reference run: 331 s optim + ~11 min dev "
          "(eval_every=1). With eval_every=3 the average is ~%d s/epoch."
          % (331 + 660 // 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-model", action="store_true",
                    help="skip check 1 (no torch model needed)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config = {args.config}   device = {device}   "
          f"arch = {cfg['model'].get('arch', 'cnn')}")

    if not args.skip_model:
        check_consistency_gradient(cfg, device)
    check_agc_phase(cfg)
    check_objective_matches_theorem(cfg, device)
    check_eps_scale(cfg, device)
    check_jitter(cfg)
    check_vad_removal(cfg)
    check_geometry_scale(cfg, device)
    check_cache_budget(cfg)

    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} blocking issue(s) found -- do not start a long run yet:")
        for f in FAILED:
            print(f"  * {f}")
        w = cfg.get("wcl", {})
        if (str(w.get("cost_mode", "product")).lower() != "product"
                or float(w.get("beta_t", 1.0)) == 0.0
                or str(w.get("geo_mode", "calibrate")).lower() == "fixed"):
            print("\n  note: this config deliberately switches off one of the "
                  "components.\n  If it comes from campaign/*/abl_* or _afe, a "
                  "FAIL here is the expected reading,\n  not a defect.")
        return 1
    print("no blocking issue found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
