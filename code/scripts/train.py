#!/usr/bin/env python3
"""Train WCL (Wasserstein consistency learning) on ASVspoof 2019 LA.

Implements the paper's protocol (Section 5.4):
  - dual-view training: clean view + AFE-distorted view of the same utterance;
  - objective: L = CE(clean) + CE(distorted) + lambda*(W_t + W_f), with
    Sinkhorn-approximated W_{1,eps} (k = 20) and linear eps annealing;
  - Adam (lr 1e-4, wd 1e-4), early stopping on the dev set;
  - stage-wise dev evaluation (clean -> AEC -> ... -> full cascade).

Usage:
    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/train_full.yaml --tag run1
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from wcl.dataset import build_train_loader, load_meta, WCLPhaseDataset
from wcl.losses import (wcl_loss, annealed_eps, resolve_lambda, calibrate_geometry,
                        TFCLConsistency)
from wcl.model import build_model
from wcl.utils import (load_config, set_seed, get_device, compute_eer,
                       compute_auc, Logger, afe_trim_cells)


def evaluate_phases(model, cfg, split: str, phases, device,
                    logger: Logger, meta: list[dict] | None = None) -> dict:
    """Stage-wise EER/AUC over the given AFE phases."""
    model.eval()
    cache_dir = Path(cfg["data"]["cache_dir"])
    if meta is None:
        meta = load_meta(cache_dir, split)
    results = {}
    with torch.no_grad():
        for phase in phases:
            ds = WCLPhaseDataset(meta, cache_dir, split, phase,
                                 feature_type=cfg["feature"].get("type", "logmel"))
            loader = DataLoader(ds, batch_size=cfg["eval"].get("batch_size", 32),
                                num_workers=cfg["eval"].get("num_workers", 2),
                                pin_memory=torch.cuda.is_available())
            scores, labels = [], []
            for x, y in loader:
                x = x.to(device)
                logits, _ = model(x)
                p = torch.softmax(logits, dim=1)[:, 1]  # bonafide probability
                scores.append(p.detach().cpu().numpy())
                labels.append(y.numpy())
            scores = np.concatenate(scores)
            labels = np.concatenate(labels)
            results[phase] = {
                "eer": compute_eer(scores, labels),
                "auc": compute_auc(scores, labels),
            }
    model.train()
    return results


def calibrate_lambda(model, loader, cfg, wcfg, device, n_batches: int = 2,
                     tfcl_mod=None, loss_mode: str = "wcl",
                     lam_t: float | None = None,
                     lam_f: float | None = None) -> dict:
    """One-shot calibration of the two scale-carrying numbers.

    ``lam_eff = alpha * CE_ref / cons_ref`` (Remark 1) and
    ``lam_t = beta * mean feature cost / displacement`` (see
    ``calibrate_geometry``).  Both are ratios measured on the initial model, so
    neither has to be guessed per backbone.  The same routine serves the
    baseline arm, which is what makes the comparison controlled: the cosine+CKA
    arm gets the weight that makes *its* consistency term the same fraction of
    the classification loss at initialisation, instead of reusing a constant
    that happens to be right for a different functional form.
    """
    eps0 = float(wcfg["eps_init"])
    geo_mode = str(wcfg.get("geo_mode", "calibrate")).lower()
    stride = int(wcfg.get("grid_stride", 1))
    shift_cells = afe_trim_cells(cfg.get("afe", {}), stride)
    ce_acc = cons_acc = 0.0
    seen = 0
    geo_t, geo_f = lam_t, lam_f
    geo_info: dict = {}
    it = iter(loader)
    model.eval()
    with torch.no_grad():
        for _ in range(min(n_batches, len(loader))):
            try:
                clean, distorted, labels = next(it)
            except StopIteration:
                break
            clean, distorted, labels = (clean.to(device), distorted.to(device),
                                        labels.to(device))
            logits_c, feat_c = model(clean)
            logits_d, feat_d = model(distorted)

            if loss_mode == "wcl" and geo_mode == "calibrate":
                lt, lf, info = calibrate_geometry(
                    feat_c.float(), feat_d.float(),
                    beta_t=float(wcfg.get("beta_t", 1.0)),
                    beta_f=float(wcfg.get("beta_f", 1.0)), stride=stride,
                    shift_cells=shift_cells,
                    band_cells=float(wcfg.get("band_cells", 1.0)))
                geo_t, geo_f = lt, lf
                geo_info = info

            if loss_mode == "tfcl" and tfcl_mod is not None:
                cons, c = tfcl_mod(feat_c.float(), feat_d.float())
                ce = (F.cross_entropy(logits_c.float(), labels)
                      + F.cross_entropy(logits_d.float(), labels))
                ce_acc += float(ce.item())
                cons_acc += float((c["w_t"] + c["w_f"]))
            elif loss_mode == "wcl":
                _, comp = wcl_loss(
                    logits_c.float(), logits_d.float(), labels,
                    feat_c.float(), feat_d.float(),
                    lam=0.0, eps=eps0,
                    n_iter=wcfg.get("sinkhorn_iters", 20),
                    use_temporal=wcfg.get("use_temporal", True),
                    use_frequency=wcfg.get("use_frequency", True),
                    cost_mode=wcfg.get("cost_mode", "product"),
                    lambda_t=geo_t or 0.0, lambda_f=geo_f or 0.0,
                    cost_normalize=wcfg.get("cost_normalize", "mean"),
                    grid_stride=stride,
                    return_components=True)
                ce_acc += comp["ce_c"] + comp["ce_d"]
                cons_acc += comp["cons"]
            seen += 1
    model.train()
    out = {"lam_t": geo_t, "lam_f": geo_f, "geo_info": geo_info,
           "lam": None, "ce_ref": None, "cons_ref": None}
    if seen == 0:
        out["lam"] = float(wcfg.get("alpha", 0.3))
        return out
    out["ce_ref"] = ce_acc / seen
    out["cons_ref"] = cons_acc / seen
    out["lam"] = resolve_lambda("ratio", 0.0, float(wcfg.get("alpha", 0.3)),
                                out["ce_ref"], out["cons_ref"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None,
                    help="override data.root (e.g. synthetic data dir)")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--loss", default=None,
                    choices=["auto", "ce", "tfcl", "wcl"],
                    help="training objective; auto reads wcl.loss_mode from the config")
    ap.add_argument("--resume", default=None,
                    help="path to a checkpoint to resume from (restores weights, "
                         "epoch, best EER and lambda_eff)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override train.seed (multi-seed campaign)")
    ap.add_argument("--cache-dir", default=None,
                    help="override data.cache_dir (e.g. an unseen-distortion holdout)")
    ap.add_argument("--out-dir", default=None,
                    help="override out_dir (keeps the campaign tree tidy)")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="override train.eval_every; raise it to trade dev "
                         "resolution for GPU hours")
    ap.add_argument("--dev-limit", type=int, default=None,
                    help="subsample the dev set used for early stopping "
                         "(the final numbers must still come from the full dev)")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip dev evaluation entirely (smoke / throughput tests)")
    ap.add_argument("--abort-on-flat", action="store_true",
                    help="stop this run if the consistency term has not decreased "
                         "after --flat-patience epochs (a flat term means the "
                         "gradient is not reaching the transport, and a long run "
                         "on a broken objective only burns GPU hours)")
    ap.add_argument("--flat-patience", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.root:
        cfg["data"]["root"] = args.root
    if args.cache_dir:
        cfg["data"]["cache_dir"] = args.cache_dir
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    if args.eval_every is not None:
        cfg["train"]["eval_every"] = int(args.eval_every)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    out_dir = Path(cfg["out_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)
    logger.save_json("run_config.json", cfg)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg["train"].get("seed", 0))
    logger.log(f"[cfg] device={device} out_dir={out_dir}")

    train_loader = build_train_loader(cfg, "train", shuffle=True)
    dev_meta = load_meta(Path(cfg["data"]["cache_dir"]), "dev")
    if args.dev_limit:
        # stratified would be better, but a deterministic prefix is enough for
        # model selection and keeps the per-epoch cost predictable
        dev_meta = dev_meta[: args.dev_limit]
    logger.log(f"[data] train={len(train_loader.dataset)} dev={len(dev_meta)} "
               f"(dev{' SUBSAMPLED' if args.dev_limit else ''})")

    model = build_model(cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    wcfg = cfg["wcl"]
    tcfg = cfg["train"]
    loss_mode = args.loss if (args.loss and args.loss != "auto") else str(
        wcfg.get("loss_mode", "wcl")).lower()

    # TFCL-style baseline needs learnable Q/K/V projections (same budget as
    # the reference design); WCL and pure-CE need no extra parameters.
    tfcl_mod = None
    if loss_mode == "tfcl":
        feat_dim = int(getattr(model, "out_dim",
                               cfg["model"].get("embed_dim", 128)))
        tfcl_mod = TFCLConsistency(
            feat_dim, cka_dim=str(wcfg.get("tfcl_cka_dim", "cell"))).to(device)
        logger.log(f"[baseline] TFCL consistency head: dim={feat_dim} "
                   f"cka_dim={wcfg.get('tfcl_cka_dim', 'cell')}")
        if str(wcfg.get("tfcl_cka_dim", "cell")).lower() == "batch" \
                and int(cfg["train"]["batch_size"]) < 32:
            logger.log("[WARN] TFCL cka_dim=batch with batch "
                       f"{cfg['train']['batch_size']}: the centred matrices have "
                       "rank <= B-1, so the CKA estimator is biased upward "
                       "(~0.89 at B=8) and the term is nearly constant. "
                       "Use wcl.tfcl_cka_dim: cell for the strong baseline.")
    extra = list(tfcl_mod.parameters()) if tfcl_mod is not None else []
    # gradient clipping has to cover the extra head as well, otherwise the
    # baseline arm is trained with an unclipped consistency head
    clip_params = trainable + extra

    # A consistency term computed on a frozen representation cannot train
    # anything: warn loudly instead of producing a plausible-looking log.
    if (loss_mode in ("wcl", "tfcl")
            and cfg["model"].get("arch") == "xlsr_aasist"
            and str(cfg["model"].get("measure_from", "frontend")) == "frontend"
            and bool(cfg["model"].get("frontend", {}).get("freeze", True))):
        logger.log("[WARN] measure_from=frontend with frontend.freeze=true: the "
                   "consistency term has NO gradient path to any trainable "
                   "parameter and this run is pure dual-view cross-entropy. "
                   "Set model.measure_from: backend.")

    logger.log(f"[model] arch={cfg['model'].get('arch', 'cnn')} "
               f"loss={loss_mode} "
               f"measure_from={cfg['model'].get('measure_from', '-')} "
               f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
               f"trainable={sum(p.numel() for p in trainable) / 1e6:.2f}M")
    if loss_mode == "wcl":
        acfg = cfg.get("afe", {})
        vp = (acfg.get("vad", {}) or {})
        logger.log(f"[geo] cost_mode={wcfg.get('cost_mode', 'euclidean')} "
                   f"cost_normalize={wcfg.get('cost_normalize', 'none')} "
                   f"geo_mode={wcfg.get('geo_mode', 'calibrate')} "
                   f"beta_t={wcfg.get('beta_t', 1.0)} "
                   f"beta_f={wcfg.get('beta_f', 1.0)} "
                   f"grid_stride={wcfg.get('grid_stride', 1)} | "
                   f"vad.mode={vp.get('mode', 'truncate')} "
                   f"truncate_mode={vp.get('truncate_mode', 'span')} "
                   f"quantile={vp.get('quantile', 0.25)} | "
                   f"afe.randomize={acfg.get('randomize', False)} "
                   f"afe.jitter={acfg.get('jitter', 0.4)} | "
                   f"cache_dir={cfg['data'].get('cache_dir')}")
    optimizer = torch.optim.Adam(trainable + extra,
                                 lr=tcfg["lr"],
                                 weight_decay=tcfg["weight_decay"])
    epochs = tcfg["epochs"]
    phases = cfg["eval"]["phases"]
    # cost lever: the five-stage dev sweep is ~11 min/epoch. Restricting it to
    # the two stages the claim is judged on (clean + full cascade) roughly
    # halves the run at the price of a coarser early-stopping criterion. The
    # final table still comes from scripts/evaluate.py over all five.
    dev_phases = list(tcfg.get("dev_phases") or phases)
    if dev_phases != phases:
        logger.log(f"[dev] early stopping on {dev_phases} only "
                   f"(final eval still uses {phases})")
    accum = max(int(tcfg.get("grad_accum", 1)), 1)

    lam_mode = str(wcfg.get("lambda_mode", "abs")).lower()
    geo_mode = str(wcfg.get("geo_mode", "calibrate")).lower()
    lam_t = float(wcfg.get("lambda_t", 0.0))
    lam_f = float(wcfg.get("lambda_f", 0.0))
    if loss_mode == "wcl" and geo_mode == "calibrate":
        lam_t = lam_f = None          # decided below, from the initial batch
    if loss_mode in ("wcl", "tfcl") and lam_mode == "ratio":
        cal = calibrate_lambda(model, train_loader, cfg, wcfg, device,
                               tfcl_mod=tfcl_mod, loss_mode=loss_mode,
                               lam_t=lam_t, lam_f=lam_f)
        lam_eff = float(cal["lam"])
        if cal["lam_t"] is not None:
            lam_t, lam_f = cal["lam_t"], cal["lam_f"]
        logger.log(f"[wcl] lambda_mode=ratio alpha={wcfg.get('alpha', 0.3)} "
                   f"CE_ref={cal['ce_ref']:.4f} cons_ref={cal['cons_ref']:.4f} "
                   f"-> lambda_eff={lam_eff:.4f}  (loss={loss_mode})")
        if cal["geo_info"]:
            gi = cal["geo_info"]
            logger.log(f"[geo] beta_t={wcfg.get('beta_t', 1.0)} "
                       f"beta_f={wcfg.get('beta_f', 1.0)}: "
                       f"mean_feature_cost={gi['mean_feature_cost']:.3f} "
                       f"trim={gi['shift_cells']:.2f} cells "
                       f"-> lambda_t={lam_t:.1f} lambda_f={lam_f:.1f} "
                       f"(a trim of that size now costs {wcfg.get('beta_t', 1.0)} "
                       "mean feature gaps)")
    else:
        lam_eff = float(wcfg.get("lambda", 0.0))
        logger.log(f"[wcl] loss={loss_mode} lambda_abs={lam_eff:.4f}")
    lam_t = float(lam_t or 0.0)
    lam_f = float(lam_f or 0.0)

    use_amp = tcfg.get("fp16", False) and torch.cuda.is_available()
    amp_dtype = (torch.bfloat16
                 if str(tcfg.get("amp_dtype", "fp16")).lower() == "bf16"
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16) if use_amp else None

    start_epoch = 0
    best_avg = float("inf")
    history = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        start_epoch = int(ckpt.get("epoch", 0))
        best_avg = float(ckpt.get("best_avg_eer", best_avg))
        if ckpt.get("lambda_eff") is not None:
            lam_eff = float(ckpt["lambda_eff"])
        if ckpt.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                logger.log("[resume] optimizer/Adam state restored")
            except Exception as exc:      # shape drift after a code change
                logger.log(f"[resume] WARN could not restore optimizer state: {exc}")
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            try:
                history = json.loads(hist_path.read_text())
            except Exception:
                history = []
        logger.log(f"[resume] {args.resume} -> restart at epoch {start_epoch + 1}, "
                   f"best_avg_eer={best_avg:.4f}, lambda_eff={lam_eff:.4f}")

    # Prop. 4 (correct form): lambda * W_{1,eps}(c) = W_{1,lambda*eps}(lambda*c),
    # so with a *fixed* cost the pair enters only through (c/eps, lambda*eps):
    # eps sets the temperature and lambda*eps sets the strength at that
    # temperature.  Annealing eps with lambda held fixed therefore weakens the
    # consistency term by eps_init/eps_final (5x with 0.5 -> 0.1) relative to
    # CE.  ``lambda_couple: const_product`` cancels that by holding lambda*eps
    # fixed, which is a pure temperature anneal; ``none`` reproduces the
    # schedule described in the draft.
    couple = str(wcfg.get("lambda_couple", "none")).lower()
    lam_now = lam_eff

    best_ckpt = out_dir / "best.pt"
    patience = tcfg.get("early_stop_patience", 10)
    no_improve = 0
    epoch = start_epoch - 1

    def objective(logits_c, logits_d, feat_c, feat_d, labels, eps):
        """Task + consistency loss on already-computed views."""
        if loss_mode == "ce":
            ce_c = F.cross_entropy(logits_c, labels)
            ce_d = F.cross_entropy(logits_d, labels)
            comp = {"ce_c": ce_c.item(), "ce_d": ce_d.item(),
                    "w_t": 0.0, "w_f": 0.0, "w_prod": 0.0, "cons": 0.0}
            return ce_c + ce_d, comp
        if loss_mode == "tfcl":
            ce_c = F.cross_entropy(logits_c, labels)
            ce_d = F.cross_entropy(logits_d, labels)
            cons, c = tfcl_mod(feat_c, feat_d)
            comp = {"ce_c": ce_c.item(), "ce_d": ce_d.item(), **c,
                    "w_prod": 0.0, "cons": float(cons.item())}
            return ce_c + ce_d + lam_now * cons, comp
        loss, comp = wcl_loss(
            logits_c, logits_d, labels, feat_c, feat_d,
            lam=lam_now, eps=eps,
            n_iter=wcfg.get("sinkhorn_iters", 20),
            use_temporal=wcfg.get("use_temporal", True),
            use_frequency=wcfg.get("use_frequency", True),
            cost_mode=wcfg.get("cost_mode", "product"),
            lambda_t=lam_t,
            lambda_f=lam_f,
            cost_normalize=wcfg.get("cost_normalize", "mean"),
            grid_stride=int(wcfg.get("grid_stride", 1)),
            return_components=True)
        return loss, comp

    cons_ref = None            # first-epoch consistency value, for the flat guard
    flat_epochs = 0
    for epoch in range(start_epoch, epochs):
        eps = annealed_eps(wcfg["eps_init"], wcfg["eps_final"], epoch, epochs)
        lam_now = (lam_eff * float(wcfg["eps_init"]) / max(eps, 1e-9)
                   if couple == "const_product" else lam_eff)
        model.train()
        t0 = time.time()
        tot_loss, tot_ce, tot_wt, tot_wf = 0.0, 0.0, 0.0, 0.0
        tot_cons, tot_wp, n_batch = 0.0, 0.0, 0
        for i, (clean, distorted, labels) in enumerate(train_loader):
            clean, distorted, labels = (clean.to(device), distorted.to(device),
                                        labels.to(device))
            if i % accum == 0:
                optimizer.zero_grad(set_to_none=True)

            def step():
                logits_c, feat_c = model(clean)
                logits_d, feat_d = model(distorted)
                # The transport loss is numerically delicate (cost/eps lives in
                # the log-domain kernel) and cheap next to the backbone, so it
                # is evaluated in fp32 even under bf16 autocast: in bf16 the
                # ~0.4% error of the cost matrix is amplified by 1/eps.
                with torch.autocast(device_type="cuda", enabled=False):
                    return objective(logits_c.float(), logits_d.float(),
                                     feat_c.float(), feat_d.float(), labels, eps)

            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss, comp = step()
                loss_scaled = loss / accum
                if scaler is not None:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(clip_params,
                                                 tcfg.get("max_grad_norm", 5.0))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(clip_params,
                                                 tcfg.get("max_grad_norm", 5.0))
                        optimizer.step()
            else:
                loss, comp = step()
                (loss / accum).backward()
                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(clip_params,
                                             tcfg.get("max_grad_norm", 5.0))
                    optimizer.step()

            tot_loss += loss.item(); tot_ce += comp["ce_c"] + comp["ce_d"]
            tot_wt += comp["w_t"]; tot_wf += comp["w_f"]
            tot_cons += comp.get("cons", 0.0)
            tot_wp += comp.get("w_prod", 0.0)
            n_batch += 1

        avg = {k: v / max(n_batch, 1) for k, v in
               [("loss", tot_loss), ("ce", tot_ce), ("w_t", tot_wt),
                ("w_f", tot_wf), ("cons", tot_cons), ("w_prod", tot_wp)]}
        logger.log(f"[train] epoch {epoch + 1}/{epochs} eps={eps:.4f} "
                   f"lam={lam_now:.4f} loss={avg['loss']:.4f} ce={avg['ce']:.4f} "
                   f"cons={avg['cons']:.4f} w_t={avg['w_t']:.4f} "
                   f"w_f={avg['w_f']:.4f} w_prod={avg['w_prod']:.4f} "
                   f"({time.time() - t0:.1f}s)")

        lam_eps = float(lam_now) * float(eps)      # the strength, not the weight
        logger.log(f"[lam] lambda={lam_now:.4f} eps={eps:.4f} lambda*eps={lam_eps:.4f}"
                   + ("  (fixed product: anneal is pure temperature)"
                      if couple == "const_product" else
                      "  (fixed lambda: strength falls with eps)"))
        if avg["cons"] > 0:
            if cons_ref is None:
                cons_ref = avg["cons"]
            elif avg["cons"] > cons_ref * 0.98:
                flat_epochs += 1
                logger.log(f"[guard] consistency term flat ({avg['cons']:.4f} vs "
                           f"{cons_ref:.4f}) {flat_epochs}/{args.flat_patience}")
                if args.abort_on_flat and flat_epochs >= args.flat_patience:
                    logger.log("[guard] ABORT: the transport term is not being "
                               "optimised; fix the objective before spending more "
                               "GPU hours on this run")
                    logger.save_json("history.json", history)
                    return 2
            else:
                flat_epochs = 0

        rec = {"epoch": epoch + 1, "eps": eps, "lambda": lam_now,
               "lambda_eps": lam_eps, **avg}
        if not args.no_eval and (epoch + 1) % int(tcfg.get("eval_every", 1)) == 0:
            dev = evaluate_phases(model, cfg, "dev", dev_phases, device, logger,
                                  meta=dev_meta)
            avg_eer = sum(v["eer"] for v in dev.values()) / len(dev)
            line = " | ".join(f"{p}={dev[p]['eer'] * 100:.2f}%" for p in dev_phases)
            logger.log(f"[dev] avgEER={avg_eer * 100:.2f}%  {line}")
            rec.update({"avg_eer": avg_eer,
                        **{p: v for p, v in dev.items()}})
            if avg_eer < best_avg:
                best_avg = avg_eer
                ck = {"model": model.state_dict(), "config": cfg,
                      "epoch": epoch + 1, "best_avg_eer": best_avg,
                      "lambda_mode": lam_mode, "lambda_eff": lam_eff,
                      "lambda_t": lam_t, "lambda_f": lam_f,
                      "optimizer": optimizer.state_dict()}
                torch.save(ck, best_ckpt)
                logger.log(f"[ckpt] saved {best_ckpt} (avgEER={best_avg * 100:.2f}%)")
                no_improve = 0
            else:
                no_improve += 1
        history.append(rec)
        # write after every epoch: an 11-hour run must not lose its history to
        # one crash, and "last" has to be resumable from the exact epoch
        logger.save_json("history.json", history)
        torch.save({"model": model.state_dict(), "config": cfg,
                    "epoch": epoch + 1, "best_avg_eer": best_avg,
                    "lambda_mode": lam_mode, "lambda_eff": lam_eff,
                    "lambda_t": lam_t, "lambda_f": lam_f,
                    "optimizer": optimizer.state_dict()},
                   out_dir / "last.pt")
        if no_improve >= patience and not args.no_eval:
            logger.log(f"[stop] early stopping at epoch {epoch + 1}")
            break

    logger.save_json("history.json", history)
    # save the final (last-epoch) model as well, independent of early stopping
    torch.save({"model": model.state_dict(), "config": cfg,
                "epoch": epoch + 1, "best_avg_eer": best_avg,
                "lambda_mode": lam_mode, "lambda_eff": lam_eff,
                "lambda_t": lam_t, "lambda_f": lam_f,
                "optimizer": optimizer.state_dict()},
               out_dir / "final.pt")
    logger.log(f"[done] best avg EER = {best_avg * 100:.2f}%  ->  {best_ckpt}")
    logger.log(f"[done] final model saved ->  {out_dir / 'final.pt'}")
    logger.log("Evaluate with:  python scripts/evaluate.py "
               f"--config {args.config} --ckpt {best_ckpt}")


if __name__ == "__main__":
    main()
