"""Unit tests: Sinkhorn implementation, WCL loss, AFE pipeline, features.

Run from the project root:
    python -m pytest tests -q
    (or: python tests/test_wcl.py)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wcl.losses import (sinkhorn_plan, wasserstein_1_entropic, wcl_loss,
                         annealed_eps, product_cost, calibrate_geometry,
                         ground_cost)
from wcl.afe import apply_stage, apply_cascade, cascade_prefix, STAGES
from wcl.features import log_mel, pad_or_crop, normalize
from wcl.model import build_model


# ------------------------------------------------------------------ Sinkhorn
def test_sinkhorn_marginals():
    torch.manual_seed(0)
    cost = torch.rand(2, 6, 6) * 3.0
    P = sinkhorn_plan(cost, eps=0.1, n_iter=50)
    row = P.sum(dim=2)   # [B, n]
    col = P.sum(dim=1)   # [B, m]
    # finite-k entropic Sinkhorn leaves a small residual (~1e-3)
    assert torch.allclose(row, torch.ones_like(row) / 6, atol=5e-3)
    assert torch.allclose(col, torch.ones_like(col) / 6, atol=5e-3)


def test_sinkhorn_marginals_are_not_forced_on_one_side():
    """Regression: a final unilateral column correction used to break the row
    sums that the iteration had just converged. Both residuals must stay small
    on their own, without any forcing."""
    from wcl.losses import marginal_residual
    torch.manual_seed(0)
    cost = torch.rand(2, 6, 6) * 3.0
    P = sinkhorn_plan(cost, eps=0.1, n_iter=50)
    row_dev, col_dev = marginal_residual(P)
    assert row_dev < 5e-3 and col_dev < 5e-3
    # and the plan is not a permutation: a forced column fix would concentrate
    # mass one-per-column
    assert (P > 1e-4).float().sum() > 6 * 2


def test_sinkhorn_large_eps_uniform():
    """As eps -> inf the plan approaches the uniform (independent) coupling."""
    cost = torch.tensor([[0.0, 5.0, 9.0], [5.0, 0.0, 4.0], [9.0, 4.0, 0.0]]).unsqueeze(0)
    P = sinkhorn_plan(cost, eps=50.0, n_iter=500)
    assert torch.allclose(P, torch.full_like(P, 1.0 / 9.0), atol=2e-2)


def test_sinkhorn_small_eps_diagonal():
    """As eps -> 0 the plan concentrates on the zero-cost diagonal:
    with n=m=2 the diagonal coupling P00=P11=1/n=0.5 is the optimum, and the
    off-diagonal mass vanishes."""
    cost = torch.tensor([[0.0, 10.0], [10.0, 0.0]]).unsqueeze(0)
    P = sinkhorn_plan(cost, eps=0.01, n_iter=200)
    assert P[0, 0, 0].item() > 0.49 and P[0, 1, 1].item() > 0.49
    assert P[0, 0, 1].item() < 0.01 and P[0, 1, 0].item() < 0.01


def test_wasserstein_value_symmetric():
    cost = torch.rand(4, 8, 8) * 2.0
    w = wasserstein_1_entropic(cost, eps=0.1, n_iter=30)
    assert w.shape == (4,)
    assert torch.all(w >= 0)


def test_annealing():
    e0, e1 = annealed_eps(0.1, 0.01, 0, 100), annealed_eps(0.1, 0.01, 99, 100)
    assert abs(e0 - 0.1) < 1e-6 and abs(e1 - 0.01) < 1e-6


# ------------------------------------------------------------------ WCL loss
def test_wcl_loss_shapes_and_entropic_identity():
    """Identical views: W_{1,eps}(mu,mu) = eps*log(n) is an intrinsic bias of
    entropic OT (KL term), so the consistency terms equal eps*log(T') and
    eps*log(F') exactly — matching the W_{1,eps} definition of the paper.
    """
    import math
    torch.manual_seed(0)
    B, C, Tp, Fp = 4, 64, 10, 4
    feat = torch.randn(B, C, Tp, Fp)
    logits = torch.randn(B, 2)
    labels = torch.randint(0, 2, (B,))
    loss, comp = wcl_loss(logits, logits, labels, feat, feat,
                          lam=1.0, eps=0.1, cost_mode="marginal",
                          cost_normalize="none", return_components=True)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert abs(comp["w_t"] - 0.1 * math.log(10)) < 1e-2
    assert abs(comp["w_f"] - 0.1 * math.log(4)) < 1e-2
    # loss = CE(clean) + CE(distorted) + lam*(W_t + W_f)
    ce = torch.nn.functional.cross_entropy(logits, labels) * 2
    expected = ce + 1.0 * (comp["w_t"] + comp["w_f"])
    assert abs(loss.item() - expected) < 1e-3


def test_wcl_loss_displaced_views_larger():
    """Moved feature measures should give a larger WCL loss (consistency
    penalty reacts to mass displacement — the core geometric property)."""
    torch.manual_seed(1)
    B, C, Tp, Fp = 2, 32, 8, 4
    feat = torch.randn(B, C, Tp, Fp)
    feat2 = feat + torch.randn_like(feat) * 3.0          # displaced
    feat3 = feat + torch.randn_like(feat) * 0.1          # near
    labels = torch.zeros(B, dtype=torch.long)
    lg = torch.zeros(B, 2)
    _, c_far = wcl_loss(lg, lg, labels, feat, feat2, lam=1.0, eps=0.1,
                        cost_mode="marginal", cost_normalize="none",
                        return_components=True)
    _, c_near = wcl_loss(lg, lg, labels, feat, feat3, lam=1.0, eps=0.1,
                         cost_mode="marginal", cost_normalize="none",
                         return_components=True)
    assert c_far["w_t"] > c_near["w_t"] and c_far["w_f"] > c_near["w_f"]


# ------------------------------------------------------------------ AFE
def _synthetic_speech(sr=16000, dur=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    # 3 formant-like tones + noise floor
    x = (0.5 * np.sin(2 * np.pi * 180 * t)
         + 0.3 * np.sin(2 * np.pi * 450 * t)
         + 0.2 * np.sin(2 * np.pi * 1200 * t)
         + 0.05 * rng.standard_normal(len(t)))
    return x.astype(np.float32)


def test_afe_shape_preserved():
    """AEC / NS / AGC preserve length; VAD does not (it truncates by design)."""
    x = _synthetic_speech()
    cfg = {"randomize": False}
    for stage in STAGES[:3]:
        y = apply_stage(stage, x, 16000, cfg, rng=np.random.default_rng(0))
        assert y.shape == x.shape and np.isfinite(y).all()
    y = apply_stage("vad", x, 16000, {"vad": {"mode": "zero"}})
    assert y.shape == x.shape and np.isfinite(y).all()


def test_afe_cascade_order():
    x = _synthetic_speech()
    cfg = {"randomize": False}
    prefix = cascade_prefix(STAGES, "vad")
    assert prefix == STAGES
    # prefix cascade without VAD keeps the length; with VAD it truncates
    y_pre = apply_cascade(x, 16000, cfg, seed=0, stages=cascade_prefix(STAGES, "agc"))
    assert y_pre.shape == x.shape
    y = apply_cascade(x, 16000, cfg, seed=0)
    assert len(y) <= len(x)
    assert len(y) > 0
    # cascade is not a no-op
    assert np.abs(y[:min(len(y), len(x))] - x[:min(len(y), len(x))]).max() > 1e-6


def test_afe_agc_target_rms():
    x = _synthetic_speech()
    y = apply_stage("agc", x, 16000, {"agc": {"target_rms": 0.02}},
                    rng=np.random.default_rng(0))
    rms = np.sqrt(np.mean(y ** 2))
    assert abs(rms - 0.02) < 1e-3


def test_afe_vad_zeros_frames():
    """VAD should mute a large part of a mostly-quiet signal."""
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    x = (0.001 * rng.standard_normal(16000)
         + np.where((t > 0.4) & (t < 0.6), 0.5 * np.sin(2 * np.pi * 300 * t), 0.0))
    y = apply_stage("vad", x.astype(np.float32), 16000,
                    {"vad": {"quantile": 0.3}}, rng=None)
    assert np.count_nonzero(y) < np.count_nonzero(x)


# ------------------------------------------------------------------ features
def test_features_shapes_and_normalize():
    x = _synthetic_speech()
    feat = log_mel(x, 16000, n_fft=512, hop=160, n_mels=80)
    assert feat.shape[1] == 80
    feat = pad_or_crop(feat, max_frames=320)
    assert feat.shape[0] == 320
    norm = normalize(feat, "meanstd")
    assert abs(norm.mean()) < 1e-3 and abs(norm.std() - 1.0) < 0.05
    norm2 = normalize(feat, "minmax")
    assert norm2.min() >= -1e-6 and norm2.max() <= 1.0 + 1e-6


# ------------------------------------------------------------------ model
def test_model_output_structure():
    torch.manual_seed(0)
    cfg = {"model": {"base_channels": (8, 16, 32, 64),
                     "se_blocks": (True, True, True, True),
                     "embed_dim": 32, "dropout": 0.1}}
    model = build_model(cfg)
    x = torch.randn(2, 1, 320, 80)
    logits, feat = model(x)
    assert logits.shape == (2, 2)
    assert feat.ndim == 4 and feat.shape[0] == 2
    # time-frequency grid preserved: 320/16=20, 80/16=5
    assert feat.shape[2] == 20 and feat.shape[3] == 5


# ------------------------------------------------------------------ XLS-R + AASIST
_SMALL_XLSR = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=128, conv_dim=(32, 32), conv_stride=(10, 5),
    conv_kernel=(10, 3), num_conv_pos_embeddings=16,
    num_conv_pos_embedding_groups=2, feat_extract_norm="layer",
)


def test_xlsr_frontend_shapes():
    """Tiny randomly-initialised XLS-R (no weight download): [B,N]->[B,T,D]."""
    from wcl.model_xlsr_aasist import XLSRFrontend
    front = XLSRFrontend(pretrained=False, freeze=True,
                         config_overrides=_SMALL_XLSR)
    wav = torch.randn(2, 2048)
    h = front(wav)
    assert h.shape[0] == 2 and h.shape[2] == 64 and h.shape[1] > 0
    # frozen front-end: no parameter requires grad
    assert not any(p.requires_grad for p in front.parameters())


def test_aasist_backend_shapes():
    from wcl.model_xlsr_aasist import AASISTBackend
    backend = AASISTBackend(in_dim=64, hidden=32, n_layers=2, n_heads=2)
    x = torch.randn(3, 41, 64)
    logits, h = backend(x)
    assert logits.shape == (3, 2)
    assert h.shape == (3, 41, 32)


def test_xlsr_aasist_full_model_and_wcl_loss():
    """Full backbone: logits + measure grid [B,C,T,F], then the WCL loss."""
    from wcl.model_xlsr_aasist import WCLXLSRAASIST
    model = WCLXLSRAASIST(
        frontend={"pretrained": False, "freeze": True,
                  "config_overrides": _SMALL_XLSR},
        backend={"hidden": 32, "n_layers": 2, "n_heads": 2},
        n_subbands=16,
    )
    wav = torch.randn(2, 2048)
    logits, feat = model(wav)
    assert logits.shape == (2, 2)
    assert feat.ndim == 4 and feat.shape[0] == 2
    # default measure_from="backend": width = backend hidden, not frontend d
    assert feat.shape[1] == model.measure_dim // 16 and feat.shape[3] == 16
    labels = torch.tensor([1, 0])
    loss, comp = wcl_loss(logits, logits, labels, feat, feat * 0.9,
                          lam=0.3, eps=0.1, return_components=True)
    assert torch.isfinite(loss) and loss.item() > 0
    assert comp["w_t"] >= 0 and comp["w_f"] >= 0


def test_build_model_arch_switch():
    """config arch=xlsr_aasist routes to the TFCL backbone."""
    from wcl.model_xlsr_aasist import WCLXLSRAASIST
    cfg = {"model": {"arch": "xlsr_aasist", "n_subbands": 16,
                     "frontend": {"pretrained": False, "freeze": True,
                                  "config_overrides": _SMALL_XLSR},
                     "backend": {"hidden": 32, "n_layers": 2, "n_heads": 2}}}
    model = build_model(cfg)
    assert isinstance(model, WCLXLSRAASIST)
    cfg_cnn = {"model": {"base_channels": (8, 16, 32, 64)}}
    assert build_model(cfg_cnn).__class__.__name__ == "WCLModel"


def test_waveform_pad_crop_normalize():
    from wcl.features import pad_or_crop_wave, normalize_wave
    x = np.random.randn(1000).astype(np.float32)
    padded = pad_or_crop_wave(x, 2048)
    assert padded.shape == (2048,) and np.all(padded[1000:] == 0)
    cropped = pad_or_crop_wave(x, 600)
    assert cropped.shape == (600,)
    z = normalize_wave(x)
    assert abs(z.mean()) < 1e-5 and abs(z.std() - 1.0) < 1e-3


def test_resolve_lambda_ratio():
    from wcl.losses import resolve_lambda
    # abs mode ignores alpha and returns the configured weight
    assert abs(resolve_lambda("abs", 0.3, 0.5, 1.0, 2.0) - 0.3) < 1e-12
    # ratio mode: lam_eff = alpha * CE_ref / W_ref  (scale-invariant in features)
    assert abs(resolve_lambda("ratio", 0.0, 0.3, 1.0, 2.0) - 0.15) < 1e-12
    # degenerate reference (no transport gap) falls back to alpha, stays finite
    assert abs(resolve_lambda("ratio", 0.0, 0.3, 1.0, 0.0) - 0.3) < 1e-12





# ------------------------------------------------------------------ geometry v2
def test_vad_truncate_shortens_and_shifts():
    """Truncating VAD must shorten the clip and move the content earlier.

    This is the property the first implementation lacked: zeroing silence
    in place keeps len(y) == len(x) and leaves the support where it was.
    """
    from wcl.afe import vad_truncate

    sr = 16000
    lead = int(0.5 * sr)                     # 0.5 s silence
    tone = np.sin(2 * np.pi * 300 * np.arange(int(1.0 * sr)) / sr).astype(np.float32)
    x = np.concatenate([np.zeros(lead, dtype=np.float32), tone,
                        np.zeros(int(0.3 * sr), dtype=np.float32)])

    y = vad_truncate(x, sr, quantile=0.25, pad=2, mode="span")

    # 1) the length actually shrinks (no in-place zeroing)
    assert len(y) < len(x) * 0.95
    # 2) the long leading silence is gone: only the pad margin (~30 ms) survives
    run = int(np.argmax(np.abs(y) > 1e-3))
    assert run < int(0.12 * sr)
    # 3) the truncation is a shift, not a deletion: content survives intact
    #    (align at the tone onset, which the truncation moved earlier)
    k = int(np.argmax(np.abs(y) > 1e-3))
    seg = y[k: k + 2000]
    assert len(seg) == 2000
    assert np.corrcoef(seg, tone[:2000])[0, 1] > 0.95


def test_vad_zero_keeps_length_for_reproducibility():
    """The legacy in-place VAD must still reproduce the first run."""
    from wcl.afe import vad

    sr = 16000
    x = np.random.default_rng(0).standard_normal(sr).astype(np.float32)
    y = vad(x, sr, quantile=0.25, pad=3)
    assert len(y) == len(x)


def test_apply_stage_vad_dispatch_modes():
    sr = 16000
    x = np.zeros(sr, dtype=np.float32)
    x[int(0.3 * sr): int(0.8 * sr)] = np.sin(
        2 * np.pi * 200 * np.arange(int(0.5 * sr)) / sr)

    y_trunc = apply_stage("vad", x, sr, {"vad": {"mode": "truncate"}})
    y_zero = apply_stage("vad", x, sr, {"vad": {"mode": "zero"}})
    assert len(y_trunc) < len(x)
    assert len(y_zero) == len(x)


def test_ground_cost_physical_adds_displacement():
    """Identical features at different positions must still cost something."""
    from wcl.losses import ground_cost

    n = 7
    v = torch.ones(1, n, 4)                  # all frames identical
    c_euc = ground_cost(v, v, "euclidean", 0.0)
    c_phys = ground_cost(v, v, "physical", 1.0)

    assert torch.allclose(c_euc, torch.zeros_like(c_euc), atol=1e-6)
    # physical cost grows monotonically with |i - j| and vanishes on the diagonal
    diag = torch.diagonal(c_phys[0])
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)
    assert c_phys[0, 0, n - 1] > c_phys[0, 0, 1] > 0


def test_wcl_loss_geometric_component_inflates_cost():
    torch.manual_seed(1)
    B, C, T, F = 2, 8, 12, 6
    logits_c = torch.randn(B, 2)
    logits_d = torch.randn(B, 2)
    labels = torch.tensor([1, 0])
    feat_c = torch.randn(B, C, T, F, requires_grad=True)
    feat_d = torch.randn(B, C, T, F, requires_grad=True)

    loss_e, comp_e = wcl_loss(logits_c, logits_d, labels, feat_c, feat_d,
                              lam=1.0, eps=0.1, n_iter=10,
                              cost_mode="marginal", cost_normalize="none",
                              return_components=True)
    loss_p, comp_p = wcl_loss(logits_c, logits_d, labels, feat_c, feat_d,
                              lam=1.0, eps=0.1, n_iter=10,
                              cost_mode="marginal", lambda_t=1.0,
                              lambda_f=0.5, cost_normalize="none",
                              return_components=True)

    assert torch.isfinite(loss_p) and loss_p.requires_grad
    assert comp_p["cost_t"] > comp_e["cost_t"]      # geometric term inflates cost
    assert comp_p["cost_f"] > comp_e["cost_f"]
    assert comp_p["w_t"] > comp_e["w_t"]




# ------------------------------------------------------------------ v3 (gradient)
def _tiny_xlsr(measure_from="backend"):
    from wcl.model_xlsr_aasist import WCLXLSRAASIST
    return WCLXLSRAASIST(
        frontend={"pretrained": False, "freeze": True,
                  "config_overrides": dict(hidden_size=64, num_hidden_layers=2,
                                           num_attention_heads=2,
                                           intermediate_size=128,
                                           conv_dim=(32, 32), conv_stride=(10, 5),
                                           conv_kernel=(10, 3),
                                           num_conv_pos_embeddings=16,
                                           num_conv_pos_embedding_groups=2,
                                           feat_extract_norm="layer")},
        backend={"hidden": 32, "n_layers": 2, "n_heads": 2},
        n_subbands=8, measure_from=measure_from)


def _grad_diff(model, lam):
    torch.manual_seed(1)
    xc, xd = torch.randn(2, 2048), torch.randn(2, 2048)
    labels = torch.tensor([1, 0])
    lc, fc = model(xc)
    ld, fd = model(xd)
    loss = wcl_loss(lc, ld, labels, fc, fd, lam=lam, eps=0.5)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()
                      if p.grad is not None])


def test_measure_from_backend_receives_gradient():
    """The transport terms must reach every trainable parameter (the run-1 bug:
    with measure_from=frontend + frozen front end they reach none)."""
    m = _tiny_xlsr("backend")
    m.eval()                                   # no dropout => deterministic
    d = (_grad_diff(m, 0.0) - _grad_diff(m, 1.0)).abs().max().item()
    assert d > 1e-6, f"consistency term still detached from the parameters ({d:.2e})"


def test_measure_from_frontend_is_detached():
    """Regression guard: documents *why* the first run was pure cross-entropy."""
    m = _tiny_xlsr("frontend")
    m.eval()
    d = (_grad_diff(m, 0.0) - _grad_diff(m, 1.0)).abs().max().item()
    assert d == 0.0, f"frontend measures unexpectedly carry gradient ({d:.2e})"


def test_agc_phase_is_observable_with_fluctuation():
    """A gain *curve* survives the per-utterance standardisation; a constant
    gain does not (that is why the AEC+NS and AEC+NS+AGC columns of Table 1
    were identical)."""
    from wcl.afe import agc
    from wcl.features import normalize_wave

    rng = np.random.default_rng(0)
    x = (0.4 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)
         + 0.05 * rng.standard_normal(16000)).astype(np.float32)

    flat = apply_stage("agc", x, 16000, {"agc": {"fluctuation": 0.0}},
                       rng=np.random.default_rng(0))
    curved = apply_stage("agc", x, 16000, {"agc": {"fluctuation": 0.3}},
                         rng=np.random.default_rng(0))
    d_flat = np.abs(normalize_wave(x) - normalize_wave(flat)).max()
    d_curve = np.abs(normalize_wave(x) - normalize_wave(curved)).max()
    assert d_flat < 1e-4, "constant gain should be annihilated by normalisation"
    assert d_curve > 1e-2, f"gain curve not observable after normalisation ({d_curve:.2e})"


def test_randomize_jitters_parameters():
    """afe.randomize must produce different members of the distortion family."""
    x = _synthetic_speech(seed=3)
    cfg = {"randomize": True, "jitter": 0.4,
           "vad": {"mode": "truncate"}, "ns": {}, "aec": {}, "agc": {}}
    y1 = apply_cascade(x, 16000, cfg, seed=11)
    y2 = apply_cascade(x, 16000, cfg, seed=11)
    y3 = apply_cascade(x, 16000, cfg, seed=12)
    assert np.allclose(y1, y2, atol=1e-6), "same seed must be reproducible"
    assert not np.allclose(y1[:len(y3)], y3[:len(y1)], atol=1e-4), \
        "different seeds must give different distortions"


def test_cost_normalize_keeps_eps_meaningful():
    """After normalisation mean(C)=1, so mean(C)/eps stays order 1 and the
    plan remains a soft coupling instead of collapsing to a permutation."""
    from wcl.losses import ground_cost, sinkhorn_plan

    torch.manual_seed(0)
    a = torch.randn(1, 40, 8) * 5.0
    b = a + torch.randn(1, 40, 8) * 5.0
    c = ground_cost(a, b, "physical", 1.0, "mean")
    assert abs(c.mean().item() - 1.0) < 1e-5
    P = sinkhorn_plan(c.detach(), eps=0.1, n_iter=50)
    # a soft plan has many non-zero entries per row, not a single one
    assert (P > 1e-6).sum().item() > 4 * P.shape[1] * P.shape[2] // 10


# ---------------------------------------------------------- v4 (product space)
def _diag_grid(t=16, f=8, c=4, diag=True):
    x = torch.zeros(1, c, t, f)
    for i in range(min(t, f)):
        x[0, :, i, i if diag else f - 1 - i] = 1.0
    return x


def test_product_cost_is_blind_spot_free():
    """The two-marginal objective is exactly at its floor when a displacement
    preserves both profiles; the product-space transport is not.

    This is the reason the objective had to change: Theorem 2 bounds W_1 on the
    product space, so a loss made of two 1-D marginal transports can be zero
    while the quantity that is bounded is not.
    """
    from wcl.losses import ground_cost

    a, b = _diag_grid(), _diag_grid(diag=False)

    def w_marg(x, y):
        wt = wasserstein_1_entropic(
            ground_cost(x.mean(3).transpose(1, 2), y.mean(3).transpose(1, 2),
                        "physical", 1.0, "mean"), 0.5, 20)
        wf = wasserstein_1_entropic(
            ground_cost(x.mean(2).transpose(1, 2), y.mean(2).transpose(1, 2),
                        "physical", 0.5, "mean"), 0.5, 20)
        return (wt + wf).item()

    floor = w_marg(a, a)
    marginal_excess = w_marg(a, b) - floor
    prod_floor = wasserstein_1_entropic(
        _prod(a, a), 0.5, 20).item()
    product_excess = wasserstein_1_entropic(_prod(a, b), 0.5, 20).item() - prod_floor

    assert abs(marginal_excess) < 1e-6, (
        f"marginal objective should be at its floor, got {marginal_excess:.2e}")
    assert product_excess > 1e-3, (
        f"product-space transport must see the displacement, got {product_excess:.2e}")


def _prod(x, y, lam_t=1.0, lam_f=0.5):
    return product_cost(x, y, lam_t, lam_f, "mean")


def test_product_cost_translation_costs_geometry():
    """Shifting identical features along time must increase the cost."""
    v = torch.zeros(1, 4, 12, 6)
    v[0, :, 0, :] = 1.0
    w = torch.zeros_like(v)
    w[0, :, 3, :] = 1.0                       # same content, 3 frames later
    c0 = _prod(v, v, 1.0, 0.5).diagonal(dim1=-2, dim2=-1).max()
    c_shift = _prod(v, w, 1.0, 0.5)
    assert c0 < 1e-5, "self-transport must be free"
    # the cell that carries the mass must now pay for 3 frames of displacement
    assert c_shift[0].min() >= 0
    assert c_shift.mean() > 1e-3


def test_calibrate_geometry_tracks_feature_scale():
    """lam_t must scale with the representation, not be a magic constant."""
    torch.manual_seed(0)
    a = torch.randn(2, 8, 20, 4)
    b = a + torch.randn_like(a) * 0.3
    lt1, lf1, i1 = calibrate_geometry(a, b)
    lt2, lf2, i2 = calibrate_geometry(a * 10.0, b * 10.0)
    assert lt1 > 0 and lf1 >= 0
    # a 10x larger representation needs ~10x larger absolute weight
    assert 8.0 < lt2 / lt1 < 12.0, (lt1, lt2)
    assert i1["displacement_t"] > 0


def test_wcl_loss_product_mode_has_gradient():
    torch.manual_seed(0)
    feat = torch.randn(2, 8, 10, 4, requires_grad=True)
    logits = torch.zeros(2, 2)
    labels = torch.zeros(2, dtype=torch.long)
    loss, comp = wcl_loss(logits, logits, labels, feat, feat * 1.1,
                          lam=1.0, eps=0.5, cost_mode="product",
                          return_components=True)
    loss.backward()
    assert feat.grad is not None and feat.grad.abs().sum() > 0
    assert comp["w_prod"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
