"""Wasserstein consistency loss — log-domain Sinkhorn implementation.

Default objective (cost_mode="product", paper Section 5.1):

    L_WCL = L_ce^c + L_ce^d + lambda * W_{1,eps}(mu_c, mu_d)

with mu_c / mu_d empirical measures on the T' x F' cells of the product ground
space, and a single transport problem of size (T'F') x (T'F'). The ground cost
of a cell pair is

    C[(t,f),(t',f')] = ||v_tf - v'_t'f'||_2
                       + lambda_t |t-t'|/(T'-1) + lambda_f |f-f'|/(F'-1)

where lambda_t is derived by calibrate_geometry() from the measured trim of the
front-end (lambda_t = beta_t * mean||v-v'|| * (T'-1)/Delta); lambda_f prices the
gap between channel groups, which is NOT a physical frequency (see the paper's
limitations).

Why a single product-space transport: the sum of two marginal transports
(cost_mode="marginal", the earlier form) is blind to displacements that keep
both 1-D profiles fixed (diagonal -> anti-diagonal), so it is not the quantity
Theorem 2 bounds. The marginal and euclidean modes are kept only as ablations.

Details aligned with the paper:
  * Sinkhorn in the log domain, k = 20 iterations (Section 5.2);
  * the cost matrix is divided by its own detached mean, so Sinkhorn sees
    mean(C)/eps in [2, 10] over the schedule instead of collapsing onto a
    permutation (Section 5.2);
  * Proposition 4 (scaling identity): lambda*W_{1,eps}(c) = W_{1,lambda*eps}
    (lambda*c) = lambda*eps*W_{1,1}(c/eps). eps sets the temperature and
    lambda*eps sets the strength; the ratio lambda/eps is NOT an invariant.
    Holding lambda fixed while annealing eps therefore WEAKENS the term by
    eps_init/eps_final, which the training loop records (lambda, eps,
    lambda*eps per epoch) rather than silently performing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["sinkhorn_plan", "wasserstein_1_entropic", "wcl_loss",
           "linear_cka", "TFCLConsistency", "resolve_lambda", "ground_cost"]


def _norm_coord(n_a: int, n_b: int, device) -> torch.Tensor:
    """Normalised absolute index distance |i - j| / (n - 1) in [0, 1]."""
    a = torch.arange(n_a, device=device, dtype=torch.float32)
    b = torch.arange(n_b, device=device, dtype=torch.float32)
    return (a[:, None] - b[None, :]).abs() / float(max(n_a - 1, 1))


def product_cost(feat_c: torch.Tensor, feat_d: torch.Tensor,
                 lambda_t: float = 1.0, lambda_f: float = 0.5,
                 normalize: str = "mean", stride: int = 1) -> torch.Tensor:
    """Ground cost on the product space of Section 3.1, flattened to cells.

    ``feat_c/feat_d`` are [B, C, T, F] grids; the measure lives on the T*F
    cells, each cell carrying its C-dim representation.  The cost is

        C[(t,f), (t',f')] = ||v_{t,f} - v'_{t',f'}||
                            + lam_t |t - t'| / (T - 1)
                            + lam_f |f - f'| / (F - 1),

    i.e. exactly the sum metric of the paper, without the two marginal
    projections.  This matters: pooling to the time axis first gives a
    transport between two 1-D histograms, and two measures can have identical
    temporal *and* spectral profiles while their joint measure is displaced
    (a diagonal blob and an anti-diagonal blob), in which case the marginal
    loss is exactly at its floor while the product-space distance is not.
    Theorem 2 bounds the product-space distance, so this is the object the
    objective has to minimise if the bound is to apply to it.

    ``stride`` pools time by average (kernel = stride) before the transport,
    which divides the cost-matrix size by stride^2.  Use it if the 20-iteration
    Sinkhorn on T*F cells becomes a bottleneck; state the value used.
    """
    import torch.nn.functional as Fn

    s = max(int(stride), 1)
    if s > 1:
        feat_c = Fn.avg_pool2d(feat_c, kernel_size=(s, 1), stride=(s, 1))
        feat_d = Fn.avg_pool2d(feat_d, kernel_size=(s, 1), stride=(s, 1))
    b, c, t, f = feat_c.shape
    b2, c2, t2, f2 = feat_d.shape
    cells_c = feat_c.permute(0, 2, 3, 1).reshape(b, t * f, c)
    cells_d = feat_d.permute(0, 2, 3, 1).reshape(b2, t2 * f2, c2)

    dev = cells_c.device
    ct_c = torch.arange(t, device=dev, dtype=torch.float32).repeat_interleave(f)
    cf_c = torch.arange(f, device=dev, dtype=torch.float32).repeat(t)
    ct_d = torch.arange(t2, device=dev, dtype=torch.float32).repeat_interleave(f2)
    cf_d = torch.arange(f2, device=dev, dtype=torch.float32).repeat(t2)

    d = torch.cdist(cells_c, cells_d, p=2)
    geo_t = (ct_c[:, None] - ct_d[None, :]).abs() / float(max(t - 1, 1))
    geo_f = (cf_c[:, None] - cf_d[None, :]).abs() / float(max(f - 1, 1))
    d = d + float(lambda_t) * geo_t + float(lambda_f) * geo_f

    if str(normalize).lower() in ("mean", "avg"):
        d = d / d.detach().mean().clamp_min(1e-6)
    return d


def calibrate_geometry(feat_c: torch.Tensor, feat_d: torch.Tensor,
                       beta_t: float = 1.0, beta_f: float = 1.0,
                       stride: int = 1, shift_cells: float | None = None,
                       band_cells: float = 1.0) -> tuple[float, float, dict]:
    """Set (lam_t, lam_f) so that a *typical distortion* costs beta feature gaps.

    The paper's default ``lambda_t = 1`` means "crossing the whole span costs
    one feature distance".  With T = 200 that is 0.005 per frame, while a VAD
    trim of 80-200 ms is 4-10 frames: the geometry contributes 0.02-0.05 to a
    cost whose feature part is O(1), so the optimal plan is once again decided
    by feature similarity alone and the |i - j| term only adds a constant.
    That is the same failure mode as the first run, one order of magnitude
    smaller, and it is why lam_t has to be derived rather than guessed.

    The calibration prices the displacement that the AFE actually applies.  If
    the cascade shifts the support by ``shift_cells`` grid cells on average
    (measured by ``scripts/check_afe_geometry.py``, and implied by
    ``afe.vad.extra_trim_s``), set

        lam_t = beta_t * mean_feature_cost * (T_cells - 1) / shift_cells,

    so that this typical shift costs ``beta_t`` times a mean feature gap.
    ``beta_t = 1`` therefore reads as "the geometry matters as much as the
    appearance", ``beta_t = 0`` recovers a purely representational cost.  This
    is Remark 1 one level down: the absolute weight carries the units of the
    representation, the ratio beta_t does not.

    ``band_cells`` plays the same role on the second axis; there is no physical
    band displacement, so a single band is the reference by default.

    Returns (lam_t, lam_f, diagnostics).
    """
    s = max(int(stride), 1)
    if s > 1:
        import torch.nn.functional as Fn
        feat_c = Fn.avg_pool2d(feat_c, kernel_size=(s, 1), stride=(s, 1))
        feat_d = Fn.avg_pool2d(feat_d, kernel_size=(s, 1), stride=(s, 1))
    with torch.no_grad():
        b, c, t, f = feat_c.shape
        b2, c2, t2, f2 = feat_d.shape
        cc = feat_c.permute(0, 2, 3, 1).reshape(b, t * f, c)
        cd = feat_d.permute(0, 2, 3, 1).reshape(b2, t2 * f2, c2)
        m_feat = float(torch.cdist(cc, cd, p=2).mean().item())

        dev = cc.device
        ct_c = torch.arange(t, device=dev, dtype=torch.float32).repeat_interleave(f)
        cf_c = torch.arange(f, device=dev, dtype=torch.float32).repeat(t)
        ct_d = torch.arange(t2, device=dev, dtype=torch.float32).repeat_interleave(f2)
        cf_d = torch.arange(f2, device=dev, dtype=torch.float32).repeat(t2)
        geo_t = (ct_c[:, None] - ct_d[None, :]).abs() / float(max(t - 1, 1))
        geo_f = (cf_c[:, None] - cf_d[None, :]).abs() / float(max(f - 1, 1))
        P = sinkhorn_plan(torch.cdist(cc, cd, p=2) / max(m_feat, 1e-6), 0.5, 10)
        d_t = float((P * geo_t.unsqueeze(0)).sum(dim=(1, 2)).mean().item())
        d_f = float((P * geo_f.unsqueeze(0)).sum(dim=(1, 2)).mean().item())

    D_t = float(shift_cells) if shift_cells else max(1.0, 0.02 * (t - 1))
    D_f = max(float(band_cells), 1e-3)
    lam_t = beta_t * m_feat * max(t - 1, 1) / max(D_t, 1e-6)
    lam_f = beta_f * m_feat * max(f - 1, 1) / D_f
    info = {"mean_feature_cost": m_feat, "shift_cells": D_t,
            "band_cells": D_f, "displacement_t": d_t, "displacement_f": d_f,
            "lambda_t": lam_t, "lambda_f": lam_f}
    return float(lam_t), float(lam_f), info


def ground_cost(va: torch.Tensor, vb: torch.Tensor,
                mode: str = "euclidean", lam_geo: float = 0.0,
                normalize: str = "none") -> torch.Tensor:
    """Ground cost between two sets of points on the time-frequency plane.

    ``mode="euclidean"`` is the cost used by the first GPU run: the Euclidean
    distance between frame (or band) feature vectors, with no reference to
    where the two points sit on the plane.  Under that cost the cheapest way to
    transport a region of zeroed silence is to spread it over any frames whose
    features happen to be close, whatever their position, so the loss carries no
    incentive to preserve the temporal support.

    ``mode="physical"`` adds the geometric term that Theorem 2 assumes: the
    normalised index displacement multiplied by a coefficient,

        C[i, j] = ||v_i - w_j||_2 + lam_geo * |i - j| / (n - 1).

    The displacement is normalised to [0, 1] so that ``lam_geo`` reads in units
    of feature distance: ``lam_geo = 1`` means "moving mass across the whole
    span costs as much as the largest feature distance observed".  This restores
    the premise of the bound - unit transport cost = physical displacement -
    and makes the plan itself carry the temporal geometry.

    ``normalize="mean"`` divides the whole matrix by its mean (detached, so it
    is a pure rescaling and contributes no gradient).  This is what makes
    ``eps`` meaningful: Sinkhorn is invariant to the *shape* of C, but the
    ratio mean(C)/eps decides whether the plan is a soft coupling or a hard
    assignment.  With raw feature distances (mean ~14 for XLS-R frames) and
    eps in [0.1, 0.01] that ratio sits at 140-1400, i.e. the plan collapses to
    a permutation and the eps-annealing of Prop. 4 stops acting as a
    temperature.  After normalisation, ``eps`` reads as "a fraction of the
    typical ground cost" and 0.5 -> 0.1 keeps mean(C)/eps in [2, 10].
    """
    d = torch.cdist(va, vb, p=2)
    if str(mode).lower() in ("physical", "geo", "time"):
        d = d + lam_geo * _norm_coord(va.shape[1], vb.shape[1], va.device)
    if str(normalize).lower() in ("mean", "avg"):
        scale = d.detach().mean().clamp_min(1e-6)
        d = d / scale
    return d


def sinkhorn_plan(cost: torch.Tensor, eps: float, n_iter: int = 20,
                  tol: float = 1e-6, n_polish: int = 0) -> torch.Tensor:
    """Log-domain Sinkhorn scaling with uniform marginals.

    Args:
        cost: [B, n, m] pairwise cost (e.g. Euclidean distances).
        eps:  entropic regularization strength (> 0).
        n_iter: number of alternating row/column normalizations.
        tol:  marginal tolerance used by the optional polish pass.
        n_polish: extra alternating passes, run until both marginals are within
            ``tol`` (capped at ``n_polish``).  Default 0: the plan carries the
            finite-k residual symmetrically, as a Sinkhorn plan should.
    Returns
    -------
        plan P [B, n, m].  Row sums are 1/n and column sums are 1/m up to the
        finite-k residual, which is *the same on both sides* and is reported by
        ``marginal_residual``.  Note that a finite number of scaling iterations
        cannot make both marginals exact at once; an earlier implementation
        forced the column sums at the very end, which silently broke the row
        sums it had just converged -- do not reintroduce that.
    """
    assert eps > 0
    B, n, m = cost.shape
    dev = cost.device
    logK = -cost / eps                                    # [B, n, m]
    log_r = -torch.log(torch.tensor(float(n), device=dev))  # uniform 1/n
    log_c = -torch.log(torch.tensor(float(m), device=dev))  # uniform 1/m
    log_u = torch.zeros(B, n, device=dev)
    log_v = torch.zeros(B, m, device=dev)
    for _ in range(n_iter):
        # column normalization: sum_i P_ij = 1/m
        log_v = log_c - torch.logsumexp(logK + log_u.unsqueeze(-1), dim=1)
        # row normalization: sum_j P_ij = 1/n
        log_u = log_r - torch.logsumexp(logK + log_v.unsqueeze(-2), dim=2)
    for _ in range(max(0, int(n_polish))):
        log_P = logK + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        P = torch.exp(log_P)
        dev = max(float((P.sum(dim=-1) - 1.0 / n).abs().max()),
                  float((P.sum(dim=-2) - 1.0 / m).abs().max()))
        if dev <= tol:
            return P
        log_v = log_c - torch.logsumexp(logK + log_u.unsqueeze(-1), dim=1)
        log_u = log_r - torch.logsumexp(logK + log_v.unsqueeze(-2), dim=2)
    log_P = logK + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return torch.exp(log_P)


def marginal_residual(plan: torch.Tensor) -> tuple[float, float]:
    """Worst row / column deviation from the uniform marginals (for logging)."""
    n, m = plan.shape[-2], plan.shape[-1]
    return (float((plan.sum(dim=-1) - 1.0 / n).abs().max()),
            float((plan.sum(dim=-2) - 1.0 / m).abs().max()))


def wasserstein_1_entropic(cost: torch.Tensor, eps: float, n_iter: int = 20,
                           return_plan: bool = False):
    """Entropy-regularized 1-Wasserstein value W_{1,eps}(mu, nu).

    W_{1,eps} = <P, C> + eps * KL(P || mu x nu)   (uniform marginals)

    Returns a tensor of shape [B] with one value per batch element (or the
    plan when return_plan=True).
    """
    P = sinkhorn_plan(cost, eps, n_iter)
    transport = (P * cost).sum(dim=(1, 2))
    B, n, m = cost.shape
    ent = torch.sum(P * torch.log(P.clamp_min(1e-12)), dim=(1, 2))
    kl = ent + torch.log(torch.tensor(float(n * m), device=cost.device))
    val = transport + eps * kl
    if return_plan:
        return val, P
    return val


def resolve_lambda(mode: str, lam: float, alpha: float,
                   ce_ref: float, cons_ref: float) -> float:
    """Turn the dimensionless consistency ratio into an absolute weight.

    Remark 1 of the paper: the absolute weight carries the units of the
    problem (nats per unit transport cost), so a hand-tuned constant does
    not transfer across backbones. ``mode="ratio"`` therefore sets

        lam_eff = alpha * CE_ref / W_ref,

    where CE_ref and W_ref are the classification loss and the transport
    term measured at initialisation. ``alpha`` then reads as "the fraction
    of the classification loss that the consistency term contributes at
    initialisation" and is invariant to feature scale.

    Args:
        mode: "abs" (use lam directly) or "ratio" (calibrate from alpha).
        lam: absolute weight, used only when mode == "abs".
        alpha: dimensionless consistency ratio.
        ce_ref: reference classification loss (CE_c + CE_d).
        cons_ref: reference consistency term (W_t + W_f).

    Returns:
        The effective absolute weight (float).
    """
    if str(mode).lower() != "ratio":
        return float(lam)
    if cons_ref <= 0 or not math.isfinite(cons_ref):
        # Degenerate reference (identical views): fall back to alpha itself,
        # which keeps the training run finite instead of dividing by zero.
        return float(alpha)
    return float(alpha * ce_ref / cons_ref)


def wcl_loss(logits_c: torch.Tensor, logits_d: torch.Tensor,
             labels: torch.Tensor, feat_c: torch.Tensor, feat_d: torch.Tensor,
             lam: float = 1.0, eps: float = 0.1, n_iter: int = 20,
             use_temporal: bool = True, use_frequency: bool = True,
             cost_mode: str = "product", lambda_t: float = 0.0,
             lambda_f: float = 0.0, cost_normalize: str = "mean",
             grid_stride: int = 1, return_components: bool = False):
    """Full WCL objective.

    Args:
        logits_c / logits_d: classification logits of clean / distorted views.
        feat_c / feat_d: feature maps [B, C, T', F'].
        lam: consistency weight (single remaining hyperparameter, Prop. 4).
        eps: entropic regularization at this training step (annealed by caller).
        cost_mode:
            ``"product"`` (default) - one transport on the T'*F' grid with the
                sum metric of Section 3.1.  This is the object Theorem 2
                bounds, so the bound applies to the loss that is minimised.
            ``"marginal"`` - W_t + W_f, i.e. two 1-D transports after pooling
                (the formulation of Section 5.1, and the closest analogue of
                the two-term design it replaces).  Kept as an ablation: it is
                blind to a displacement that preserves both profiles.
            ``"euclidean"`` - feature distance only, no geometric term; this is
                what run 1 used.
        lambda_t / lambda_f: geometric coefficients.  With ``product`` they are
            the two entries of the sum metric; see ``calibrate_geometry`` for
            how to size them against the representation scale.
        cost_normalize: "mean" divides the cost by its own mean (detached) so
            that eps is a fraction of the typical cost; "none" reproduces the
            first run, where mean(C)/eps was O(1e3) and the plan was a
            permutation.
        use_temporal / use_frequency: only consulted by ``cost_mode="marginal"``
            and ``"euclidean"``, where they select which of the two 1-D terms
            is computed. In ``product`` mode they are ignored on purpose: the
            geometric content there is carried by lambda_t / lambda_f, and a
            boolean switch would let a caller silently drop one axis of the
            sum metric while the cost matrix still describes both.
        grid_stride: time-pooling applied before the product-space transport.
    Returns:
        loss (scalar); optionally (loss, components dict).
    """
    ce_c = F.cross_entropy(logits_c, labels)
    ce_d = F.cross_entropy(logits_d, labels)

    comp = {"ce_c": ce_c.item(), "ce_d": ce_d.item(), "cons": 0.0,
            "w_t": 0.0, "w_f": 0.0, "w_prod": 0.0,
            "cost_t": 0.0, "cost_f": 0.0, "cost_prod": 0.0}

    mode = str(cost_mode).lower()
    cons = 0.0

    if mode in ("product", "prod", "joint"):
        cost = product_cost(feat_c, feat_d, lambda_t, lambda_f,
                            cost_normalize, grid_stride)
        w = wasserstein_1_entropic(cost, eps, n_iter)
        cons = cons + w.mean()
        comp["w_prod"] = float(w.mean().item())
        comp["cost_prod"] = float(cost.mean().item())
    else:
        if use_temporal:
            # temporal measures: pool over frequency -> [B, T', C]
            t_c = feat_c.mean(dim=3).transpose(1, 2)
            t_d = feat_d.mean(dim=3).transpose(1, 2)
            cost_t = ground_cost(t_c, t_d, "physical" if mode == "marginal"
                                 else "euclidean", lambda_t, cost_normalize)
            w_t = wasserstein_1_entropic(cost_t, eps, n_iter)
            cons = cons + w_t.mean()
            comp["w_t"] = float(w_t.mean().item())
            comp["cost_t"] = float(cost_t.mean().item())

        if use_frequency:
            # frequency measures: pool over time -> [B, F', C]
            f_c = feat_c.mean(dim=2).transpose(1, 2)
            f_d = feat_d.mean(dim=2).transpose(1, 2)
            cost_f = ground_cost(f_c, f_d, "physical" if mode == "marginal"
                                 else "euclidean", lambda_f, cost_normalize)
            w_f = wasserstein_1_entropic(cost_f, eps, n_iter)
            cons = cons + w_f.mean()
            comp["w_f"] = float(w_f.mean().item())
            comp["cost_f"] = float(cost_f.mean().item())

    comp["cons"] = float(cons.item()) if torch.is_tensor(cons) else float(cons)
    loss = ce_c + ce_d + lam * cons
    if return_components:
        return loss, comp
    return loss


def annealed_eps(eps_init: float, eps_final: float, epoch: int,
                 max_epochs: int) -> float:
    """Linear annealing of the entropic parameter (paper Section 5.4)."""
    if max_epochs <= 1:
        return float(eps_init)
    return float(eps_init + (eps_final - eps_init) * epoch / (max_epochs - 1))


# ---------------------------------------------------------------------------
# TFCL-style baseline consistency (for a controlled comparison)
# ---------------------------------------------------------------------------
def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Linear centered kernel alignment between two [n, d] representations.

    CKA = ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F), computed over the batch
    dimension n (the same convention used by the reference design).
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    num = (x.t() @ y).pow(2).sum()
    den = (x.t() @ x).pow(2).sum().sqrt() * (y.t() @ y).pow(2).sum().sqrt()
    return num / (den + eps)


class TFCLConsistency(nn.Module):
    """Cosine cross-attention (time) + linear CKA (frequency) consistency.

    Reproduces the baseline consistency objective so that WCL and this design
    can be compared under one backbone, one data pipeline, and one schedule.
    Input features are the same [B, C, T, F'] grids used by WCL, so both
    objectives see exactly the same tensor.

    ``cka_dim`` decides what the CKA is estimated over:

      * ``"batch"`` - the reference convention: one CKA per step, computed on
        the [B, C*F'] matrix.  With B = 8 the centred matrices have rank <= 7
        and Cauchy-Schwarz is nearly an equality, so for two *independent*
        random matrices the estimator returns ~0.89 and the loss is a constant
        with a tiny gradient.  Faithful to the reference, but a weak arm.
      * ``"cell"`` (default) - one CKA per sample over the T'*F' cells, then
        averaged.  n = T'*F' = 800 rows instead of 8, so the estimator is not
        rank-starved and the arm gets a gradient to compete with.  This is the
        stronger, defensible version; report which one produced a number.
    """

    def __init__(self, dim: int, dropout: float = 0.0, cka_dim: str = "cell"):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.cka_dim = str(cka_dim).lower()

    @staticmethod
    def _frames(feat: torch.Tensor) -> torch.Tensor:
        # [B, C, T, F'] -> [B, T, C*F']  (frame-level representation)
        b, c, t, f = feat.shape
        return feat.permute(0, 2, 1, 3).reshape(b, t, c * f)

    def _one_way(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        q = self.q(src)
        k = self.k(tgt)
        v = self.v(tgt)
        out = F.scaled_dot_product_attention(q, k, v)      # [B, T, D]
        cos = F.cosine_similarity(self.drop(out), src, dim=-1)
        return 1.0 - cos.mean()

    def _frequency_term(self, feat_c: torch.Tensor,
                        feat_d: torch.Tensor) -> torch.Tensor:
        if self.cka_dim == "batch":
            xc = feat_c.mean(dim=2).reshape(feat_c.shape[0], -1)
            xd = feat_d.mean(dim=2).reshape(feat_d.shape[0], -1)
            return 1.0 - linear_cka(xc, xd)
        b, c, t, f = feat_c.shape
        xc = feat_c.permute(0, 2, 3, 1).reshape(b, t * f, c)
        xd = feat_d.permute(0, 2, 3, 1).reshape(b, t * f, c)
        vals = [linear_cka(xc[i], xd[i]) for i in range(b)]
        return 1.0 - torch.stack(vals).mean()

    def forward(self, feat_c: torch.Tensor, feat_d: torch.Tensor):
        fc = self._frames(feat_c)
        fd = self._frames(feat_d)

        l_cd = self._one_way(fc, fd)
        l_dc = self._one_way(fd, fc)
        l_t = 0.5 * (l_cd + l_dc)

        l_f = self._frequency_term(feat_c, feat_d)

        return l_t + l_f, {"w_t": float(l_t.item()), "w_f": float(l_f.item())}
