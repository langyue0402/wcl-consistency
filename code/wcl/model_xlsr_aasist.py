"""XLS-R front-end + AASIST back-end (the architecture used by TFCL).

Rationale
---------
TFCL (Xue et al., ACM MM '26) uses:
  * **XLS-R** (`facebook/wav2vec2-xls-r-300m`) as the shared front-end that
    produces frame-level representations `F in R^{T x 1024}`;
  * the **AASIST** back-end (graph-attention classifier) on top of those
    frames;
  * Adam (lr 1e-4, wd 1e-4), 100 epochs, early stopping (patience 10),
    consistency weight lambda = 0.3.

To make a like-for-like comparison with TFCL's "Ours" row, this module
reproduces that backbone so that the *only* difference between the systems is
the consistency objective (TFCL: cosine + CKA; WCL: entropic Wasserstein).

Design notes / honest differences from the official code
---------------------------------------------------------
* The back-end is structurally equivalent to AASIST's HS-GAL stack
  (full-graph self-attention + FFN + residual), implemented with
  `nn.MultiheadAttention` and LayerNorm. The official AASIST uses
  `BatchNorm1d`; LayerNorm is used here because batches of variable-length
  frame sequences are more stable under LN (documented, not hidden).
* The consistency measure grid is obtained by reshaping the 1024-d frame
  representation into `F` sub-bands of `1024/F` channels. This is exactly the
  grouping TFCL uses for its frequency-domain CKA, so the two objectives see
  the same tensor and the comparison is fair.
* "Frequency" axis here means *channel sub-bands* of the self-supervised
  representation, not physical mel bins: XLS-R has no explicit frequency
  layout. This limitation is stated in the README.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class XLSRFrontend(nn.Module):
    """Wav2Vec2/XLS-R front-end returning frame-level hidden states.

    Args:
        model_name: HF hub id (default: the 300M XLS-R checkpoint).
        freeze: if True, the front-end is frozen (recommended: it keeps the
            optimizer state small enough for a single 24 GB GPU).
        layer_idx: which hidden-states layer to export (-1 = last).
        pretrained: set False to build a randomly initialised model (tests).
        config_overrides: kwargs for `Wav2Vec2Config` (tests use tiny values).
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-xls-r-300m",
                 freeze: bool = True, layer_idx: int = -1,
                 pretrained: bool = True, config_overrides: dict | None = None):
        super().__init__()
        try:
            from transformers import Wav2Vec2Config, Wav2Vec2Model
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "XLS-R front-end needs `transformers`. Install with:\n"
                "    pip install 'transformers>=4.30'"
            ) from exc

        if config_overrides:
            cfg = Wav2Vec2Config(**config_overrides)
            self.model = Wav2Vec2Model(cfg)
        elif pretrained:
            self.model = Wav2Vec2Model.from_pretrained(model_name)
        else:
            self.model = Wav2Vec2Model(Wav2Vec2Config())

        self.hidden_size = int(self.model.config.hidden_size)
        self.layer_idx = layer_idx
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, N] (or [B,1,N]) -> frame features [B, T, D].

        Note: we take ``last_hidden_state`` by default instead of indexing
        ``hidden_states[-1]`` — transformers >= 5 may prune the hidden-state
        tuple in some execution paths, which makes indexing brittle.
        """
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        if self.layer_idx == -1:
            return self.model(wav).last_hidden_state
        out = self.model(wav, output_hidden_states=True)
        return out.hidden_states[self.layer_idx]


class GraphAttentionLayer(nn.Module):
    """HS-GAL style block: full-graph self-attention + FFN + residuals."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1,
                 ffn_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ffn_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ffn_ratio), dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class AASISTBackend(nn.Module):
    """AASIST-style graph-attention classifier over frame features.

    forward(x [B,T,D_in]) -> (logits [B,2], frame_repr [B,T,D])
    """

    def __init__(self, in_dim: int = 1024, hidden: int = 128, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.out_dim = int(hidden)                # width of the frame output
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            GraphAttentionLayer(hidden, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor):
        h = self.proj(x)                          # [B,T,H]
        for layer in self.layers:
            h = layer(h)                          # [B,T,H]
        mean = h.mean(dim=1)                      # [B,H]
        mx = h.max(dim=1).values                  # [B,H]
        logits = self.pool_proj(torch.cat([mean, mx], dim=-1))
        return logits, h


class WCLXLSRAASIST(nn.Module):
    """XLS-R + AASIST backbone with a WCL-compatible feature grid.

    forward(wav [B,N] or [B,1,N]) -> (logits [B,2], feat_map [B,C,T,F])
    where feat_map is a frame representation reshaped into `n_subbands`
    channel sub-bands:  C = measure_dim / n_subbands.

    ``measure_from`` decides WHICH representation the transport terms are
    computed on - this is the difference between a working objective and a
    constant added to the loss:

      * ``"backend"`` (recommended, used by the configs): the AASIST frame
        output `hb [B,T,hidden]`.  It is trainable, and it is the tensor the
        classifier actually pools over, so ``W_t + W_f`` has a gradient path to
        every trainable parameter and Theorem 2's statement ("measures of the
        encoder output, h = l o f") refers to the representation being
        regularised.
      * ``"frontend"`` (legacy, first GPU run; must be requested explicitly):
        `h [B,T,1024]`.  With ``frontend.freeze: true`` no trainable parameter
        is upstream of this tensor and no downstream parameter consumes it, so
        the consistency term contributes **exactly zero gradient** - training
        reduces to dual-view cross-entropy while the logs still print a
        non-zero, slowly drifting W_t.  Kept only for reproducing run 1.
    """

    def __init__(self, frontend: dict | None = None, backend: dict | None = None,
                 n_subbands: int = 16, num_classes: int = 2,
                 measure_from: str = "backend"):
        super().__init__()
        self.frontend = XLSRFrontend(**(frontend or {}))
        self.measure_from = str(measure_from).lower()
        if self.measure_from not in ("frontend", "backend"):
            raise ValueError(f"unknown measure_from: {measure_from}")
        self.n_subbands = int(n_subbands)
        d = self.frontend.hidden_size
        self.backend = AASISTBackend(in_dim=d, num_classes=num_classes,
                                     **(backend or {}))
        measure_dim = self.backend.out_dim if self.measure_from == "backend" else d
        if measure_dim % self.n_subbands != 0:
            raise ValueError(
                f"measure width {measure_dim} not divisible by n_subbands "
                f"{self.n_subbands} (measure_from={self.measure_from}); "
                f"pick n_subbands in {[k for k in range(1, 33) if measure_dim % k == 0]}")
        self.measure_dim = int(measure_dim)
        self.sub_dim = self.measure_dim // self.n_subbands
        # flattened width of the measure grid, consumed by the TFCL baseline head
        self.out_dim = self.measure_dim

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:                       # [B,1,N] -> [B,N]
            x = x.squeeze(1)
        h = self.frontend(x)                   # [B,T,1024]  (frozen by default)
        logits, hb = self.backend(h)           # hb: [B,T,hidden]  (trainable)
        src = hb if self.measure_from == "backend" else h
        b, t, d = src.shape
        feat = src.reshape(b, t, self.n_subbands, self.sub_dim).permute(0, 3, 1, 2)
        # [B, C, T, F_sub]: rows = time, columns = channel sub-bands
        return logits, feat
