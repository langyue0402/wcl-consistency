"""Lightweight 2D-CNN encoder with a time-frequency-preserving output.

Architecture rationale (WCL paper, Section 5.1): the consistency terms are
Wasserstein distances between *frame-level empirical measures* of the encoder
output in time and frequency directions. The encoder therefore must NOT
globally pool before the consistency layer — it has to output a feature map
[B, C, T', F'] whose rows/columns are the measures we transport.

Design: 4 conv-blocks (Conv2d + BN + ReLU + MaxPool) with SE blocks,
keeping a 40 x 5 feature grid for a 640-frame / 80-mel input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel recalibration."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, use_se: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.se = SEBlock(cout) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.pool(self.conv(x)))


class WCLEncoder(nn.Module):
    """Feature-map encoder: spectrogram [B,1,T,F] -> features [B,C,T',F']."""

    def __init__(self, in_channels: int = 1, base_channels=(32, 64, 128, 256),
                 se_blocks=(True, True, True, True)):
        super().__init__()
        blocks = []
        cin = in_channels
        for cout, use_se in zip(base_channels, se_blocks):
            blocks.append(ConvBlock(cin, cout, use_se=use_se))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = cin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class WCLModel(nn.Module):
    """Encoder + linear classification head.

    forward(x) -> (logits [B,2], feat_map [B,C,T',F'])
    """

    def __init__(self, base_channels=(32, 64, 128, 256),
                 se_blocks=(True, True, True, True),
                 embed_dim: int = 128, dropout: float = 0.3,
                 num_classes: int = 2, in_frames: int = 640, in_mels: int = 80):
        super().__init__()
        self.encoder = WCLEncoder(1, base_channels, se_blocks)
        cin = self.encoder.out_channels
        # flattened width of the frame representation, used by the TFCL
        # baseline head: every ConvBlock halves both axes, so F' = in_mels/2^n
        n_pool = len(tuple(base_channels))
        self.out_dim = cin * max(int(in_mels) // (2 ** n_pool), 1)
        self.head = nn.Sequential(
            nn.Linear(cin, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)                       # [B,C,T',F']
        pooled = feat.mean(dim=(2, 3))               # global avg pool
        logits = self.head(pooled)
        return logits, feat


def build_model(cfg: dict) -> nn.Module:
    """Build the encoder according to ``model.arch``.

    arch = "cnn" (default)       -> lightweight 2D-CNN (spectrogram input)
    arch = "xlsr_aasist"         -> XLS-R front-end + AASIST back-end
                                    (TFCL's architecture; raw waveform input)

    ``model.measure_from`` selects which representation the WCL transport terms
    are computed on for the xlsr_aasist backbone; see
    ``wcl.model_xlsr_aasist.WCLXLSRAASIST``.  The CNN encoder is trainable
    end to end, so its feature map is a valid measure for either setting.
    """
    m = cfg["model"]
    if m.get("arch") == "xlsr_aasist":
        from .model_xlsr_aasist import WCLXLSRAASIST
        return WCLXLSRAASIST(
            frontend=m.get("frontend", {}),
            backend=m.get("backend", {}),
            n_subbands=m.get("n_subbands", 16),
            measure_from=m.get("measure_from", "frontend"),
        )
    return WCLModel(
        base_channels=tuple(m.get("base_channels", (32, 64, 128, 256))),
        se_blocks=tuple(m.get("se_blocks", (True, True, True, True))),
        embed_dim=m.get("embed_dim", 128),
        dropout=m.get("dropout", 0.3),
        in_frames=int(cfg.get("feature", {}).get("max_frames", 640)),
        in_mels=int(cfg.get("feature", {}).get("n_mels", 80)),
    )
