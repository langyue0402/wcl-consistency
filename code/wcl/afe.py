"""AFE (Acoustic Front-End) distortion pipeline — pure-Python/numpy simulation.

Implements a simplified, dependency-light simulation of the four-stage
front-end cascade used in the robustness literature (AEC -> NS -> AGC -> VAD).
The physical picture is the one formalized in the WCL paper: each stage is a
*structured mass operation* on the time-frequency plane:

  - AEC : injects a delayed, attenuated copy of the signal (residual echo),
          i.e. mass is duplicated along delayed paths;
  - NS  : spectral subtraction erodes mass in low-energy (noise-dominated)
          bins and reshapes the spectral envelope;
  - AGC : rescales the whole signal to a target RMS, i.e. global mass rescaling,
          optionally with slow gain fluctuation;
  - VAD : removes low-energy regions, i.e. it *truncates* the measure along
          time and shifts the temporal support of everything that follows.

The VAD default changed after the first GPU run.  The original implementation
zeroed silence frames *in place*, which leaves the signal length T unchanged and
therefore never produces the geometric operation the paper's Section 3 assumes
("truncating mass along the time axis and shifting the temporal support").  The
default is now a truncating VAD (`mode="truncate"`), so that the distortion
family contains a genuine truncation + support shift; the legacy in-place
behaviour is still reachable with `mode="zero"` for reproducing the first run.

This is NOT a bit-exact reproduction of the WebRTC components used in the
reference paper; it reproduces the *type* of distortion each stage applies,
which is what the WCL consistency objective is designed to be invariant to.

Every stage operates on float32 waveform numpy arrays and is deterministic
given the (optional) random seed, so results are reproducible.
"""

from __future__ import annotations

import numpy as np
import librosa

__all__ = ["apply_stage", "apply_cascade", "STAGES", "cascade_prefix",
           "vad", "vad_truncate"]


STAGES = ["aec", "ns", "agc", "vad"]


def _rng(seed: int | None):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# AEC — residual echo simulation
# --------------------------------------------------------------------------
def aec(x: np.ndarray, sr: int, delay_s: float = 0.08, echo_gain: float = 0.15,
        nonlinear: bool = True, rng=None) -> np.ndarray:
    """Simulate residual echo after acoustic echo cancellation.

    The near-end signal is superimposed with a delayed, attenuated copy of
    itself (the imperfectly cancelled echo) plus a mild non-linearity that
    models the harmonic distortion left by the AEC non-linear processor.
    """
    delay = int(round(delay_s * sr))
    if delay <= 0:
        return x.astype(np.float32)
    out = x.copy()
    out[delay:] += echo_gain * x[:-delay]
    if nonlinear:
        out = np.tanh(2.0 * out)  # soft clipping -> harmonic distortion
    return (out / (np.max(np.abs(out)) + 1e-8) * 0.9).astype(np.float32)


# --------------------------------------------------------------------------
# NS — spectral subtraction noise suppression
# --------------------------------------------------------------------------
def ns(x: np.ndarray, sr: int, n_fft: int = 512, hop: int = 160,
       alpha: float = 1.0, floor_ratio: float = 0.05) -> np.ndarray:
    """Single-channel spectral subtraction.

    The noise profile is estimated as the low percentile of the magnitude
    spectrum over time (a robust proxy for the noise floor). After subtraction
    the magnitude is floored to a fraction of the original to avoid musical
    noise; the clean phase is kept.
    """
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                     window="hann", center=True)
    mag, phase = np.abs(S), np.angle(S)
    noise_est = np.percentile(mag, 15, axis=1, keepdims=True)
    mag_clean = np.maximum(mag - alpha * noise_est, floor_ratio * mag)
    S_clean = mag_clean * np.exp(1j * phase)
    y = librosa.istft(S_clean, hop_length=hop, win_length=n_fft,
                      window="hann", center=True, length=len(x))
    return (y / (np.max(np.abs(y)) + 1e-8) * 0.9).astype(np.float32)


# --------------------------------------------------------------------------
# AGC — automatic gain control (global mass rescaling)
# --------------------------------------------------------------------------
def agc(x: np.ndarray, sr: int, target_rms: float = 0.02,
        fluctuation: float = 0.0, rng=None) -> np.ndarray:
    """Rescale to a target RMS (peak-normalize first, then apply gain).

    If fluctuation > 0, a slowly varying gain curve (1 period over the clip)
    is applied on top, modelling AGC's time-varying gain reallocation.
    """
    y = x / (np.max(np.abs(x)) + 1e-8)
    cur_rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    g = target_rms / cur_rms
    if fluctuation > 0 and rng is not None:
        n = len(y)
        t = np.linspace(0, 1, n)
        phase = rng.uniform(0, 2 * np.pi)
        g = g * (1.0 + fluctuation * np.sin(2 * np.pi * t + phase))
    return (y * g).astype(np.float32)


# --------------------------------------------------------------------------
# VAD — voice activity detection (mass truncation along time)
# --------------------------------------------------------------------------
def _frame_energy(x: np.ndarray, sr: int, hop: int,
                  win_s: float = 0.025) -> tuple[np.ndarray, int, int]:
    """Per-frame energy and the (win, hop) used to compute it."""
    win = int(win_s * sr)
    if len(x) < win:
        return np.array([float(np.mean(x ** 2))]), win, hop
    n_frames = 1 + (len(x) - win) // hop
    frames = np.stack([x[i * hop: i * hop + win] for i in range(n_frames)])
    return np.mean(frames ** 2, axis=1), win, hop


def _voiced_mask(energy: np.ndarray, quantile: float, pad: int) -> np.ndarray:
    thr = float(np.quantile(energy, quantile))
    voiced = energy > thr
    if pad > 0:
        kernel = np.ones(2 * pad + 1, dtype=bool)
        voiced = np.convolve(voiced, kernel, mode="same") > 0
    return voiced


def vad_truncate(x: np.ndarray, sr: int, hop: int = 160, quantile: float = 0.25,
                 pad: int = 3, mode: str = "span",
                 min_keep_s: float = 0.25,
                 extra_trim_s: float = 0.0,
                 extra_trim_jitter_s: float = 0.0, rng=None) -> np.ndarray:
    """Voice activity detection that *truncates* the time axis.

    The distortion the paper models is a truncation of the measure along time
    plus a shift of its support, not an in-place attenuation.  Two variants are
    provided:

      - ``mode="span"``   keep the contiguous span from the first voiced frame
                          to the last one (the behaviour of a phone-side VAD:
                          leading and trailing silence are cut off, so the
                          remaining content is shifted earlier in the clip);
      - ``mode="stitch"`` additionally drop interior non-voiced regions and
                          splice what remains (the timeline of everything after
                          a removed region is shifted, possibly several times).

    The returned waveform is shorter than the input whenever any silence is
    found; ``pad_or_crop_wave`` in the feature stage then re-pads it to the
    fixed model length, which is where the residual shift becomes visible to
    the network.  If nothing is voiced, or the result would fall below
    ``min_keep_s``, the input is returned unchanged so that the phase stays
    well defined.
    """
    energy, win, hop = _frame_energy(x, sr, hop)
    voiced = _voiced_mask(energy, quantile, pad)

    if mode == "span":
        idx = np.flatnonzero(voiced)
        if idx.size == 0:
            return x.astype(np.float32)
        start = max(0, int(idx[0]) * hop)
        end = min(len(x), int(idx[-1]) * hop + win)
        y = x[start:end]
    elif mode == "stitch":
        n_frames = voiced.shape[0]
        pieces = []
        for i in range(n_frames):
            if not voiced[i]:
                continue
            s = i * hop
            e = min(s + win, len(x))
            if e > s:
                pieces.append(x[s:e])
        if not pieces:
            return x.astype(np.float32)
        y = np.concatenate(pieces)
    else:
        raise ValueError(f"unknown vad truncation mode: {mode}")

    if len(y) < int(min_keep_s * sr):
        return x.astype(np.float32)
    # Optional extra head-trim: ASVspoof LA clips are cut tight and often carry
    # almost no leading/trailing silence, in which case an energy-based VAD
    # removes nothing and the geometric distortion of the paper never occurs.
    # Setting extra_trim_s > 0 guarantees a genuine support shift (content
    # moves earlier by that many seconds), which is the cheapest way to make
    # the distortion family contain a truncation even on tightly cut data.
    if extra_trim_s > 0:
        cut_s = float(extra_trim_s)
        if extra_trim_jitter_s > 0:
            cut_s += float(rng.uniform(0.0, extra_trim_jitter_s)) if rng is not None else 0.0
        cut = int(cut_s * sr)
        if len(y) - cut >= int(min_keep_s * sr):
            y = y[cut:]
    return y.astype(np.float32)


def vad(x: np.ndarray, sr: int, hop: int = 160, quantile: float = 0.25,
        pad: int = 3) -> np.ndarray:
    """Legacy in-place VAD: zero out low-energy frames, keep the length.

    This is the behaviour of the first GPU run (dev epoch 31, eval avgEER
    2.56%).  It is kept only so that the first run can be reproduced and so
    that the two VAD variants can be compared directly; the paper's geometry
    is not realised by this variant.
    """
    win = int(0.025 * sr)  # 25 ms
    frames = librosa.util.frame(x, frame_length=win, hop_length=hop)
    energy = np.mean(frames ** 2, axis=0)  # per-frame energy
    thr = np.quantile(energy, quantile)
    voiced = energy > thr
    # morphological dilation to keep voiced-neighbour frames
    if pad > 0:
        kernel = np.ones(2 * pad + 1, dtype=bool)
        voiced = np.convolve(voiced, kernel, mode="same") > 0
    y = x.copy()
    n_frames = voiced.shape[0]
    n_samples = (n_frames - 1) * hop + win
    n_samples = min(n_samples, len(x))
    start = (len(x) - n_samples) // 2
    seg = np.zeros((voiced.shape[0], win), dtype=np.float32)
    for i in range(voiced.shape[0]):
        s = start + i * hop
        e = s + win
        if e <= len(x):
            seg[i] = x[s:e] if voiced[i] else 0.0
    out = np.zeros_like(x)
    for i in range(voiced.shape[0]):
        s = start + i * hop
        e = min(s + win, len(out))
        if s < len(out):
            out[s:e] = seg[i][: e - s]
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
def _jit(v: float, rel: float, rng, lo: float | None = None,
         hi: float | None = None) -> float:
    """Multiplicative jitter of a stage parameter about its nominal value."""
    if rng is None or rel <= 0:
        return float(v)
    out = float(v) * (1.0 + float(rng.uniform(-rel, rel)))
    if lo is not None:
        out = max(out, float(lo))
    if hi is not None:
        out = min(out, float(hi))
    return out


def apply_stage(stage: str, x: np.ndarray, sr: int, cfg: dict,
                rng=None) -> np.ndarray:
    """Apply a single AFE stage to a waveform.

    cfg may carry per-stage parameters; values not present fall back to the
    defaults documented above.

    ``cfg["afe"]["randomize"] = True`` (or ``cfg["randomize"]``) turns every
    stage into a *family*: the nominal parameters are jittered by
    ``cfg["afe"]["jitter"]`` (default 0.4, i.e. +-40%) around their defaults,
    and the VAD energy quantile / AGC gain curve are redrawn per call.  This is
    the code path that the WCL theory assumes ("the distortion family G"); with
    ``randomize: False`` every utterance receives one fixed transformation.
    """
    p = (cfg or {}).get(stage, {}) or {}
    # accept both cfg["randomize"] and the afe-scoped cfg["afe"]["randomize"]
    root = cfg or {}
    afe_root = root.get("afe", root) if isinstance(root.get("afe", None), dict) else root
    rand = bool(root.get("randomize", afe_root.get("randomize", False)))
    rel = float(root.get("jitter", afe_root.get("jitter", 0.4))) if rand else 0.0
    j = (lambda v, lo=None, hi=None: _jit(v, rel, rng, lo, hi)) if rand else \
        (lambda v, lo=None, hi=None: float(v))

    if stage == "aec":
        gain = j(p.get("echo_gain", 0.15), 0.0, 0.9)
        return aec(x, sr,
                   delay_s=j(p.get("delay_s", 0.08), 0.0, 0.25),
                   echo_gain=gain,
                   nonlinear=p.get("nonlinear", True), rng=rng)
    if stage == "ns":
        return ns(x, sr,
                  n_fft=p.get("n_fft", 512), hop=p.get("hop", 160),
                  alpha=j(p.get("alpha", 1.0), 0.0, 3.0),
                  floor_ratio=j(p.get("floor_ratio", 0.05), 0.0, 0.5))
    if stage == "agc":
        flu = float(p.get("fluctuation", 0.0))
        if rand and flu <= 0:
            # randomize => the gain curve is part of the family
            flu = float(rng.uniform(0.15, 0.5)) if rng is not None else 0.3
        return agc(x, sr,
                   target_rms=j(p.get("target_rms", 0.02), 1e-3, 0.5),
                   fluctuation=flu, rng=rng)
    if stage == "vad":
        mode = str(p.get("mode", "truncate")).lower()
        if mode in ("truncate", "span", "stitch"):
            return vad_truncate(x, sr, hop=p.get("hop", 160),
                                quantile=j(p.get("quantile", 0.25), 0.05, 0.6),
                                pad=p.get("pad", 3),
                                mode=p.get("truncate_mode", "span")
                                if mode == "truncate" else mode,
                                min_keep_s=p.get("min_keep_s", 0.25),
                                extra_trim_s=float(p.get("extra_trim_s", 0.0)),
                                extra_trim_jitter_s=float(
                                    p.get("extra_trim_jitter_s", 0.0)),
                                rng=rng)
        if mode == "zero":                      # legacy in-place variant
            return vad(x, sr, hop=p.get("hop", 160),
                       quantile=j(p.get("quantile", 0.25), 0.05, 0.6),
                       pad=p.get("pad", 3))
        raise ValueError(f"unknown vad mode: {mode}")
    raise ValueError(f"unknown AFE stage: {stage}")


def apply_cascade(x: np.ndarray, sr: int, cfg: dict, seed: int | None = 0,
                  stages: list[str] | None = None) -> np.ndarray:
    """Apply the stage cascade in order (AEC -> NS -> AGC -> VAD by default).

    With cfg.afe.randomize = True the per-stage parameters are jittered around
    their defaults so the training distortion space is a *family* rather than a
    single fixed transformation — the setting the WCL theory is designed for.
    """
    stages = stages or STAGES
    rng = _rng(seed)
    y = x.astype(np.float32)
    for stage in stages:
        y = apply_stage(stage, y, sr, cfg, rng=rng)
    return y


def cascade_prefix(stages: list[str], upto: str) -> list[str]:
    """Return the stage list of the pipeline prefix ending at `upto`."""
    idx = stages.index(upto)
    return stages[: idx + 1]
