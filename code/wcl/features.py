"""Feature extraction utilities — log-mel spectrograms for WCL training.

The log-mel spectrogram is the input to the 2D-CNN encoder. Consistent with the
WCL paper the *Wasserstein* consistency is computed on frame-level features in
the encoder output space (Section 5.1 of the paper), not on the raw spectrogram;
here we only prepare the network input.

Normalization is per-utterance (mean/std or min/max over the whole utterance),
which is the standard front end for spectrogram-based anti-spoofing models.
"""

from __future__ import annotations

import numpy as np
import librosa


def load_waveform(path: str, sr: int = 16000) -> np.ndarray:
    """Load an audio file, resample to `sr`, return mono float32 in [-1, 1]."""
    x, _ = librosa.load(path, sr=sr, mono=True)
    return x.astype(np.float32)


def log_mel(x: np.ndarray, sr: int, n_fft: int = 512, hop: int = 160,
            n_mels: int = 80, fmin: int = 0, fmax: int | None = 8000) -> np.ndarray:
    """Compute the log-mel spectrogram: shape [T, F].

    The returned array is NOT normalized (normalization is separate so that
    caching stays flexible).
    """
    mel = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window="hann", center=True, n_mels=n_mels, fmin=fmin, fmax=fmax,
        power=2.0,
    )
    mel = np.log(mel + 1e-5)
    return mel.T.astype(np.float32)  # [T, F]


def pad_or_crop(feat: np.ndarray, max_frames: int, center: bool = True,
                offset: int | None = None) -> np.ndarray:
    """Pad (zero) or crop `feat` to exactly `max_frames` frames along axis 0."""
    T, F = feat.shape
    if T >= max_frames:
        if offset is None:
            offset = (T - max_frames) // 2 if center else 0
        return feat[offset: offset + max_frames]
    out = np.zeros((max_frames, F), dtype=np.float32)
    out[:T] = feat
    return out


def normalize(feat: np.ndarray, mode: str = "meanstd") -> np.ndarray:
    """Per-utterance normalization on the [T, F] array.

    meanstd : (x - mean) / std   (standard for anti-spoofing CNNs)
    minmax  : (x - min) / (max - min + 1e-8)  ->  [0, 1] (the non-negative
              construction of the paper's measure view)
    """
    if mode == "meanstd":
        mean, std = feat.mean(), feat.std() + 1e-8
        return (feat - mean) / std
    if mode == "minmax":
        lo, hi = feat.min(), feat.max()
        return (feat - lo) / (hi - lo + 1e-8)
    raise ValueError(f"unknown normalize mode: {mode}")


def pad_or_crop_wave(x: np.ndarray, n_samples: int,
                     center: bool = True) -> np.ndarray:
    """Pad (zero) or crop a waveform to exactly `n_samples` samples."""
    n = len(x)
    if n >= n_samples:
        offset = (n - n_samples) // 2 if center else 0
        return x[offset: offset + n_samples].astype(np.float32)
    out = np.zeros(n_samples, dtype=np.float32)
    out[:n] = x
    return out


def normalize_wave(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean / unit-variance waveform normalization.

    This mirrors the `do_normalize=True` behaviour of the Wav2Vec2 feature
    extractor used with XLS-R: the raw waveform is standardized before it
    enters the convolutional feature encoder.
    """
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + eps)


def extract_waveform(path: str, sr: int, n_samples: int,
                     normalize_mode: str = "meanstd") -> np.ndarray:
    """Load wav -> pad/crop to `n_samples` -> standardize -> [N]."""
    x = load_waveform(path, sr)
    x = pad_or_crop_wave(x, n_samples)
    if normalize_mode != "none":
        x = normalize_wave(x)
    return x


def extract_logmel(path: str, sr: int, n_fft: int, hop: int, n_mels: int,
                   max_frames: int, normalize_mode: str = "meanstd") -> np.ndarray:
    """End-to-end: load wav -> log-mel [T,F] -> pad/crop -> normalize -> [T,F]."""
    x = load_waveform(path, sr)
    feat = log_mel(x, sr, n_fft, hop, n_mels)
    feat = pad_or_crop(feat, max_frames)
    feat = normalize(feat, normalize_mode)
    return feat
