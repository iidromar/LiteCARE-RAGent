"""
Common preprocessing utilities shared across all datasets.
- Resampling to target_fs (32 Hz)
- Per-subject z-score normalization
- Sliding window segmentation
"""
from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly
from math import gcd


def _resample_channel(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Resample 1-D signal from orig_fs to target_fs using polyphase filter."""
    if orig_fs == target_fs:
        return signal.astype(np.float32)
    g = gcd(orig_fs, target_fs)
    up = target_fs // g
    down = orig_fs // g
    return resample_poly(signal, up, down).astype(np.float32)


def resample_signals(signals: dict[str, np.ndarray],
                     native_fs: dict[str, int],
                     target_fs: int = 32) -> dict[str, np.ndarray]:
    """
    Resample a dict of {channel_name: 1-D array} to target_fs.

    Args:
        signals:   dict mapping channel name → raw signal array
        native_fs: dict mapping channel name → native sampling rate (Hz)
        target_fs: desired output sampling rate

    Returns:
        dict of resampled signals (same keys, float32 arrays)
    """
    resampled = {}
    for name, sig in signals.items():
        resampled[name] = _resample_channel(sig, native_fs[name], target_fs)
    return resampled


def acc_magnitude(acc: np.ndarray) -> np.ndarray:
    """
    Convert 3-axis ACC (shape [N, 3] or [3, N]) to scalar magnitude.
    Returns 1-D array.
    """
    if acc.ndim == 2:
        if acc.shape[1] == 3:
            return np.sqrt((acc ** 2).sum(axis=1)).astype(np.float32)
        elif acc.shape[0] == 3:
            return np.sqrt((acc ** 2).sum(axis=0)).astype(np.float32)
    raise ValueError(f"Unexpected ACC shape: {acc.shape}")


def znorm_per_subject(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize over the whole recording (per-subject normalization)."""
    mean = signal.mean()
    std = signal.std()
    return ((signal - mean) / (std + eps)).astype(np.float32)


def sliding_window(X: np.ndarray,
                   y: np.ndarray,
                   window: int = 1920,
                   step: int = 960,
                   label_strategy: str = "majority_vote") -> tuple[np.ndarray, np.ndarray]:
    """
    Segment multivariate time series into fixed-size overlapping windows.

    Args:
        X:               shape [C, T] — C channels, T time steps
        y:               shape [T]   — sample-level labels
        window:          window length in samples (default 1920 = 60s @ 32Hz)
        step:            step size in samples (default 960 = 50% overlap)
        label_strategy:  'majority_vote' or 'first'

    Returns:
        X_wins: shape [N, C, window]
        y_wins: shape [N]
    """
    C, T = X.shape
    starts = range(0, T - window + 1, step)
    X_wins, y_wins = [], []

    for s in starts:
        x_w = X[:, s:s + window]
        y_w = y[s:s + window]

        # Skip windows that are entirely unlabeled (label == 0 means undefined in WESAD)
        valid = y_w[y_w != 0]
        if len(valid) == 0:
            continue

        if label_strategy == "majority_vote":
            label = int(np.bincount(valid.astype(int)).argmax())
        else:
            label = int(y_w[0])

        X_wins.append(x_w)
        y_wins.append(label)

    if len(X_wins) == 0:
        return np.empty((0, C, window), dtype=np.float32), np.empty(0, dtype=np.int64)

    return np.stack(X_wins).astype(np.float32), np.array(y_wins, dtype=np.int64)


def build_channel_matrix(signals: dict[str, np.ndarray],
                          channel_order: list[str]) -> np.ndarray:
    """
    Stack resampled channels into shape [C, T].

    Args:
        signals:       dict of resampled 1-D arrays
        channel_order: list of channel names in desired order

    Returns:
        np.ndarray of shape [len(channel_order), min_T]
    """
    arrays = [signals[ch] for ch in channel_order]
    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays])  # [C, T]
