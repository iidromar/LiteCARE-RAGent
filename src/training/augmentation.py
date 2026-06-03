"""
Data Augmentation for Physiological Time-Series
================================================
Online augmentation applied during training to combat overfitting with small
datasets (~1300 windows per LOSO fold).

Transforms (all applied per-sample, not batch-level):
  1. GaussianNoise     — adds σ-scaled noise to each channel independently
  2. ChannelDropout    — zeros a random channel with probability p
  3. TimeWarp          — stretches/compresses time axis by ±alpha, then crops/pads
  4. WindowSlice       — takes a random sub-window and interpolates back to full size
  5. SignalFlip        — flips the time axis (temporal mirror)
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    """
    Wraps a (X, y) numpy pair with on-the-fly augmentation.

    Args:
        X:          shape [N, C, T]
        y:          shape [N]
        augment:    whether to apply augmentation (set False for val/test)
        noise_std:  Gaussian noise standard deviation (fraction of signal std)
        dropout_p:  probability of dropping a channel
        warp_alpha: max time-warp stretch ratio (e.g. 0.1 = ±10%)
        flip_p:     probability of time-axis flip
    """
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 augment:     bool  = True,
                 noise_std:   float = 0.05,
                 dropout_p:   float = 0.15,
                 warp_alpha:  float = 0.1,
                 flip_p:      float = 0.3):
        self.X          = X.astype(np.float32)
        self.y          = y
        self.augment    = augment
        self.noise_std  = noise_std
        self.dropout_p  = dropout_p
        self.warp_alpha = warp_alpha
        self.flip_p     = flip_p

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()   # [C, T]
        y = int(self.y[idx])

        if self.augment:
            x = self._apply(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    # ── Individual transforms ──────────────────────────────────────────────────

    def _apply(self, x: np.ndarray) -> np.ndarray:
        """Apply a random subset of augmentations."""
        if np.random.random() < 0.8:
            x = _gaussian_noise(x, self.noise_std)
        if np.random.random() < self.dropout_p:
            x = _channel_dropout(x)
        if np.random.random() < 0.5:
            x = _window_slice(x, crop_ratio=0.9)
        if np.random.random() < self.flip_p:
            x = x[:, ::-1].copy()
        return x


def _gaussian_noise(x: np.ndarray, std_frac: float) -> np.ndarray:
    """Add Gaussian noise scaled to each channel's standard deviation."""
    C, T = x.shape
    for c in range(C):
        sigma = x[c].std() * std_frac + 1e-8
        x[c] += np.random.normal(0, sigma, T).astype(np.float32)
    return x


def _channel_dropout(x: np.ndarray) -> np.ndarray:
    """Zero out one randomly selected channel."""
    c = np.random.randint(0, x.shape[0])
    x[c] = 0.0
    return x


def _window_slice(x: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    """
    Take a random sub-window (crop_ratio fraction of the full window),
    then linearly interpolate back to the original length.
    Simulates temporal scale variation without changing sequence length.
    """
    C, T = x.shape
    crop_len = int(T * crop_ratio)
    start    = np.random.randint(0, T - crop_len + 1)
    sliced   = x[:, start:start + crop_len]    # [C, crop_len]

    # Resize back to T using linear interpolation per channel
    indices_new = np.linspace(0, crop_len - 1, T)
    indices_old = np.arange(crop_len)
    out = np.zeros((C, T), dtype=np.float32)
    for c in range(C):
        out[c] = np.interp(indices_new, indices_old, sliced[c])
    return out
