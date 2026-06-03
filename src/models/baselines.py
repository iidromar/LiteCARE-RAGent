"""
Baseline Models
===============
Three baselines for comparison with Lite-TCN-SE:

1. Random Forest (RF)      — handcrafted statistical features, per Schmidt et al. 2018
2. BiLSTM                  — 2-layer bidirectional LSTM, per Can et al. 2019
3. DilatedCNN              — dilated 1-D CNN without SE blocks, per Motaman et al. 2025
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ── Handcrafted Feature Extraction for RF ────────────────────────────────────

def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """
    Extract per-channel statistical features from windowed data.

    Args:
        X: [N, C, T]  — N windows, C channels, T time steps

    Returns:
        features: [N, C * 8]
    """
    N, C, T = X.shape
    feats = []
    for i in range(N):
        win = X[i]   # [C, T]
        sample_feats = []
        for c in range(C):
            ch = win[c]
            sample_feats.extend([
                ch.mean(),
                ch.std(),
                ch.min(),
                ch.max(),
                np.percentile(ch, 25),
                np.percentile(ch, 75),
                np.median(ch),
                ch.max() - ch.min(),      # range
            ])
        feats.append(sample_feats)
    return np.array(feats, dtype=np.float32)   # [N, C*8]


def build_rf_pipeline(n_estimators: int = 200,
                      max_depth: int | None = None,
                      class_weight: str = "balanced") -> Pipeline:
    """Random Forest with z-score normalization."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42,
        )),
    ])


# ── Bidirectional LSTM with Temporal Attention ────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    2-layer Bidirectional LSTM with additive temporal attention + linear head.
    Matches the LSTM-Attention architecture cited in Can et al. (2019).
    Input: [B, C, T]  (same convention as TCN — transposed internally)
    """
    def __init__(self,
                 input_size:  int   = 4,
                 hidden_size: int   = 128,
                 num_layers:  int   = 2,
                 dropout:     float = 0.3,
                 num_classes: int   = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Additive (Bahdanau-style) temporal attention
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size * 2, num_classes)   # ×2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] → transpose to [B, T, C]
        if x.dim() == 3 and x.shape[1] != x.shape[2]:
            x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)               # [B, T, 2*H]
        # Temporal attention: score each timestep, softmax → weighted sum
        scores  = self.attn(out).squeeze(-1) # [B, T]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        context = (out * weights).sum(dim=1)  # [B, 2*H]
        context = self.drop(context)
        return self.head(context)


# ── Dilated CNN (without SE) ──────────────────────────────────────────────────

class DilatedCNNClassifier(nn.Module):
    """
    Simple dilated 1-D CNN without SE blocks (baseline for ablation comparison).
    Same channel sizes and dilation schedule as Lite-TCN-SE but no SE attention.
    """
    def __init__(self,
                 input_channels:    int       = 4,
                 num_classes:       int       = 2,
                 channels_per_layer: list[int] = None,
                 dilation_schedule:  list[int] = None,
                 kernel_size:        int       = 3,
                 dropout:            float     = 0.3):
        super().__init__()
        if channels_per_layer is None:
            channels_per_layer = [32, 64, 64, 128]
        if dilation_schedule is None:
            dilation_schedule = [1, 2, 4, 8]

        self.convs = nn.ModuleList(
            _DilatedBlock(in_c, out_c, kernel_size, dil, dropout)
            for in_c, out_c, dil in zip(
                [input_channels] + channels_per_layer[:-1],
                channels_per_layer,
                dilation_schedule)
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels_per_layer[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.convs:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)


class _DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dil, dropout):
        super().__init__()
        pad = (kernel - 1) * dil
        self.pad  = nn.ConstantPad1d((pad, 0), 0)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dil, padding=0)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.res  = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        out = self.pad(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out + res
