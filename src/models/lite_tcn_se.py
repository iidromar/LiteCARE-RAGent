from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        layers = [
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(reduced, channels, bias=False), nn.Sigmoid()]
        self.excite = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.squeeze(x).squeeze(-1)
        return x * self.excite(z).unsqueeze(-1)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                              dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class MultiScaleTCNSEBlock(nn.Module):
    """Three parallel causal dilated convolutions (d×1, d×2, d×4) concatenated
    and processed by InstanceNorm → ReLU → SE → Dropout → residual add."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 dilation: int = 1, dropout: float = 0.3, se_r: int = 4):
        super().__init__()
        b1, b2, b3 = out_ch // 3, out_ch // 3, out_ch - 2 * (out_ch // 3)
        self.branch1  = CausalConv1d(in_ch, b1, kernel, dilation * 1)
        self.branch2  = CausalConv1d(in_ch, b2, kernel, dilation * 2)
        self.branch3  = CausalConv1d(in_ch, b3, kernel, dilation * 4)
        self.bn       = nn.InstanceNorm1d(out_ch, affine=True)
        self.act      = nn.ReLU(inplace=True)
        self.se       = SEBlock(out_ch, reduction=se_r)
        self.drop     = nn.Dropout(p=dropout)
        self.residual = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return self.drop(self.se(self.act(self.bn(out)))) + res


class LiteTCNSE(nn.Module):
    """Multi-scale Temporal Convolutional Network with Squeeze-and-Excitation.

    Input:  [B, 4, 1920]  (4 channels, 60 s at 32 Hz)
    HRV:    [B, 12]  optional feature vector fused at the classifier head
    Output: [B, 2]   logits (stress / no-stress)
    """

    def __init__(self,
                 input_channels: int = 4,
                 num_classes: int = 2,
                 channels_per_layer: list[int] = None,
                 dilation_schedule: list[int] = None,
                 kernel_size: int = 3,
                 dropout_rate: float = 0.3,
                 se_reduction: int = 2,
                 hrv_features: int = 12):
        super().__init__()
        if channels_per_layer is None:
            channels_per_layer = [64, 128, 128, 256]
        if dilation_schedule is None:
            dilation_schedule = [1, 2, 4, 8]

        blocks, in_ch = [], input_channels
        for out_ch, dil in zip(channels_per_layer, dilation_schedule):
            blocks.append(MultiScaleTCNSEBlock(
                in_ch, out_ch, kernel=kernel_size,
                dilation=dil, dropout=dropout_rate, se_r=se_reduction))
            in_ch = out_ch
        self.tcn_blocks    = nn.Sequential(*blocks)
        self.gap           = nn.AdaptiveAvgPool1d(1)
        self.hrv_features  = hrv_features
        tcn_out = channels_per_layer[-1]
        if hrv_features > 0:
            self.hrv_proj = nn.Sequential(
                nn.Linear(hrv_features, 32), nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate))
            self.head = nn.Linear(tcn_out + 32, num_classes)
        else:
            self.hrv_proj = None
            self.head     = nn.Linear(tcn_out, num_classes)

    def forward(self, x: torch.Tensor, hrv: torch.Tensor = None) -> torch.Tensor:
        out = self.gap(self.tcn_blocks(x)).squeeze(-1)
        if self.hrv_proj is not None:
            if hrv is None:
                hrv = torch.zeros(out.shape[0], self.hrv_features,
                                  device=out.device, dtype=out.dtype)
            out = torch.cat([out, self.hrv_proj(hrv)], dim=-1)
        return self.head(out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return self.count_parameters() * 4 / 1024 ** 2


class LiteTCNSE_NoSE(LiteTCNSE):
    """Ablation variant: SE blocks removed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for block in self.tcn_blocks:
            block.se = nn.Identity()

    def forward(self, x: torch.Tensor, hrv: torch.Tensor = None) -> torch.Tensor:
        out = x
        for block in self.tcn_blocks:
            res = block.residual(out)
            cat = torch.cat([block.branch1(out), block.branch2(out),
                             block.branch3(out)], dim=1)
            out = block.drop(block.act(block.bn(cat))) + res
        out = self.gap(out).squeeze(-1)
        if self.hrv_proj is not None:
            if hrv is None:
                hrv = torch.zeros(out.shape[0], self.hrv_features,
                                  device=out.device, dtype=out.dtype)
            out = torch.cat([out, self.hrv_proj(hrv)], dim=-1)
        return self.head(out)


class LiteTCNSE_FixedDilation(LiteTCNSE):
    """Ablation variant: fixed dilation=1 across all blocks."""

    def __init__(self, **kwargs):
        kwargs["dilation_schedule"] = [1, 1, 1, 1]
        super().__init__(**kwargs)


def build_model(variant: str = "full", input_channels: int = 4, **kwargs) -> nn.Module:
    """Build a LiteTCNSE model variant.

    variant: 'full' | 'no_se' | 'fixed_dilation'
    """
    kw = dict(input_channels=input_channels, **kwargs)
    if variant == "full":
        return LiteTCNSE(**kw)
    if variant == "no_se":
        return LiteTCNSE_NoSE(**kw)
    if variant == "fixed_dilation":
        return LiteTCNSE_FixedDilation(**kw)
    raise ValueError(f"Unknown variant: {variant!r}")


if __name__ == "__main__":
    m = LiteTCNSE()
    x = torch.randn(2, 4, 1920)
    print(f"Output : {m(x).shape}")
    print(f"Params : {m.count_parameters():,}")
    print(f"Size   : {m.model_size_mb():.2f} MB")
