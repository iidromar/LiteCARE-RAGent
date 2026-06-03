"""
Domain-Adversarial Neural Network (DANN) wrapper for Lite-TCN-SE
================================================================
Implements Ganin et al. (2016) "Domain-Adversarial Training of Neural Networks"
for cross-domain stress detection (WESAD → AffectiveROAD).

Architecture:
    Backbone (LiteTCNSE)
        ↓  get_features()  → embedding [B, feat_dim]
        ├── Stress Classifier:  Linear(feat_dim → 2)   — unchanged from v8b
        └── Domain Classifier:  GRL → Linear(feat_dim → 64) → ReLU → Linear(64 → 2)
                                       (WESAD=0, AffectiveROAD=1)

At inference: domain classifier is dropped; model behaves exactly as v8b.

Lambda schedule (Ganin et al.):
    λ(p) = 2 / (1 + exp(−10·p)) − 1,   p = current_step / total_steps
    λ starts near 0 (warm-up), reaches 1 at end of training.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ── Gradient Reversal Layer ───────────────────────────────────────────────────
class GradientReversalFunction(Function):
    """
    Forward pass: identity.
    Backward pass: multiply gradient by -lambda.
    This forces the feature extractor to learn domain-invariant representations.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lam)


def dann_lambda(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """
    Ganin et al. (2016) lambda schedule.
    Smoothly increases from ~0 to ~1 over training.
    """
    p = step / max(total_steps, 1)
    return 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0


# ── DANN Wrapper ──────────────────────────────────────────────────────────────
class DANNWrapper(nn.Module):
    """
    Wraps LiteTCNSE with an additional domain classifier branch.

    Args:
        backbone:   trained or fresh LiteTCNSE instance
        feat_dim:   embedding dimension after GAP + optional HRV fusion
                    (256 without HRV, 256+32=288 with HRV)
        n_domains:  number of domains (default 2: WESAD, AffectiveROAD)
    """
    def __init__(self,
                 backbone:  nn.Module,
                 feat_dim:  int = 288,
                 n_domains: int = 2):
        super().__init__()
        self.backbone = backbone

        # Domain classifier: small 2-layer MLP applied after GRL
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, n_domains),
        )

    def get_features(self,
                     x:   torch.Tensor,
                     hrv: torch.Tensor = None) -> torch.Tensor:
        """
        Extract embedding vector from backbone (before stress classifier head).
        Returns [B, feat_dim].
        """
        out = self.backbone.tcn_blocks(x)   # [B, C, T]
        out = self.backbone.gap(out).squeeze(-1)   # [B, C]

        if self.backbone.hrv_proj is not None:
            if hrv is None:
                hrv = torch.zeros(out.shape[0], self.backbone.hrv_features,
                                  device=out.device, dtype=out.dtype)
            hrv_emb = self.backbone.hrv_proj(hrv)
            out = torch.cat([out, hrv_emb], dim=-1)

        return out   # [B, feat_dim]

    def forward(self,
                x:    torch.Tensor,
                hrv:  torch.Tensor = None,
                lam:  float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full DANN forward pass.

        Returns:
            stress_logits: [B, 2]
            domain_logits: [B, n_domains]
        """
        feat = self.get_features(x, hrv)                    # [B, feat_dim]
        stress_logits = self.backbone.head(feat)             # [B, 2]
        feat_rev      = grad_reverse(feat, lam)              # gradient reversed
        domain_logits = self.domain_classifier(feat_rev)    # [B, n_domains]
        return stress_logits, domain_logits

    def predict(self,
                x:   torch.Tensor,
                hrv: torch.Tensor = None) -> torch.Tensor:
        """
        Inference-only forward — domain classifier ignored.
        Identical to LiteTCNSE.forward().
        """
        feat = self.get_features(x, hrv)
        return self.backbone.head(feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
