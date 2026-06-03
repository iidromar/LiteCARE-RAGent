"""
Loss Functions
==============
Composite loss from proposal Equation 1:
    L = L_Focal(y, ŷ) + λ * L_Calib

L_Focal = Focal Loss (Lin et al., 2017) — focuses on hard examples,
          more effective than CE for class-imbalanced physiological signals.
          Replaces standard CE: L_FL = -α(1-p)^γ * log(p)
          Default: gamma=2.0, alpha per-class from class weights.
L_Calib = Expected Calibration Error — differentiable approximation
          (soft binning via temperature-scaled logits)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CalibrationLoss(nn.Module):
    """
    Differentiable Expected Calibration Error (ECE) loss.

    Groups predictions into confidence bins and penalizes the gap between
    confidence and accuracy within each bin.

    Reference: Guo et al. (2017), "On Calibration of Modern Neural Networks"
    """
    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins   = n_bins
        self.bin_edges = torch.linspace(0, 1, n_bins + 1)

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] — raw model outputs
            targets: [B]    — integer class labels

        Returns:
            ece: scalar tensor
        """
        probs      = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values              # [B]
        preds      = probs.argmax(dim=-1)                  # [B]
        correct    = (preds == targets).float()            # [B]

        ece = torch.tensor(0.0, device=logits.device, requires_grad=True)
        n   = logits.shape[0]

        for i in range(self.n_bins):
            lo = self.bin_edges[i].item()
            hi = self.bin_edges[i + 1].item()
            mask = (confidence >= lo) & (confidence < hi)
            if mask.sum() == 0:
                continue
            bin_conf = confidence[mask].mean()
            bin_acc  = correct[mask].mean()
            ece      = ece + (mask.float().sum() / n) * (bin_conf - bin_acc).abs()

        return ece


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017).

    FL(p) = -α * (1 - p)^γ * log(p)

    Focuses training on hard, misclassified examples by down-weighting
    easy correct predictions. Particularly effective for class-imbalanced
    physiological datasets (22% stress in WESAD).

    Args:
        gamma:        focusing exponent (default 2.0)
        class_weight: per-class alpha weights (same as CE weight)
        label_smooth: label smoothing epsilon (default 0.1)
    """
    def __init__(self,
                 gamma:        float             = 2.0,
                 class_weight: torch.Tensor | None = None,
                 label_smooth: float             = 0.1):
        super().__init__()
        self.gamma        = gamma
        self.class_weight = class_weight
        self.label_smooth = label_smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing: soften one-hot targets
        n_classes = logits.shape[-1]
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smooth / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smooth)

        log_probs = F.log_softmax(logits, dim=-1)          # [B, C]
        probs     = torch.exp(log_probs)                   # [B, C]

        # Focal weight: (1 - p_t)^gamma where p_t is prob of true class
        p_t      = (probs * smooth_targets).sum(dim=-1)    # [B]
        focal_wt = (1.0 - p_t).pow(self.gamma)             # [B]

        # Cross-entropy with smooth targets
        ce = -(smooth_targets * log_probs).sum(dim=-1)     # [B]

        # Per-class alpha weighting
        if self.class_weight is not None:
            alpha = self.class_weight.to(logits.device)
            alpha_t = (alpha.unsqueeze(0) * smooth_targets).sum(dim=-1)  # [B]
            ce = alpha_t * ce

        loss = focal_wt * ce
        return loss.mean()


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss (Ridnik et al., 2021).

    Uses different gamma values for positive (stress) and negative (no-stress)
    classes to independently control recall and precision:
      - gamma_neg: high value (e.g. 2.5) → aggressively suppress easy FP
      - gamma_pos: low value  (e.g. 1.0) → preserve TP, don't suppress stress signal

    This directly addresses the precision/recall imbalance without sacrificing F1.
    """
    def __init__(self,
                 gamma_pos:    float             = 1.0,
                 gamma_neg:    float             = 2.5,
                 class_weight: torch.Tensor | None = None,
                 label_smooth: float             = 0.1):
        super().__init__()
        self.gamma_pos    = gamma_pos
        self.gamma_neg    = gamma_neg
        self.class_weight = class_weight
        self.label_smooth = label_smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[-1]
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smooth / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smooth)

        log_probs = F.log_softmax(logits, dim=-1)
        probs     = torch.exp(log_probs)

        # Asymmetric gamma: stress class uses gamma_pos, no-stress uses gamma_neg
        p_t = (probs * smooth_targets).sum(dim=-1)           # [B] prob of true class
        # gamma per sample: gamma_pos if target=1 (stress), gamma_neg if target=0
        gamma_t = torch.where(targets == 1,
                              torch.full_like(p_t, self.gamma_pos),
                              torch.full_like(p_t, self.gamma_neg))
        focal_wt = (1.0 - p_t).pow(gamma_t)

        ce = -(smooth_targets * log_probs).sum(dim=-1)

        if self.class_weight is not None:
            alpha   = self.class_weight.to(logits.device)
            alpha_t = (alpha.unsqueeze(0) * smooth_targets).sum(dim=-1)
            ce = alpha_t * ce

        return (focal_wt * ce).mean()


class CompositeLoss(nn.Module):
    """
    Composite loss: L = L_Focal + λ * L_Calib

    Uses Focal Loss (instead of plain CE) for better handling of:
    - Class imbalance (22% stress in WESAD)
    - Hard examples (subjects like S17 with atypical physiology)

    Args:
        lambda_calib:  weight for calibration loss (default 0.1)
        n_bins:        number of calibration bins
        class_weight:  optional tensor of per-class weights
        focal_gamma:   focal loss focusing exponent (default 2.0)
        label_smooth:  label smoothing epsilon (default 0.1)
    """
    def __init__(self,
                 lambda_calib:  float             = 0.1,
                 n_bins:        int               = 15,
                 class_weight:  torch.Tensor | None = None,
                 focal_gamma:   float             = 2.0,
                 label_smooth:  float             = 0.1,
                 focal_gamma_neg: float | None    = None):
        super().__init__()
        self.lambda_calib = lambda_calib
        if focal_gamma_neg is not None:
            # Asymmetric focal loss: gamma_pos=focal_gamma, gamma_neg=focal_gamma_neg
            self.focal_loss = AsymmetricFocalLoss(
                gamma_pos=focal_gamma,
                gamma_neg=focal_gamma_neg,
                class_weight=class_weight,
                label_smooth=label_smooth)
        else:
            self.focal_loss = FocalLoss(gamma=focal_gamma,
                                        class_weight=class_weight,
                                        label_smooth=label_smooth)
        self.calib_loss   = CalibrationLoss(n_bins=n_bins)

    def forward(self,
                logits:  torch.Tensor,
                targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar tensor
            loss_dict:  {'ce': float, 'calib': float, 'total': float}
        """
        l_focal = self.focal_loss(logits, targets)
        l_calib = self.calib_loss(logits, targets)
        total   = l_focal + self.lambda_calib * l_calib

        return total, {
            "ce":    l_focal.item(),   # named 'ce' for backward compatibility
            "calib": l_calib.item(),
            "total": total.item(),
        }


def compute_class_weight(y: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to handle class imbalance.

    Args:
        y: integer label array

    Returns:
        weight tensor of shape [num_classes]
    """
    classes, counts = np.unique(y, return_counts=True)
    weight = 1.0 / counts
    weight = weight / weight.sum() * len(classes)   # normalize
    w_tensor = torch.zeros(len(classes))
    for i, c in enumerate(classes):
        w_tensor[c] = weight[i]
    return w_tensor
