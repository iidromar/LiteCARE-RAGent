"""Monte Carlo Dropout inference and uncertainty-based abstention."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Union


def mc_predict(model: nn.Module,
               x: torch.Tensor,
               n_samples: int = 30,
               device: str | None = None,
               hrv: "torch.Tensor | None" = None) -> dict:
    """
    MC-Dropout inference: N stochastic forward passes.

    Args:
        model:     Lite-TCN-SE (dropout layers present)
        x:         input tensor [B, C, T] or [C, T] (will add batch dim)
        n_samples: number of MC forward passes (N)
        device:    'cpu' or 'cuda'

    Returns:
        dict with:
            'mean_probs'   : [B, num_classes]  — averaged softmax probabilities
            'pred_class'   : [B]               — argmax of mean_probs
            'uncertainty'  : [B]               — predictive entropy per sample
            'all_probs'    : [N, B, num_classes] — full distribution
    """
    if device is None:
        device = next(model.parameters()).device

    if x.dim() == 2:
        x = x.unsqueeze(0)   # [1, C, T]

    x = x.to(device)
    if hrv is not None:
        hrv = hrv.to(device)

    # Enable dropout for MC sampling
    model.train()

    with torch.no_grad():
        probs_list = []
        for _ in range(n_samples):
            logits = model(x, hrv=hrv)                   # [B, num_classes]
            probs  = torch.softmax(logits, dim=-1)       # [B, num_classes]
            probs_list.append(probs.cpu().numpy())

    all_probs  = np.stack(probs_list, axis=0)           # [N, B, num_classes]
    mean_probs = all_probs.mean(axis=0)                 # [B, num_classes]
    pred_class = mean_probs.argmax(axis=-1)             # [B]
    uncertainty = _predictive_entropy(mean_probs)       # [B]

    return {
        "mean_probs":   mean_probs,
        "pred_class":   pred_class,
        "uncertainty":  uncertainty,
        "all_probs":    all_probs,
    }


def _predictive_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute predictive entropy :
        U(X) = -Σ_c P̄_c * log(P̄_c)

    Args:
        probs: [B, C] — mean probabilities
    Returns:
        entropy: [B]
    """
    return -(probs * np.log(probs + eps)).sum(axis=-1)


def calibrate_threshold(model: nn.Module,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        hrv_val: "np.ndarray | None" = None,
                        n_samples: int = 30,
                        percentile: float = 95.0,
                        batch_size: int = 64,
                        device: str = "cpu") -> float:
    """
    Calibrate abstention threshold τ on a validation set.

    Strategy: τ = percentile of entropy values from CORRECT predictions.
    Windows where the model is confident and correct → reference distribution.

    Args:
        model:      trained Lite-TCN-SE
        X_val:      [N, C, T] numpy array
        y_val:      [N] numpy array
        percentile: e.g. 95 means only 5% of correct predictions abstain
        device:     device string

    Returns:
        tau: float — abstention threshold
    """
    entropies_correct = []
    model.to(device)

    for start in range(0, len(X_val), batch_size):
        x_batch = torch.tensor(X_val[start:start + batch_size], dtype=torch.float32)
        y_batch = y_val[start:start + batch_size]
        hrv_batch = (torch.tensor(hrv_val[start:start + batch_size], dtype=torch.float32)
                     if hrv_val is not None else None)

        result = mc_predict(model, x_batch, n_samples=n_samples, device=device, hrv=hrv_batch)
        correct_mask = result["pred_class"] == y_batch
        entropies_correct.extend(result["uncertainty"][correct_mask].tolist())

    if not entropies_correct:
        return 0.5   # fallback

    tau = float(np.percentile(entropies_correct, percentile))
    print(f"Calibrated τ = {tau:.4f}  "
          f"(p{percentile} of entropy on {len(entropies_correct)} correct val predictions)")
    return tau


def agent_decision(uncertainty: float,
                   pred_class: int,
                   tau: float,
                   class_names: list[str] | None = None) -> dict:
    """
    Apply abstention rule .

    Returns:
        dict with 'decision': 'abstain' | 'stress' | 'no_stress'
                  'uncertainty': float
                  'confidence': float (1 - entropy / log(num_classes))
    """
    if class_names is None:
        class_names = ["no_stress", "stress"]

    uncertainty = float(uncertainty)
    pred_class  = int(pred_class)
    n_classes   = len(class_names)
    max_entropy = float(np.log(n_classes + 1e-8))
    confidence  = 1.0 - (uncertainty / max_entropy)

    if uncertainty > tau:
        return {
            "decision":    "abstain",
            "uncertainty": uncertainty,
            "confidence":  confidence,
            "reason":      f"U={uncertainty:.4f} > τ={tau:.4f}",
        }

    pred_label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
    return {
        "decision":    pred_label,
        "uncertainty": uncertainty,
        "confidence":  confidence,
        "reason":      f"U={uncertainty:.4f} ≤ τ={tau:.4f}",
    }
