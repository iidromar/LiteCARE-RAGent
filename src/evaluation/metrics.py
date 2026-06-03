"""
Evaluation Metrics (Table 8 from proposal)
===========================================
- Accuracy, Precision, Recall, F1-Score (binary + macro)
- AUROC
- Expected Calibration Error (ECE)
- Abstention Rate
- Model Size (MB) and Inference Time (ms)
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray | None = None) -> dict:
    """
    Compute standard classification metrics.

    Args:
        y_true: ground-truth labels [N]
        y_pred: predicted labels    [N]
        y_prob: predicted probs     [N, C] — required for AUROC and ECE

    Returns:
        metrics dict
    """
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary",
                                           zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="binary",
                                        zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="binary",
                                    zero_division=0)),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0)),
    }

    if y_prob is not None:
        # AUROC (binary: use probability of positive class)
        try:
            if y_prob.ndim == 2:
                pos_prob = y_prob[:, 1]
            else:
                pos_prob = y_prob
            metrics["auroc"] = float(roc_auc_score(y_true, pos_prob))
        except ValueError:
            metrics["auroc"] = float("nan")

        # ECE
        metrics["ece"] = expected_calibration_error(y_true, y_prob)

    return metrics


def expected_calibration_error(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                n_bins: int = 15) -> float:
    """
    Expected Calibration Error (non-differentiable version for reporting).

    Args:
        y_true: [N]    — integer labels
        y_prob: [N, C] — predicted probabilities

    Returns:
        ECE: float in [0, 1]
    """
    if y_prob.ndim == 2:
        confidence = y_prob.max(axis=1)
        preds      = y_prob.argmax(axis=1)
    else:
        confidence = y_prob
        preds      = (y_prob >= 0.5).astype(int)

    correct = (preds == y_true).astype(float)
    ece     = 0.0
    bins    = np.linspace(0, 1, n_bins + 1)
    n       = len(y_true)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidence[mask].mean()
        bin_acc  = correct[mask].mean()
        ece     += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


def measure_inference_time(model: nn.Module,
                            input_shape: tuple = (1, 4, 1920),
                            n_runs: int = 100,
                            device: str = "cpu") -> float:
    """
    Measure average inference time in milliseconds for a single window.

    Args:
        model:       PyTorch model
        input_shape: (batch, channels, timesteps)
        n_runs:      number of timing runs
        device:      'cpu' or 'cuda'

    Returns:
        mean inference time in ms
    """
    model.to(device)
    model.eval()
    x = torch.randn(*input_shape, device=device)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)   # ms

    return float(np.mean(times))


def compute_abstention_metrics(y_true:       np.ndarray,
                                y_pred:       np.ndarray,
                                uncertainties: np.ndarray,
                                tau:           float) -> dict:
    """
    Compute metrics with abstention.

    Returns:
        dict with 'abstention_rate', 'accuracy_after_abstention',
                  'precision_after_abstention', 'f1_after_abstention'
    """
    abstained   = uncertainties > tau
    non_abstain = ~abstained

    rate = float(abstained.mean())
    if non_abstain.sum() == 0:
        return {"abstention_rate": rate, "accuracy_after_abstention": float("nan")}

    y_t = y_true[non_abstain]
    y_p = y_pred[non_abstain]

    return {
        "abstention_rate":             rate,
        "accuracy_after_abstention":   float(accuracy_score(y_t, y_p)),
        "precision_after_abstention":  float(precision_score(y_t, y_p, average="binary",
                                                               zero_division=0)),
        "recall_after_abstention":     float(recall_score(y_t, y_p, average="binary",
                                                           zero_division=0)),
        "f1_after_abstention":         float(f1_score(y_t, y_p, average="binary",
                                                       zero_division=0)),
        "n_abstained":                 int(abstained.sum()),
        "n_decided":                   int(non_abstain.sum()),
    }
