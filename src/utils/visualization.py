"""
Visualization Utilities
========================
Produces all figures used in the thesis results chapter:
  - Confusion matrices
  - ROC curves
  - Calibration reliability diagrams
  - Uncertainty distribution plots
  - Ablation bar charts
  - Model size vs. accuracy scatter
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)

FIG_DPI = 150
PALETTE = {
    "full":           "#2196F3",
    "no_se":          "#FF9800",
    "no_mc":          "#9C27B0",
    "fixed_dilation": "#F44336",
    "no_rag":         "#4CAF50",
    "rf":             "#795548",
    "lstm":           "#607D8B",
    "dilated_cnn":    "#009688",
}


def _save(fig: plt.Figure, out_path: str | Path, tight: bool = True):
    if tight:
        fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Confusion Matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    out_path: str | Path | None = None,
    labels: list[str] | None = None,
) -> plt.Figure:
    labels = labels or ["No Stress", "Stress"]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    if out_path:
        _save(fig, out_path)
    return fig


# ── ROC Curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(
    results: dict[str, dict],
    title: str = "ROC Curves",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Parameters
    ----------
    results : {model_name: {'y_true': ..., 'y_prob': ...}}
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")

    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        color = PALETTE.get(name, None)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    if out_path:
        _save(fig, out_path)
    return fig


# ── Calibration Reliability Diagram ────────────────────────────────────────────

def plot_calibration_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    title: str = "Calibration Diagram",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Reliability diagram: mean predicted confidence vs. actual accuracy per bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    fraction_of_positives = []
    mean_predicted = []
    bin_counts = []

    for lo, hi in zip(bin_lowers, bin_uppers):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        fraction_of_positives.append(y_true[mask].mean())
        mean_predicted.append(y_prob[mask].mean())
        bin_counts.append(mask.sum())

    fraction_of_positives = np.array(fraction_of_positives)
    mean_predicted = np.array(mean_predicted)

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_predicted, fraction_of_positives, "s-", color="#2196F3", lw=2, label="Model")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim([-0.05, 1.05])

    ax2 = axes[1]
    ax2.bar(mean_predicted, bin_counts, width=0.05, color="#2196F3", alpha=0.7)
    ax2.set_xlabel("Mean Predicted Confidence")
    ax2.set_ylabel("Count")

    if out_path:
        _save(fig, out_path)
    return fig


# ── Uncertainty Distribution ────────────────────────────────────────────────────

def plot_uncertainty_distribution(
    uncertainties: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tau: float | None = None,
    title: str = "Uncertainty Distribution",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Histogram of predictive entropy split by outcome:
        correct prediction | wrong prediction | (abstained if tau given)
    """
    correct_mask = y_true == y_pred
    wrong_mask   = ~correct_mask

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, uncertainties.max() + 0.01, 40)

    ax.hist(uncertainties[correct_mask], bins=bins, alpha=0.6,
            label="Correct", color="#4CAF50")
    ax.hist(uncertainties[wrong_mask], bins=bins, alpha=0.6,
            label="Incorrect", color="#F44336")

    if tau is not None:
        ax.axvline(tau, color="k", linestyle="--", lw=1.5, label=f"τ={tau:.3f}")

    ax.set_xlabel("Predictive Entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if out_path:
        _save(fig, out_path)
    return fig


# ── Ablation Bar Chart ─────────────────────────────────────────────────────────

def plot_ablation_bars(
    ablation_results: dict[str, dict],
    metric: str = "f1",
    title: str = "Ablation Study",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Parameters
    ----------
    ablation_results : {variant_name: {'f1': mean, 'f1_std': std, ...}}
    """
    names  = list(ablation_results.keys())
    values = [ablation_results[n].get(metric, 0) for n in names]
    stds   = [ablation_results[n].get(f"{metric}_std", 0) for n in names]
    colors = [PALETTE.get(n, "#9E9E9E") for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, values, yerr=stds, capsize=5, color=colors, width=0.6, alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_ylim([max(0, min(values) - 0.1), 1.05])

    if out_path:
        _save(fig, out_path)
    return fig


# ── Model Size vs. Accuracy Scatter ─────────────────────────────────────────────

def plot_size_vs_accuracy(
    model_info: dict[str, dict],
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Parameters
    ----------
    model_info : {name: {'size_mb': float, 'f1': float, 'params': int}}
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, info in model_info.items():
        x = info["size_mb"]
        y = info["f1"]
        color = PALETTE.get(name, "#9E9E9E")
        ax.scatter(x, y, s=120, color=color, zorder=3, label=name)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Size vs. F1 Score")
    ax.legend(fontsize=8, loc="lower right")

    if out_path:
        _save(fig, out_path)
    return fig


# ── Training History ──────────────────────────────────────────────────────────

def plot_training_history(
    history: dict,
    title: str = "Training History",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Parameters
    ----------
    history : {'train_loss': [...], 'val_loss': [...], 'train_f1': [...], 'val_f1': [...]}
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    ax = axes[0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val Loss", color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} — Loss")
    ax.legend()

    ax = axes[1]
    if "train_f1" in history:
        ax.plot(epochs, history["train_f1"], label="Train F1", color="#2196F3")
    if "val_f1" in history:
        ax.plot(epochs, history["val_f1"], label="Val F1", color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{title} — F1")
    ax.legend()

    if out_path:
        _save(fig, out_path)
    return fig


# ── LOSO Summary ──────────────────────────────────────────────────────────────

def plot_loso_f1_per_subject(
    fold_results: list[dict],
    title: str = "LOSO F1 per Subject",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Parameters
    ----------
    fold_results : list of {'test_subject': str, 'f1': float}
    """
    subjects = [r["test_subject"] for r in fold_results]
    f1s      = [r["f1"] for r in fold_results]
    mean_f1  = np.mean(f1s)

    fig, ax = plt.subplots(figsize=(max(6, len(subjects)), 4))
    x = np.arange(len(subjects))
    ax.bar(x, f1s, color="#2196F3", alpha=0.8, label="Per-subject F1")
    ax.axhline(mean_f1, color="#F44336", linestyle="--", lw=1.5, label=f"Mean={mean_f1:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.set_ylim([0, 1.05])
    ax.legend()

    if out_path:
        _save(fig, out_path)
    return fig
