"""Script 44 — Grad-CAM Heatmaps for Lite-TCN-SE v8b"""
from __future__ import annotations
import argparse
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.lite_tcn_se import LiteTCNSE

try:
    from src.preprocessing.hrv_features import extract_hrv_batch, N_HRV_FEATURES
    HAS_HRV = True
except ImportError:
    HAS_HRV = False
    N_HRV_FEATURES = 0

BASE     = Path(__file__).parent.parent
CKPT_DIR = BASE / "results/wesad_loso/v8"
FIG_DIR  = BASE / "results/figures/gradcam"
CONFIG   = BASE / "config/config_multidomain.yaml"

CHANNEL_NAMES = ["EDA", "BVP", "TEMP", "ACC"]
CHANNEL_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
TARGET_FS = 32
STRESS_CLASS = 1

def load_averaged_model(cfg: dict, hrv_feat: int, device: str) -> LiteTCNSE:
    mcfg = cfg["model"]
    model = LiteTCNSE(
        input_channels=mcfg["input_channels"],
        num_classes=mcfg["num_classes"],
        channels_per_layer=mcfg["channels_per_layer"],
        dilation_schedule=mcfg["dilation_schedule"],
        kernel_size=mcfg["kernel_size"],
        dropout_rate=mcfg["dropout_rate"],
        se_reduction=mcfg.get("se_reduction_ratio", 4),
        hrv_features=hrv_feat,
    ).to(device)

    ckpt_paths = sorted(CKPT_DIR.glob("v8_fold_*.pt"))
    avg_state = None
    for p in ckpt_paths:
        s = torch.load(p, map_location=device)
        avg_state = {k: s[k].clone().float() for k in s} if avg_state is None \
                    else {k: avg_state[k] + s[k].float() for k in avg_state}
    for k in avg_state:
        avg_state[k] /= len(ckpt_paths)

    model.load_state_dict(avg_state)
    model.eval()
    return model

class GradCAM1D:
    """
    Grad-CAM for 1-D temporal models.

    Hooks into the specified target_layer and computes:
      heatmap[t] = ReLU( Σ_c  mean_t(∂score/∂A_ct) × A_ct )

    Args:
        model:        PyTorch model
        target_layer: nn.Module — the layer whose activations are used
        target_class: class index to explain (default 1 = stress)
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 target_class: int = STRESS_CLASS):
        self.model        = model
        self.target_class = target_class
        self._activations = None
        self._gradients   = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output: [B, C, T] — detach for memory efficiency but keep as tensor
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0]: [B, C, T]
        self._gradients = grad_output[0].detach()

    def compute(self, x: torch.Tensor,
                hrv: torch.Tensor = None) -> np.ndarray:
        """
        Run one forward+backward pass and return normalised Grad-CAM heatmap.

        Args:
            x:   [1, C, T] input tensor (single window, requires_grad=False)
            hrv: [1, H] HRV features (optional)

        Returns:
            heatmap: np.ndarray of shape [T], values in [0, 1]
        """
        self.model.eval()
        x = x.clone().requires_grad_(False)

        # Enable gradient computation for the backward pass
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.model(x_in, hrv)
            score  = logits[0, self.target_class]
            self.model.zero_grad()
            score.backward()

        # α_c = global average of gradients over time dimension
        grads = self._gradients[0]          # [C, T]
        acts  = self._activations[0]        # [C, T]
        alpha = grads.mean(dim=-1)          # [C]

        # Weighted sum of activations
        cam = torch.relu((alpha.unsqueeze(-1) * acts).sum(dim=0))  # [T]
        cam = cam.cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def channel_saliency(self, x: torch.Tensor,
                         hrv: torch.Tensor = None) -> np.ndarray:
        """
        Per-channel importance: mean |gradient| w.r.t. raw input per channel.

        Returns:
            saliency: np.ndarray of shape [C_input=4]
        """
        self.model.eval()
        # Create a fresh leaf tensor with gradient tracking
        x_in = x.detach().clone()
        x_in.requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(x_in, hrv)
            score  = logits[0, self.target_class]
            self.model.zero_grad()
            score.backward(retain_graph=False)

        if x_in.grad is not None:
            sal = x_in.grad[0].abs().mean(dim=-1).cpu().numpy()  # [4]
        else:
            # Fallback: gradient of last TCN block activations w.r.t. channels
            if self._gradients is not None:
                sal = self._gradients[0].abs().mean(dim=-1).cpu().numpy()
                # Collapse to input channel count (4) via average if needed
                sal = np.array([sal.mean()] * 4)
            else:
                sal = np.ones(4, dtype=np.float32)
        return sal

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

def load_npz_dataset(npz_dir: Path, pattern: str = "*.npz",
                     max_windows: int = 2000
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate .npz files from a directory (up to max_windows)."""
    Xs, ys = [], []
    for f in sorted(npz_dir.glob(pattern)):
        d = np.load(f)
        Xs.append(d["X"]); ys.append(d["y"])
    if not Xs:
        return np.zeros((0, 4, 1920), dtype=np.float32), np.zeros(0, dtype=np.int32)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    # Subsample if too large
    if len(X) > max_windows:
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(X))[:max_windows]
        X, y = X[idx], y[idx]
    return X, y

def sample_windows(X: np.ndarray, y: np.ndarray,
                   label: int, n: int = 5,
                   seed: int = 42) -> np.ndarray:
    """Return n randomly sampled windows with the given label."""
    idx = np.where(y == label)[0]
    rng = np.random.RandomState(seed)
    chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)
    return X[chosen]

def plot_gradcam_window(ax_signal, ax_cam,
                        window: np.ndarray, cam: np.ndarray,
                        title: str, fs: int = TARGET_FS):
    """
    Draw raw signals + Grad-CAM heatmap overlay on two paired axes.

    ax_signal: top axis — raw signal traces (4 channels)
    ax_cam:    bottom axis — Grad-CAM heatmap as a coloured bar + line
    """
    T = window.shape[-1]
    t = np.arange(T) / fs   # seconds

    # Normalize each channel for visual clarity
    for c in range(window.shape[0]):
        sig = window[c]
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        ax_signal.plot(t, sig + c * 3, color=CHANNEL_COLORS[c],
                       linewidth=0.8, alpha=0.85, label=CHANNEL_NAMES[c])

    ax_signal.set_ylabel("Norm. amplitude\n(offset per channel)")
    ax_signal.set_xlim(0, T / fs)
    ax_signal.set_title(title, fontsize=10, fontweight="bold")
    ax_signal.legend(loc="upper right", fontsize=7, ncol=2)
    ax_signal.set_xticklabels([])

    # Grad-CAM as a filled area
    ax_cam.fill_between(t, cam, alpha=0.6, color="#d62728")
    ax_cam.plot(t, cam, color="#d62728", linewidth=0.8)
    ax_cam.set_xlim(0, T / fs)
    ax_cam.set_ylim(0, 1.05)
    ax_cam.set_xlabel("Time (s)")
    ax_cam.set_ylabel("Grad-CAM\nsalience")
    ax_cam.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

def plot_channel_saliency(ax, saliencies: np.ndarray, dataset_name: str):
    """Bar chart of mean per-channel input gradient saliency."""
    means = saliencies.mean(axis=0)
    stds  = saliencies.std(axis=0)
    bars  = ax.bar(CHANNEL_NAMES, means, yerr=stds, color=CHANNEL_COLORS,
                   capsize=4, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mean |gradient| saliency")
    ax.set_title(f"Channel Saliency — {dataset_name}", fontsize=10,
                 fontweight="bold")
    ax.set_ylim(0, max(means + stds) * 1.3 + 1e-8)
    for i, (bar, val) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] * 0.1 + 1e-8,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n_samples", type=int, default=30,
                        help="Number of windows to average for channel saliency")
    args = parser.parse_args()

    cfg = load_config(str(CONFIG))
    processed_dir = Path(cfg["data"]["processed_dir"])
    bvp_ch = cfg["data"].get("bvp_channel_idx", 1)
    fs_cfg = cfg["preprocessing"]["target_fs"]
    hrv_feat = N_HRV_FEATURES if HAS_HRV else 0
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading averaged WESAD model...")
    model = load_averaged_model(cfg, hrv_feat, args.device)

    # Hook into the LAST TCN block
    target_layer = model.tcn_blocks[-1]
    gcam = GradCAM1D(model, target_layer, target_class=STRESS_CLASS)

        print("Loading WESAD windows...")
    X_wesad, y_wesad = load_npz_dataset(processed_dir / "wesad")

    print("Loading AffectiveROAD windows...")
    X_affroad, y_affroad = load_npz_dataset(processed_dir / "affectiveroad")

    print("Loading EmpaticaE4Stress windows...")
    X_e4, y_e4 = load_npz_dataset(processed_dir / "empatica_e4stress")

    datasets = {
        "WESAD (Lab)":          (X_wesad,   y_wesad,   "wesad"),
        "AffectiveROAD (Drive)": (X_affroad, y_affroad, "affroad"),
        "EmpaticaE4Stress (WP)": (X_e4,      y_e4,      "e4stress"),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1: Combined Grad-CAM heatmaps (3 datasets × 2 classes)
    # ══════════════════════════════════════════════════════════════════════════
    print("\nGenerating combined Grad-CAM figure...")
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Grad-CAM Temporal Saliency — Lite-TCN-SE v8b\n"
        "(Stress class = 1, Target layer = last TCN-SE block)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    n_datasets = len(datasets)
    outer_gs = gridspec.GridSpec(n_datasets, 2, figure=fig,
                                  hspace=0.55, wspace=0.35)

    for row_idx, (ds_name, (X, y, ds_key)) in enumerate(datasets.items()):
        if len(X) == 0:
            print(f"  [SKIP] {ds_name} — no windows found")
            continue

        for col_idx, (label, label_name) in enumerate([(1, "Stress"), (0, "No Stress")]):
            sample_arr = sample_windows(X, y, label=label, n=1)
            if len(sample_arr) == 0:
                print(f"  [SKIP] {ds_name} — no windows with label={label}")
                continue

            window = sample_arr[0]    # (4, 1920)

            # Prepare tensors
            x_t = torch.tensor(window[None], dtype=torch.float32).to(args.device)
            hrv_dummy = torch.zeros(1, hrv_feat, dtype=torch.float32).to(args.device) \
                        if hrv_feat > 0 else None

            cam = gcam.compute(x_t, hrv_dummy)

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[row_idx, col_idx],
                height_ratios=[3, 1], hspace=0.08,
            )
            ax_sig = fig.add_subplot(inner_gs[0])
            ax_cam = fig.add_subplot(inner_gs[1])

            plot_gradcam_window(
                ax_sig, ax_cam, window, cam,
                title=f"{ds_name} — {label_name}",
            )

    out_path = FIG_DIR / "gradcam_all_datasets.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2: Per-dataset stress Grad-CAM (individual, thesis-quality)
    # ══════════════════════════════════════════════════════════════════════════
    for ds_name, (X, y, ds_key) in datasets.items():
        if len(X) == 0:
            continue
        for label, label_name in [(1, "stress"), (0, "nostress")]:
            samples = sample_windows(X, y, label=label, n=1)
            if len(samples) == 0:
                continue
            window = samples[0]
            x_t = torch.tensor(window[None], dtype=torch.float32).to(args.device)
            hrv_dummy = torch.zeros(1, hrv_feat, dtype=torch.float32).to(args.device) \
                        if hrv_feat > 0 else None
            cam = gcam.compute(x_t, hrv_dummy)

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5),
                                             gridspec_kw={"height_ratios": [3, 1],
                                                          "hspace": 0.08})
            plot_gradcam_window(ax1, ax2, window, cam,
                                title=f"Grad-CAM | {ds_name} | {label_name.replace('_', ' ').title()}")
            fname = FIG_DIR / f"gradcam_{ds_key}_{label_name}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved → {fname}")

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 3: Channel saliency bars (per dataset)
    # ══════════════════════════════════════════════════════════════════════════
    print("\nComputing channel saliency maps...")
    fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig3.suptitle("Per-Channel Input Gradient Saliency (stress class)\n"
                  "Lite-TCN-SE v8b averaged over WESAD checkpoints",
                  fontsize=12, fontweight="bold")

    for ax, (ds_name, (X, y, ds_key)) in zip(axes, datasets.items()):
        if len(X) == 0:
            ax.set_visible(False)
            continue

        stress_idx = np.where(y == 1)[0]
        if len(stress_idx) == 0:
            ax.set_visible(False)
            continue

        rng = np.random.RandomState(42)
        chosen = rng.choice(stress_idx, size=min(args.n_samples, len(stress_idx)),
                            replace=False)
        saliencies = []
        for i in chosen:
            window = X[i]
            x_t = torch.tensor(window[None], dtype=torch.float32).to(args.device)
            hrv_dummy = torch.zeros(1, hrv_feat, dtype=torch.float32).to(args.device) \
                        if hrv_feat > 0 else None
            sal = gcam.channel_saliency(x_t, hrv_dummy)[:4]   # input channels
            saliencies.append(sal)

        saliencies = np.array(saliencies)  # (n_samples, 4)
        # Normalise each sample's saliency so max channel = 1 (relative importance)
        row_max = saliencies.max(axis=1, keepdims=True) + 1e-12
        saliencies = saliencies / row_max
        plot_channel_saliency(ax, saliencies, ds_name)
        print(f"  {ds_name}: EDA={saliencies[:,0].mean():.4f}  "
              f"BVP={saliencies[:,1].mean():.4f}  "
              f"TEMP={saliencies[:,2].mean():.4f}  "
              f"ACC={saliencies[:,3].mean():.4f}")

        # Save per-dataset channel saliency CSV
        import pandas as pd
        df_sal = pd.DataFrame(saliencies, columns=CHANNEL_NAMES)
        df_sal.to_csv(FIG_DIR / f"channel_saliency_{ds_key}.csv", index=False)

    plt.tight_layout()
    out_sal = FIG_DIR / "channel_saliency_all.png"
    plt.savefig(out_sal, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_sal}")

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 4: Multi-window averaged heatmap per dataset (more robust)
    # ══════════════════════════════════════════════════════════════════════════
    print("\nGenerating averaged Grad-CAM heatmaps (N=10 windows each)...")
    fig4, axes4 = plt.subplots(3, 1, figsize=(14, 9))
    fig4.suptitle("Averaged Grad-CAM Heatmaps (N=10 stress windows)\nLite-TCN-SE v8b",
                  fontsize=12, fontweight="bold")

    T = 1920
    t_axis = np.arange(T) / TARGET_FS

    for ax, (ds_name, (X, y, ds_key)) in zip(axes4, datasets.items()):
        if len(X) == 0:
            ax.set_visible(False)
            continue

        stress_idx = np.where(y == 1)[0]
        if len(stress_idx) == 0:
            ax.set_visible(False)
            continue

        rng = np.random.RandomState(0)
        chosen = rng.choice(stress_idx, size=min(10, len(stress_idx)), replace=False)

        cams = []
        mean_signals = np.zeros((4, T), dtype=np.float32)

        for i in chosen:
            window = X[i]
            x_t = torch.tensor(window[None], dtype=torch.float32).to(args.device)
            hrv_dummy = torch.zeros(1, hrv_feat, dtype=torch.float32).to(args.device) \
                        if hrv_feat > 0 else None
            cam_i = gcam.compute(x_t, hrv_dummy)
            cams.append(cam_i)
            mean_signals += window / len(chosen)

        avg_cam = np.mean(cams, axis=0)

        # Plot mean signals + heatmap overlay
        ax2 = ax.twinx()
        ax2.fill_between(t_axis, avg_cam, alpha=0.25, color="#d62728", label="Grad-CAM")
        ax2.set_ylim(0, 1.5)
        ax2.set_ylabel("Grad-CAM salience", color="#d62728", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#d62728")

        for c in range(4):
            sig = mean_signals[c]
            sig_norm = (sig - sig.mean()) / (sig.std() + 1e-8)
            ax.plot(t_axis, sig_norm + c * 2.5, color=CHANNEL_COLORS[c],
                    linewidth=0.9, label=CHANNEL_NAMES[c], alpha=0.9)

        ax.set_title(f"{ds_name} (averaged over {len(chosen)} stress windows)",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Norm. amplitude (offset)", fontsize=9)
        ax.set_xlim(0, T / TARGET_FS)
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        if ds_name != list(datasets.keys())[-1]:
            ax.set_xticklabels([])

    axes4[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    out_avg = FIG_DIR / "gradcam_averaged_stress.png"
    plt.savefig(out_avg, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_avg}")

    gcam.remove_hooks()
    print("\nGrad-CAM complete. All figures saved to:", FIG_DIR)

if __name__ == "__main__":
    main()
