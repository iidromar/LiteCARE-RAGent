"""Training loop for LiteTCNSE: single-run and LOSO cross-validation."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from pathlib import Path
from tqdm import tqdm

from .losses import CompositeLoss, compute_class_weight
from .callbacks import EarlyStopping, ModelCheckpoint
from .augmentation import AugmentedDataset
from ..models.lite_tcn_se import build_model
from ..evaluation.metrics import compute_metrics
from ..preprocessing.hrv_features import extract_hrv_batch, N_HRV_FEATURES


# ── HRV-aware DataLoader helper ────────────────────────────────────────────────

class HRVDataset(torch.utils.data.Dataset):
    """Dataset that returns (X_window, hrv_features, label) triples."""
    def __init__(self, X: np.ndarray, hrv: np.ndarray, y: np.ndarray,
                 augment: bool = False):
        self.base = AugmentedDataset(X, y, augment=augment)
        self.hrv  = torch.tensor(hrv, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, self.hrv[idx], y


def _make_loaders(X_tr, hrv_tr, y_tr,
                  X_val, hrv_val, y_val,
                  batch_size: int,
                  use_hrv: bool,
                  augment: bool = True):
    if use_hrv:
        train_ds = HRVDataset(X_tr,  hrv_tr,  y_tr,  augment=augment)
        val_ds   = HRVDataset(X_val, hrv_val, y_val, augment=False)
    else:
        train_ds = AugmentedDataset(X_tr,  y_tr,  augment=augment)
        val_ds   = AugmentedDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    return train_loader, val_loader


# ── Mixup helper ──────────────────────────────────────────────────────────────

def mixup_batch(X, y, hrv, alpha: float, device: str):
    """
    Apply Mixup augmentation to a batch.
    λ ~ Beta(alpha, alpha); mix X and compute mixed loss on both labels.
    Returns (X_mix, hrv_mix, y_a, y_b, lam).
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    B   = X.size(0)
    idx = torch.randperm(B, device=device)
    X_mix   = lam * X + (1 - lam) * X[idx]
    hrv_mix = (lam * hrv + (1 - lam) * hrv[idx]) if hrv is not None else None
    y_a, y_b = y, y[idx]
    return X_mix, hrv_mix, y_a, y_b, lam


# ── Single-epoch train / eval ──────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: CompositeLoss,
                    optimizer: torch.optim.Optimizer,
                    device: str,
                    use_hrv: bool = False,
                    mixup_alpha: float = 0.0) -> dict:
    model.train()
    total_loss, ce_sum, calib_sum = 0.0, 0.0, 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        if use_hrv:
            X_batch, hrv_batch, y_batch = batch
            hrv_batch = hrv_batch.to(device)
        else:
            X_batch, y_batch = batch
            hrv_batch = None

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # ── Mixup (v7) ─────────────────────────────────────────────────────
        if mixup_alpha > 0.0:
            X_batch, hrv_batch, y_a, y_b, lam = mixup_batch(
                X_batch, y_batch, hrv_batch, mixup_alpha, device)
        else:
            y_a, y_b, lam = y_batch, y_batch, 1.0

        optimizer.zero_grad()
        logits = model(X_batch, hrv_batch)

        if mixup_alpha > 0.0:
            loss_a, dict_a = criterion(logits, y_a)
            loss_b, dict_b = criterion(logits, y_b)
            loss      = lam * loss_a + (1 - lam) * loss_b
            loss_dict = {k: lam * dict_a[k] + (1 - lam) * dict_b[k]
                         for k in dict_a}
        else:
            loss, loss_dict = criterion(logits, y_batch)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict["total"]
        ce_sum     += loss_dict["ce"]
        calib_sum  += loss_dict["calib"]

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_a.cpu().numpy())

    n = len(loader)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    metrics.update({
        "loss":  total_loss / n,
        "ce":    ce_sum / n,
        "calib": calib_sum / n,
    })
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: CompositeLoss,
             device: str,
             use_hrv: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    for batch in loader:
        if use_hrv:
            X_batch, hrv_batch, y_batch = batch
            hrv_batch = hrv_batch.to(device)
        else:
            X_batch, y_batch = batch
            hrv_batch = None

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch, hrv_batch)
        _, loss_dict = criterion(logits, y_batch)
        total_loss += loss_dict["total"]

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(y_batch.cpu().numpy())

    n = len(loader)
    metrics = compute_metrics(
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
    )
    metrics["loss"] = total_loss / n
    return metrics


# ── train_model ────────────────────────────────────────────────────────────────

def train_model(X_train:    np.ndarray,
                y_train:    np.ndarray,
                X_val:      np.ndarray,
                y_val:      np.ndarray,
                cfg:        dict,
                save_path:  str,
                variant:    str = "full",
                device:     str | None = None,
                hrv_train:  np.ndarray | None = None,
                hrv_val:    np.ndarray | None = None) -> tuple[nn.Module, dict]:
    """
    Train a single Lite-TCN-SE model.

    Args:
        X_train/X_val:  [N, C, T]
        y_train/y_val:  [N]
        cfg:            loaded config dict
        save_path:      path to save best checkpoint
        variant:        'full' | 'no_se' | 'fixed_dilation'
        device:         'cpu' or 'cuda'
        hrv_train/val:  optional [N, 12] HRV feature arrays

    Returns:
        model:   best model loaded with checkpoint weights
        history: dict of per-epoch train/val metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_hrv    = (hrv_train is not None) and cfg.get("model", {}).get("use_hrv", False)
    hrv_dim    = N_HRV_FEATURES if use_hrv else 0

    # ── Build model ─────────────────────────────────────────────────────────
    mcfg  = cfg["model"]
    model = build_model(
        variant=variant,
        input_channels=mcfg["input_channels"],
        channels_per_layer=mcfg["channels_per_layer"],
        dilation_schedule=mcfg["dilation_schedule"] if variant != "fixed_dilation" else [1,1,1,1],
        kernel_size=mcfg["kernel_size"],
        dropout_rate=mcfg["dropout_rate"],
        se_reduction=mcfg["se_reduction_ratio"],
        hrv_features=hrv_dim,
    ).to(device)

    # ── Data loaders with augmentation ──────────────────────────────────────
    tcfg      = cfg["training"]
    class_wt  = compute_class_weight(y_train).to(device)
    criterion = CompositeLoss(
        lambda_calib=tcfg["lambda_calib"],
        n_bins=tcfg["n_calib_bins"],
        class_weight=class_wt,
        focal_gamma=tcfg.get("focal_gamma", 2.0),
        label_smooth=tcfg.get("label_smooth", 0.1),
        focal_gamma_neg=tcfg.get("focal_gamma_neg", None),
    )

    train_loader, val_loader = _make_loaders(
        X_train, hrv_train, y_train,
        X_val,   hrv_val,   y_val,
        batch_size=tcfg["batch_size"],
        use_hrv=use_hrv,
        augment=True,
    )

        mixup_alpha      = tcfg.get("mixup_alpha",      0.0)
    swa_start_epoch  = tcfg.get("swa_start_epoch",  0)
    swa_lr           = tcfg.get("swa_lr",            0.0005)

    # ── Optimizer: AdamW with cosine warm-restarts ───────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tcfg["learning_rate"],
                                  weight_decay=tcfg["weight_decay"])
    T_0 = tcfg.get("scheduler_T0", 30)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2)

    # ── SWA setup (v7) ───────────────────────────────────────────────────────
    swa_model     = AveragedModel(model) if swa_start_epoch > 0 else None
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr) if swa_start_epoch > 0 else None
    swa_active    = False

    # ── Callbacks ────────────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=tcfg["early_stopping_patience"], mode="max")
    checkpoint = ModelCheckpoint(save_path, mode="max")

    history = {"train": [], "val": []}

    for epoch in range(1, tcfg["max_epochs"] + 1):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer,
                                  device, use_hrv=use_hrv,
                                  mixup_alpha=mixup_alpha)
        val_m   = evaluate(model, val_loader, criterion, device, use_hrv=use_hrv)

        # ── SWA: switch scheduler and accumulate weights ──────────────────
        if swa_start_epoch > 0 and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_active = True
        else:
            scheduler.step()

        history["train"].append(train_m)
        history["val"].append(val_m)

        val_f1 = val_m.get("f1", 0.0)
        checkpoint(val_f1, model)
        if early_stop(val_f1):
            print(f"  Early stopping at epoch {epoch} (best val F1={early_stop.best:.4f})")
            break

        if epoch % 10 == 0 or epoch == 1:
            swa_tag = " [SWA]" if swa_active else ""
            print(f"  Epoch {epoch:3d}{swa_tag} | "
                  f"train_loss={train_m['loss']:.4f} f1={train_m.get('f1', 0):.4f} | "
                  f"val_loss={val_m['loss']:.4f}   f1={val_f1:.4f}")

    # ── Finalise SWA: update BN stats then save SWA model ────────────────────
    if swa_active:
        print("  Updating BatchNorm/InstanceNorm statistics for SWA model...")
        # build a plain loader (no augmentation) for BN update
        if use_hrv:
            bn_ds = HRVDataset(X_train, hrv_train, y_train, augment=False)
        else:
            bn_ds = AugmentedDataset(X_train, y_train, augment=False)
        bn_loader = DataLoader(bn_ds, batch_size=tcfg["batch_size"], shuffle=False)
        update_bn(bn_loader, swa_model, device=device)
        # Save SWA weights as the final checkpoint (overrides early-stop ckpt)
        torch.save(swa_model.module.state_dict(), save_path)
        print("  SWA checkpoint saved.")
        return swa_model.module, history

    # Load best checkpoint (non-SWA path)
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history


# ── LOSO CV ────────────────────────────────────────────────────────────────────

def run_loso_cv(all_data: dict,
                cfg:      dict,
                out_dir:  str,
                variant:  str = "full",
                device:   str | None = None) -> list[dict]:
    """
    Run Leave-One-Subject-Out cross-validation on WESAD.

    Validation set: one additional held-out subject (the subject
    immediately before the test subject in the sorted list).
    This is more representative than taking a temporal slice of data.

    Args:
        all_data: {subject_id: (X, y)}  from wesad_loader.load_all_subjects()
        cfg:      loaded config dict
        out_dir:  directory to save per-fold checkpoints and results
        variant:  model variant name

    Returns:
        results: list of per-fold metric dicts
    """
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(all_data.keys())
    results  = []
    use_hrv  = cfg.get("model", {}).get("use_hrv", False)

    print(f"\nRunning LOSO CV — {len(subjects)} folds | variant={variant} | hrv={use_hrv}")

    for i, test_sid in enumerate(subjects):
        # Validation = previous subject (wraps around)
        val_sid    = subjects[(i - 1) % len(subjects)]
        train_sids = [s for s in subjects if s != test_sid and s != val_sid]

        print(f"\n[Fold {i+1}/{len(subjects)}] "
              f"Test={test_sid}  Val={val_sid}  Train={len(train_sids)} subjects")

        X_tr  = np.concatenate([all_data[s][0] for s in train_sids])
        y_tr  = np.concatenate([all_data[s][1] for s in train_sids])
        X_val, y_val = all_data[val_sid]
        X_te,  y_te  = all_data[test_sid]

        # ── HRV feature extraction (optional) ─────────────────────────────
        hrv_tr = hrv_val = hrv_te = None
        if use_hrv:
            print("  Extracting HRV features...")
            fs     = cfg["data"].get("target_fs", 32)
            bvp_ch = cfg["data"].get("bvp_channel_idx", 1)
            hrv_tr  = extract_hrv_batch(X_tr,  bvp_channel=bvp_ch, fs=fs)
            hrv_val = extract_hrv_batch(X_val, bvp_channel=bvp_ch, fs=fs)
            hrv_te  = extract_hrv_batch(X_te,  bvp_channel=bvp_ch, fs=fs)

            # Z-score normalize HRV with training-set statistics.
            hrv_mean = hrv_tr.mean(axis=0, keepdims=True)
            hrv_std  = hrv_tr.std(axis=0,  keepdims=True) + 1e-8
            hrv_tr   = (hrv_tr  - hrv_mean) / hrv_std
            hrv_val  = (hrv_val - hrv_mean) / hrv_std
            hrv_te   = (hrv_te  - hrv_mean) / hrv_std

        ckpt_path = out_dir / f"{variant}_fold_{test_sid}.pt"
        model, _  = train_model(X_tr, y_tr, X_val, y_val, cfg,
                                save_path=str(ckpt_path),
                                variant=variant,
                                device=device,
                                hrv_train=hrv_tr,
                                hrv_val=hrv_val)

        # ── Evaluate on test subject ───────────────────────────────────────
        d = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if use_hrv and hrv_te is not None:
            test_ds = HRVDataset(X_te, hrv_te, y_te, augment=False)
        else:
            test_ds = AugmentedDataset(X_te, y_te, augment=False)

        test_loader = DataLoader(test_ds,
                                 batch_size=cfg["training"]["batch_size"],
                                 shuffle=False)
        class_wt  = compute_class_weight(y_tr)
        criterion = CompositeLoss(class_weight=class_wt.to(d))

        test_m = evaluate(model, test_loader, criterion, d, use_hrv=use_hrv)
        test_m["test_subject"] = test_sid
        results.append(test_m)
        print(f"  Test {test_sid}: F1={test_m.get('f1',0):.4f} "
              f"Acc={test_m.get('accuracy',0):.4f} "
              f"AUROC={test_m.get('auroc',0):.4f}")

    # Summary
    f1s = [r["f1"] for r in results]
    print(f"\nLOSO CV Summary ({variant}): "
          f"F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = out_dir / "loso_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    return results
