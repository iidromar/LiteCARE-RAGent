"""
Cross-Dataset Evaluation
=========================
Train on one dataset → test on another.
Evaluates generalizability across different stress contexts.

Cross-validation strategies:
  1. Train WESAD → Test AffectiveROAD  (lab → driving)
  2. Train WESAD → Test SWELL          (lab → workplace, feature-level RF)
  3. Train AffectiveROAD → Test WESAD  (reverse)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from ..training.train import train_model, evaluate
from ..training.losses import CompositeLoss, compute_class_weight
from ..models.baselines import (
    build_rf_pipeline,
    extract_statistical_features,
    BiLSTMClassifier,
    DilatedCNNClassifier,
)
from .metrics import compute_metrics


def cross_dataset_tcn(X_source: np.ndarray,
                      y_source: np.ndarray,
                      X_target: np.ndarray,
                      y_target: np.ndarray,
                      cfg: dict,
                      out_dir: str,
                      source_name: str = "wesad",
                      target_name: str = "affectiveroad",
                      variant: str = "full") -> dict:
    """
    Train Lite-TCN-SE on source dataset, evaluate on target dataset.

    Returns:
        metrics dict
    """
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"cross_{source_name}_to_{target_name}_{variant}.pt"

    # Hold out 10% of source as validation
    n_val   = max(1, int(0.1 * len(X_source)))
    X_val, y_val = X_source[-n_val:], y_source[-n_val:]
    X_tr,  y_tr  = X_source[:-n_val], y_source[:-n_val]

    print(f"\nCross-dataset: {source_name} → {target_name} ({variant})")
    model, _ = train_model(X_tr, y_tr, X_val, y_val, cfg,
                           save_path=str(ckpt_path), variant=variant)

    device    = next(model.parameters()).device
    class_wt  = compute_class_weight(y_tr).to(device)
    criterion = CompositeLoss(class_weight=class_wt)

    test_ds     = TensorDataset(
        torch.tensor(X_target, dtype=torch.float32),
        torch.tensor(y_target, dtype=torch.long),
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    metrics     = evaluate(model, test_loader, criterion, str(device))
    metrics.update({"source": source_name, "target": target_name, "variant": variant})

    print(f"  F1={metrics.get('f1',0):.4f}  "
          f"Acc={metrics.get('accuracy',0):.4f}  "
          f"AUROC={metrics.get('auroc',0):.4f}")
    return metrics


def cross_dataset_rf(X_source: np.ndarray,
                     y_source: np.ndarray,
                     X_target: np.ndarray,
                     y_target: np.ndarray,
                     source_name: str = "wesad",
                     target_name: str = "swell") -> dict:
    """
    Train Random Forest on source (statistical features), evaluate on target.
    Used for SWELL cross-evaluation since it has only feature-level data.
    """
    # Source: extract statistical features if raw windows
    if X_source.ndim == 3:
        X_src_feat = extract_statistical_features(X_source)
    else:
        X_src_feat = X_source

    if X_target.ndim == 3:
        X_tgt_feat = extract_statistical_features(X_target)
    else:
        X_tgt_feat = X_target

    rf = build_rf_pipeline()
    rf.fit(X_src_feat, y_source)
    y_pred = rf.predict(X_tgt_feat)
    y_prob = rf.predict_proba(X_tgt_feat)

    metrics = compute_metrics(y_target, y_pred, y_prob)
    metrics.update({"source": source_name, "target": target_name, "model": "RandomForest"})
    print(f"RF cross {source_name}→{target_name}: "
          f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")
    return metrics


def run_all_cross_evals(wesad_data:        dict,
                         affectiveroad_data: dict,
                         swell_feats:        tuple,
                         cfg:                dict,
                         out_dir:            str) -> pd.DataFrame:
    """
    Run all planned cross-dataset evaluations.

    Args:
        wesad_data:         {subject_id: (X, y)}  from wesad_loader
        affectiveroad_data: {drive_id:   (X, y)}  from affectiveroad_loader
        swell_feats:        (X_feat, y, meta)      from swell_loader
        cfg:                loaded config
        out_dir:            output directory

    Returns:
        DataFrame of all cross-eval results
    """
    # Concatenate all windows per dataset
    X_wesad = np.concatenate([v[0] for v in wesad_data.values()], axis=0)
    y_wesad = np.concatenate([v[1] for v in wesad_data.values()], axis=0)

    X_aroad = np.concatenate([v[0] for v in affectiveroad_data.values()], axis=0)
    y_aroad = np.concatenate([v[1] for v in affectiveroad_data.values()], axis=0)

    X_swell, y_swell, _ = swell_feats

    results = []

    # 1. WESAD → AffectiveROAD (TCN model)
    r = cross_dataset_tcn(X_wesad, y_wesad, X_aroad, y_aroad, cfg, out_dir,
                          source_name="wesad", target_name="affectiveroad")
    results.append(r)

    # 2. AffectiveROAD → WESAD (TCN model)
    r = cross_dataset_tcn(X_aroad, y_aroad, X_wesad, y_wesad, cfg, out_dir,
                          source_name="affectiveroad", target_name="wesad")
    results.append(r)

    # 3. RF intra-SWELL (LOSO - compatible feature space for SWELL evaluation)
    # Note: WESAD→SWELL transfer not feasible (different sensor channels: EDA/BVP/TEMP/ACC vs HR/RMSSD/SCL)
    from sklearn.model_selection import LeaveOneGroupOut
    X_swell_feat = X_swell   # already 12-dim statistical features
    _, _, meta = swell_feats
    groups = meta[:, 0]       # participant IDs for LOSO
    rf_preds, rf_targets = [], []
    rf_probs = []
    for tr_idx, te_idx in LeaveOneGroupOut().split(X_swell_feat, y_swell, groups):
        rf = build_rf_pipeline()
        rf.fit(X_swell_feat[tr_idx], y_swell[tr_idx])
        p = rf.predict(X_swell_feat[te_idx])
        pb = rf.predict_proba(X_swell_feat[te_idx])
        rf_preds.extend(p)
        rf_targets.extend(y_swell[te_idx])
        rf_probs.extend(pb)
    r = compute_metrics(np.array(rf_targets), np.array(rf_preds), np.array(rf_probs))
    r.update({"source": "swell", "target": "swell", "model": "RandomForest-LOSO"})
    results.append(r)
    print(f"  RF SWELL LOSO: F1={r['f1']:.4f} Acc={r['accuracy']:.4f}")

    # 4. RF intra-WESAD (feature baseline for comparison)
    X_wesad_feat = extract_statistical_features(X_wesad)
    idx = np.random.permutation(len(X_wesad_feat))
    split = int(0.8 * len(idx))
    rf = build_rf_pipeline()
    rf.fit(X_wesad_feat[idx[:split]], y_wesad[idx[:split]])
    y_pred = rf.predict(X_wesad_feat[idx[split:]])
    y_prob = rf.predict_proba(X_wesad_feat[idx[split:]])
    r = compute_metrics(y_wesad[idx[split:]], y_pred, y_prob)
    r.update({"source": "wesad", "target": "wesad", "model": "RandomForest-intra"})
    results.append(r)

    df = pd.DataFrame(results)
    out_path = Path(out_dir) / "cross_dataset_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nCross-dataset results saved to {out_path}")
    return df
