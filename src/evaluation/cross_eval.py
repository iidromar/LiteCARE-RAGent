from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from ..training.train import train_model, evaluate
from ..training.losses import CompositeLoss, compute_class_weight
from ..models.baselines import build_rf_pipeline, extract_statistical_features
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
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"cross_{source_name}_to_{target_name}_{variant}.pt"

    n_val  = max(1, int(0.1 * len(X_source)))
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


def run_all_cross_evals(wesad_data:        dict,
                        affectiveroad_data: dict,
                        cfg:               dict,
                        out_dir:           str) -> pd.DataFrame:
    X_wesad = np.concatenate([v[0] for v in wesad_data.values()], axis=0)
    y_wesad = np.concatenate([v[1] for v in wesad_data.values()], axis=0)

    X_aroad = np.concatenate([v[0] for v in affectiveroad_data.values()], axis=0)
    y_aroad = np.concatenate([v[1] for v in affectiveroad_data.values()], axis=0)

    results = []

    r = cross_dataset_tcn(X_wesad, y_wesad, X_aroad, y_aroad, cfg, out_dir,
                          source_name="wesad", target_name="affectiveroad")
    results.append(r)

    r = cross_dataset_tcn(X_aroad, y_aroad, X_wesad, y_wesad, cfg, out_dir,
                          source_name="affectiveroad", target_name="wesad")
    results.append(r)

    X_wesad_feat = extract_statistical_features(X_wesad)
    idx   = np.random.permutation(len(X_wesad_feat))
    split = int(0.8 * len(idx))
    rf    = build_rf_pipeline()
    rf.fit(X_wesad_feat[idx[:split]], y_wesad[idx[:split]])
    y_pred = rf.predict(X_wesad_feat[idx[split:]])
    y_prob = rf.predict_proba(X_wesad_feat[idx[split:]])
    r = compute_metrics(y_wesad[idx[split:]], y_pred, y_prob)
    r.update({"source": "wesad", "target": "wesad", "model": "RandomForest-intra"})
    results.append(r)

    df       = pd.DataFrame(results)
    out_path = Path(out_dir) / "cross_dataset_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nCross-dataset results saved to {out_path}")
    return df
