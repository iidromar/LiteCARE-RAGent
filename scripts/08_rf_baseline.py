"""
Script 07 — Random Forest LOSO Baseline on WESAD
==================================================
Trains a Random Forest using proper Leave-One-Subject-Out CV on WESAD,
producing a fair (apples-to-apples) comparison against the TCN LOSO results.

Usage:
    python scripts/07_rf_loso_wesad.py [--config config/config.yaml]
                                        [--out_dir results/rf_loso_wesad]
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.preprocessing.wesad_loader import load_all_subjects
from src.models.baselines import build_rf_pipeline, extract_statistical_features
from src.evaluation.metrics import compute_metrics

def run_rf_loso(all_data: dict, out_dir: Path) -> pd.DataFrame:
    subjects = sorted(all_data.keys())
    results  = []

    print(f"\nRF LOSO CV — {len(subjects)} folds")

    for i, test_sid in enumerate(subjects):
        train_sids = [s for s in subjects if s != test_sid]

        X_tr = np.concatenate([all_data[s][0] for s in train_sids])
        y_tr = np.concatenate([all_data[s][1] for s in train_sids])
        X_te, y_te = all_data[test_sid]

        # Extract statistical features: [N, C*4] = [N, 16]
        X_tr_feat = extract_statistical_features(X_tr)
        X_te_feat = extract_statistical_features(X_te)

        rf = build_rf_pipeline()
        rf.fit(X_tr_feat, y_tr)
        y_pred = rf.predict(X_te_feat)
        y_prob = rf.predict_proba(X_te_feat)

        m = compute_metrics(y_te, y_pred, y_prob)
        m["test_subject"] = test_sid
        results.append(m)

        print(f"  [{i+1:2d}/{len(subjects)}] Test={test_sid}  "
              f"F1={m.get('f1',0):.4f}  Acc={m.get('accuracy',0):.4f}  "
              f"AUROC={m.get('auroc',0):.4f}")

    df = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rf_loso_results.csv"
    df.to_csv(csv_path, index=False)

    f1s  = df["f1"].values
    accs = df["accuracy"].values
    aucs = df["auroc"].values if "auroc" in df.columns else [float("nan")] * len(df)

    print(f"\n=== RF LOSO Summary ===")
    print(f"  F1       : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  AUROC    : {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")
    print(f"\nResults saved to {csv_path}")
    return df

def main():
    parser = argparse.ArgumentParser(description="RF LOSO baseline on WESAD")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--out_dir", default="results/rf_loso_wesad")
    args = parser.parse_args()

    cfg = load_config(args.config)
    all_data = load_all_subjects(cfg["data"]["wesad_root"],
                                  cfg["data"]["processed_dir"])
    run_rf_loso(all_data, Path(args.out_dir))

if __name__ == "__main__":
    main()
