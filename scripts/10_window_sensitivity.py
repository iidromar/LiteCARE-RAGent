"""
Script 10 — Window Size Sensitivity Analysis (Table 7 equivalent)
==================================================================
Evaluates the effect of temporal window size on stress detection F1.
Tests 4 window durations: 10s, 30s, 60s (default), 120s.

For each window size:
  - Re-segments WESAD data on-the-fly (no disk writes needed)
  - Trains a Random Forest LOSO CV (fast, model-agnostic proxy)
  - Reports F1, Accuracy, AUROC

Usage:
    python scripts/10_window_sensitivity.py [--config config/config.yaml]
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.preprocessing.wesad_loader import load_subject_raw
from src.models.baselines import build_rf_pipeline, extract_statistical_features
from src.evaluation.metrics import compute_metrics

BASE = Path("")

TARGET_FS = 32
WINDOW_SECONDS = [10, 30, 60, 120]

def _window_binary(channels: np.ndarray, labels: np.ndarray,
                   win_samples: int, step_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding window on pre-processed data.
    labels: -1=undefined, 0=non-stress, 1=stress
    Skips windows with <50% valid (non-undefined) samples.
    """
    C, T = channels.shape
    X_out, y_out = [], []
    for s in range(0, T - win_samples + 1, step_samples):
        x_w = channels[:, s:s + win_samples]
        y_w = labels[s:s + win_samples]
        valid = y_w[y_w >= 0]
        if len(valid) < win_samples // 2:
            continue
        label = int(np.bincount(valid.astype(int)).argmax())
        X_out.append(x_w)
        y_out.append(label)
    if not X_out:
        return np.empty((0, C, win_samples), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(X_out).astype(np.float32), np.array(y_out, dtype=np.int64)

def window_data(all_raw: dict, win_sec: int) -> dict:
    """
    Re-segment all subjects with a new window size.
    Returns dict: {subject_id: (X, y)} with X shape [N, 4, win_samples].
    """
    win_samples  = win_sec * TARGET_FS
    step_samples = win_samples // 2   # 50% overlap

    result = {}
    for sid, (channels, labels) in all_raw.items():
        X, y = _window_binary(channels, labels, win_samples, step_samples)
        if len(y) == 0:
            continue
        result[sid] = (X, y)
    return result

def run_rf_loso(all_data: dict) -> list[dict]:
    subjects = sorted(all_data.keys())
    results  = []
    for test_sid in subjects:
        train_sids = [s for s in subjects if s != test_sid]
        X_tr = np.concatenate([all_data[s][0] for s in train_sids])
        y_tr = np.concatenate([all_data[s][1] for s in train_sids])
        X_te, y_te = all_data[test_sid]

        X_tr_f = extract_statistical_features(X_tr)
        X_te_f = extract_statistical_features(X_te)

        rf = build_rf_pipeline()
        rf.fit(X_tr_f, y_tr)
        y_pred = rf.predict(X_te_f)
        y_prob = rf.predict_proba(X_te_f)

        m = compute_metrics(y_te, y_pred, y_prob)
        m["test_subject"] = test_sid
        results.append(m)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wesad_root = Path(cfg["data"]["wesad_root"])

    # Load raw (resampled, z-normed, but NOT windowed) signals for all subjects
    print("Loading raw WESAD signals...")
    all_raw = {}
    for pkl_path in sorted(wesad_root.glob("S*/S*.pkl")):
        sid = pkl_path.parent.name
        result = load_subject_raw(pkl_path)
        if result is not None:
            all_raw[sid] = result   # (channels [4,T], labels [T])
    print(f"  Loaded {len(all_raw)} subjects")

    rows = []
    for win_sec in WINDOW_SECONDS:
        print(f"\n── Window = {win_sec}s ({win_sec * TARGET_FS} samples) ──")
        all_data = window_data(all_raw, win_sec)
        n_subjects = len(all_data)
        n_windows  = sum(len(v[1]) for v in all_data.values())
        print(f"   Subjects: {n_subjects}  |  Total windows: {n_windows}")

        results = run_rf_loso(all_data)
        f1s   = np.array([r["f1"]       for r in results])
        accs  = np.array([r["accuracy"] for r in results])
        aurocs= np.array([r.get("auroc", float("nan")) for r in results])

        row = {
            "window_sec":   win_sec,
            "n_windows":    n_windows,
            "f1_mean":      float(np.mean(f1s)),
            "f1_std":       float(np.std(f1s)),
            "acc_mean":     float(np.mean(accs)),
            "auroc_mean":   float(np.nanmean(aurocs)),
        }
        rows.append(row)
        print(f"   F1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  "
              f"Acc={row['acc_mean']:.3f}  AUROC={row['auroc_mean']:.3f}")

    df = pd.DataFrame(rows)

    print("\n" + "="*70)
    print("WINDOW SIZE SENSITIVITY — RF LOSO on WESAD (Table 7)")
    print("="*70)
    print(f"{'Window (s)':>10}  {'N windows':>10}  {'F1 ± std':>14}  {'Acc':>7}  {'AUROC':>7}")
    print("-"*70)
    for _, r in df.iterrows():
        marker = "  ← default" if r["window_sec"] == 60 else ""
        print(f"{int(r['window_sec']):>10}  {int(r['n_windows']):>10}  "
              f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}  "
              f"{r['acc_mean']:>7.3f}  {r['auroc_mean']:>7.3f}{marker}")
    print("="*70)

    out_path = BASE / "results/window_sensitivity.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
