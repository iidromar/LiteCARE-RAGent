"""
SWELL-KW Dataset Loader (Feature-Level)
========================================
Uses the minute-level processed physiology features (HR, RMSSD, SCL)
available in the SWELL feature dataset CSV.

NOTE: Raw physiology is in binary .S00 (Mobi device) format that requires
      proprietary software.  We therefore use the processed feature file only.
      This dataset is used for CROSS-DATASET EVALUATION with feature-level
      classifiers (Random Forest) — NOT for the raw-signal TCN model.

Label mapping (from Koldijk et al. 2014):
    Condition 1 = No stress  → 0
    Condition 2 = Time pressure     → 1
    Condition 3 = Email interruption → 1
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product


FEATURE_COLS = ["HR", "RMSSD", "SCL"]
WINDOW_MINUTES = 5   # 5-minute windows
STEP_MINUTES   = 1   # 1-minute step (80% overlap for minute-level data)
NAN_VALUE      = 999


def load_swell_features(swell_root: str,
                        processed_dir: str,
                        force_reprocess: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SWELL physiology feature dataset and create feature windows.

    Returns:
        X:     [N, 12]  — 5-min windows × (mean, std, min, max) for each of 3 channels
        y:     [N]      — binary stress labels
        meta:  [N, 2]   — (participant_id, condition) for LOSO splitting
    """
    swell_root    = Path(swell_root)
    processed_dir = Path(processed_dir) / "swell"
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / "swell_features.npz"
    if out_path.exists() and not force_reprocess:
        arr = np.load(out_path, allow_pickle=True)
        print(f"Loaded SWELL features from cache: {arr['X'].shape[0]} windows")
        return arr["X"], arr["y"], arr["meta"]

    # Path to feature file
    feat_path = (swell_root
                 / "3 - Feature dataset"
                 / "per sensor"
                 / "D - Physiology features (HR_HRV_SCL - final).csv")

    if not feat_path.exists():
        raise FileNotFoundError(f"SWELL physiology features not found: {feat_path}")

    df = pd.read_csv(feat_path)
    df.replace(NAN_VALUE, np.nan, inplace=True)

    # Rename columns to standard names if needed
    # Expected: PP, C, Condition, timestamp, HR, RMSSD, SCL
    df.columns = [c.strip() for c in df.columns]

    # Drop rows with missing physiology
    df.dropna(subset=FEATURE_COLS, inplace=True)

    # Binary stress label: C∈{2,3} → 1 (stressed), C=1 → 0
    df["label"] = df["C"].apply(lambda c: 1 if int(c) in [2, 3] else 0)

    X_list, y_list, meta_list = [], [], []

    for (pp, cond), group in df.groupby(["PP", "C"]):
        group = group.reset_index(drop=True)
        n = len(group)

        for start in range(0, n - WINDOW_MINUTES + 1, STEP_MINUTES):
            window = group.iloc[start:start + WINDOW_MINUTES]
            if len(window) < WINDOW_MINUTES:
                continue

            # Statistical features: mean, std, min, max for each channel
            feats = []
            for col in FEATURE_COLS:
                vals = window[col].values.astype(np.float32)
                feats.extend([vals.mean(), vals.std(), vals.min(), vals.max()])

            # Label: majority vote over window
            label = int(window["label"].mode()[0])

            X_list.append(feats)
            y_list.append(label)
            meta_list.append([str(pp), int(cond)])

    X    = np.array(X_list,    dtype=np.float32)    # [N, 12]
    y    = np.array(y_list,    dtype=np.int64)       # [N]
    meta = np.array(meta_list, dtype=object)         # [N, 2]

    np.savez_compressed(out_path, X=X, y=y, meta=meta)
    print(f"SWELL features: {X.shape[0]} windows | "
          f"stress={y.sum()} ({100*y.mean():.1f}%)")
    return X, y, meta


def get_loso_splits_swell(X: np.ndarray,
                          y: np.ndarray,
                          meta: np.ndarray) -> list[dict]:
    """
    Leave-One-Subject-Out splits for SWELL.

    Returns:
        list of dicts with X_train, y_train, X_test, y_test, test_subject
    """
    participants = np.unique(meta[:, 0])
    splits = []

    for test_pp in participants:
        test_mask  = meta[:, 0] == test_pp
        train_mask = ~test_mask
        splits.append({
            "test_subject": test_pp,
            "X_train": X[train_mask],
            "y_train": y[train_mask],
            "X_test":  X[test_mask],
            "y_test":  y[test_mask],
        })
    return splits
