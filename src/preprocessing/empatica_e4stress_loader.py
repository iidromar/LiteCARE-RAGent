"""
EmpaticaE4Stress Dataset Loader
=================================
Loads the EmpaticaE4Stress dataset (Mendeley Data, doi:10.17632/kb42z77m2g/2).

Dataset:  29 healthy adult subjects, Università Politecnica delle Marche.
Sensor:   Empatica E4 wristband — same device and CSV format as WESAD.
Protocol: Alternating stress tasks and rest periods (≈37 min total):

  Phase           Duration (s)   Label
  ─────────────────────────────────────
  Baseline        180            0 (rest)
  Task 1 – Lego (no instr.)  600   1 (stress)
  Rest            120            0
  Task 2 – Lego (with instr.) 300   1 (stress)
  Rest            120            0
  Task 3 – Count backwards   180   1 (stress)
  Rest            120            0
  Task 4 – Mental arithmetic 300   1 (stress)
  Rest            120            0
  Task 5 – Oral presentation  60   1 (stress)
  Rest            120            0

Reference: Simonetti et al. (2024), Data in Brief.
           PMC ID: PMC10847510

Folder structure expected:
  <root>/
    subject_01/
      ACC.csv, BVP.csv, EDA.csv, TEMP.csv, HR.csv, IBI.csv
    subject_02/
      ...
    ...
    subject_29/
      ...

E4 CSV format (identical to WESAD E4 export):
  Row 0: Unix start timestamp
  Row 1: Sample rate (Hz)
  Rows 2+: Signal values (ACC has 3 columns: x, y, z)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .pipeline import (
    _resample_channel,
    acc_magnitude,
    znorm_per_subject,
    sliding_window,
)


TARGET_FS = 32
WINDOW    = 1920   # 60 s × 32 Hz
STEP      = 960    # 50 % overlap

# Native E4 sampling rates (Hz)
E4_FS = {"EDA": 4, "BVP": 64, "TEMP": 4, "ACC": 32}

# Protocol timeline: list of (duration_seconds, label)
# Based on PMC10847510 and Mendeley dataset description
PROTOCOL = [
    (180, 0),   # Baseline
    (600, 1),   # Task 1: Lego without instructions (10 min)
    (120, 0),   # Rest
    (300, 1),   # Task 2: Lego with instructions (5 min)
    (120, 0),   # Rest
    (180, 1),   # Task 3: Count backwards from 180 while building Lego
    (120, 0),   # Rest
    (300, 1),   # Task 4: Mental arithmetic (subtract 13 from 511)
    (120, 0),   # Rest
    (60,  1),   # Task 5: Oral presentation (1 min)
    (120, 0),   # Rest
]


def _parse_e4_csv(path: Path, signal_name: str) -> tuple[np.ndarray, int]:
    """
    Parse an Empatica E4 CSV file from the EmpaticaE4Stress dataset.

    The EmpaticaE4Stress dataset uses a simplified E4 export format with
    only ONE header row (a timestamp, often 0), unlike the standard E4 export
    used in WESAD which has two header rows (timestamp + sample rate).

    Format detected:
      Row 0: timestamp (integer or 0) — skipped
      Rows 1+: signal values

    Sample rates are fixed and known (E4 hardware specifications):
      EDA  →  4 Hz
      BVP  → 64 Hz
      TEMP →  4 Hz
      ACC  → 32 Hz  (3-column: x, y, z)

    Returns:
        data: 1-D (or 2-D for ACC: shape [N, 3]) float32 array
        fs:   sample rate (int)
    """
    # Known E4 sample rates — no need to parse from file
    known_fs = {"EDA": 4, "BVP": 64, "TEMP": 4, "ACC": 32}
    fs = known_fs[signal_name]

    with open(path, "r") as f:
        lines = f.read().splitlines()

    # Skip first row (timestamp); detect if second row is also a header
    # by checking if it looks numeric-only or contains non-data characters.
    skip = 1
    if len(lines) > 1:
        second = lines[1].strip()
        try:
            # If the second row parses cleanly as a single float ≥ 1 and
            # the value matches a known sample rate, treat it as a header too.
            val = float(second.split(",")[0])
            if val in {4.0, 32.0, 64.0, 1.0}:
                skip = 2
        except ValueError:
            pass

    data_lines = lines[skip:]

    if signal_name == "ACC":
        rows = []
        for line in data_lines:
            line = line.strip()
            if line:
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        rows.append([float(v) for v in parts[:3]])
                    except ValueError:
                        pass
        data = np.array(rows, dtype=np.float32)   # (N, 3)
    else:
        values = []
        for line in data_lines:
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass
        data = np.array(values, dtype=np.float32)

    return data, fs


def _build_protocol_labels(n_samples: int, fs: int) -> np.ndarray:
    """
    Build a sample-level binary label array from the fixed protocol timing.

    Args:
        n_samples: total number of samples in the recording at rate fs
        fs:        sample rate (Hz)

    Returns:
        labels: int array of shape (n_samples,), values in {0, 1}
    """
    labels = np.zeros(n_samples, dtype=np.int32)
    cursor = 0
    for duration_s, label in PROTOCOL:
        n = int(duration_s * fs)
        end = min(cursor + n, n_samples)
        labels[cursor:end] = label
        cursor = end
        if cursor >= n_samples:
            break
    return labels


def load_subject(subject_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load and preprocess one EmpaticaE4Stress subject.

    Steps:
      1. Parse EDA, BVP, TEMP, ACC CSV files
      2. Resample all channels to TARGET_FS (32 Hz)
      3. Compute ACC magnitude (3-axis → scalar)
      4. Build protocol-based binary labels at TARGET_FS
      5. Per-subject z-score normalization
      6. Sliding window segmentation (60 s, 50 % overlap)
      7. Majority-vote label per window

    Returns:
        X: float32 array (N_windows, 4, WINDOW)  — [EDA, BVP, TEMP, ACC_mag]
        y: int array    (N_windows,)
        or None if any required file is missing
    """
    required = ["EDA.csv", "BVP.csv", "TEMP.csv", "ACC.csv"]
    for fname in required:
        if not (subject_dir / fname).exists():
            return None

    # ── 1. Parse raw signals ──────────────────────────────────────────────────
    eda_raw,  eda_fs  = _parse_e4_csv(subject_dir / "EDA.csv",  "EDA")
    bvp_raw,  bvp_fs  = _parse_e4_csv(subject_dir / "BVP.csv",  "BVP")
    temp_raw, temp_fs = _parse_e4_csv(subject_dir / "TEMP.csv", "TEMP")
    acc_raw,  acc_fs  = _parse_e4_csv(subject_dir / "ACC.csv",  "ACC")

    # ── 2. Resample to TARGET_FS ──────────────────────────────────────────────
    eda  = _resample_channel(eda_raw,       eda_fs,  TARGET_FS)
    bvp  = _resample_channel(bvp_raw,       bvp_fs,  TARGET_FS)
    temp = _resample_channel(temp_raw,      temp_fs, TARGET_FS)
    # ACC: resample each axis, then compute magnitude
    if acc_raw.ndim == 2 and acc_raw.shape[1] == 3:
        axes = [_resample_channel(acc_raw[:, i], acc_fs, TARGET_FS)
                for i in range(3)]
        acc  = np.sqrt(sum(a**2 for a in axes)).astype(np.float32)
    else:
        acc = _resample_channel(acc_raw.ravel(), acc_fs, TARGET_FS)

    # ── 3. Align lengths (take minimum) ──────────────────────────────────────
    n = min(len(eda), len(bvp), len(temp), len(acc))
    eda, bvp, temp, acc = eda[:n], bvp[:n], temp[:n], acc[:n]

    # ── 4. Protocol-based labels at TARGET_FS ────────────────────────────────
    labels_raw = _build_protocol_labels(n, TARGET_FS)

    # ── 5. Per-subject z-score normalization ─────────────────────────────────
    signals = np.stack([eda, bvp, temp, acc], axis=0)   # (4, N)
    for c in range(signals.shape[0]):
        mu  = signals[c].mean()
        sig = signals[c].std() + 1e-8
        signals[c] = (signals[c] - mu) / sig

    # ── 6. Sliding window ────────────────────────────────────────────────────
    X_list, y_list = [], []
    for start in range(0, n - WINDOW + 1, STEP):
        win_X = signals[:, start:start + WINDOW]          # (4, 1920)
        win_y = labels_raw[start:start + WINDOW]
        label = int(win_y.mean() >= 0.5)                  # majority vote
        X_list.append(win_X)
        y_list.append(label)

    if not X_list:
        return None

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def load_all_subjects(root: str | Path,
                      processed_dir: str | Path,
                      force_reprocess: bool = False
                      ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load all EmpaticaE4Stress subjects.

    Saves each subject's (X, y) as a .npz file in
    <processed_dir>/empatica_e4stress/ for fast re-loading.

    Args:
        root:           path to EmpaticaE4Stress root (containing subject_01/…)
        processed_dir:  base processed-data directory
        force_reprocess: if True, re-process even if .npz already exists

    Returns:
        dict: subject_id → (X, y)
    """
    root = Path(root)
    out_dir = Path(processed_dir) / "empatica_e4stress"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover subject folders — handle "subject_01", "Subject_01", "S01" etc.
    subject_dirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and any((d / f).exists()
                              for f in ["EDA.csv", "BVP.csv"])
    ])
    # Also check one level deep (some zips unpack into a sub-folder)
    if not subject_dirs:
        for subdir in sorted(root.iterdir()):
            if subdir.is_dir():
                candidates = sorted([
                    d for d in subdir.iterdir()
                    if d.is_dir() and (d / "EDA.csv").exists()
                ])
                if candidates:
                    subject_dirs = candidates
                    break

    print(f"Found {len(subject_dirs)} subject directories in {root}")

    data = {}
    for subj_dir in tqdm(subject_dirs, desc="EmpaticaE4Stress subjects"):
        subj_id = subj_dir.name
        npz_path = out_dir / f"{subj_id}_windows.npz"

        if npz_path.exists() and not force_reprocess:
            d = np.load(npz_path)
            data[subj_id] = (d["X"], d["y"])
            continue

        result = load_subject(subj_dir)
        if result is None:
            print(f"  [SKIP] {subj_id} — missing required CSV files")
            continue

        X, y = result
        np.savez_compressed(npz_path, X=X, y=y)
        data[subj_id] = (X, y)
        print(f"  {subj_id}: {X.shape[0]} windows, "
              f"stress_rate={y.mean():.2f}")

    return data
