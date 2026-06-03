"""
AffectiveROAD Dataset Loader
=============================
Extracts E4 wristband signals from Left.zip files for each drive,
loads the subjective stress metric (SM_Drv*.csv), binarizes labels,
resamples to 32 Hz, z-normalizes per drive, and segments windows.

E4 CSV format:
    Row 0: Unix timestamp (start time)
    Row 1: Sample rate (Hz)
    Rows 2+: Signal values

Signal native rates:
    ACC.csv   → 32 Hz  (3 columns: x, y, z)
    BVP.csv   → 64 Hz
    EDA.csv   →  4 Hz
    TEMP.csv  →  4 Hz

Stress metric (SM_Drv*.csv):
    Single column 'x' — continuous stress score per sample
    Binarized: score > 75th percentile → stress=1, else non-stress=0
"""
from __future__ import annotations
import zipfile
import io
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .pipeline import (
    resample_signals,
    acc_magnitude,
    znorm_per_subject,
    build_channel_matrix,
    _resample_channel,
)


TARGET_FS  = 32
WINDOW     = 1920
STEP       = 960

# Native E4 sampling rates
E4_FS = {
    "EDA":  4,
    "BVP":  64,
    "TEMP": 4,
    "ACC":  32,
}


def _parse_e4_csv(raw_bytes: bytes, signal_name: str) -> tuple[np.ndarray, int]:
    """
    Parse an Empatica E4 CSV file from raw bytes.

    Returns:
        data: 1-D (or 2-D for ACC) float32 array
        fs:   sampling rate (Hz)
    """
    text = raw_bytes.decode("utf-8")
    lines = text.strip().split("\n")

    # First line: start timestamp (ignore for now)
    # Second line: sample rate (ACC has "32.0, 32.0, 32.0" — take first value)
    fs_str = lines[1].strip().split(",")[0].strip()
    fs = int(float(fs_str))

    # Remaining lines: data
    data_lines = lines[2:]
    rows = []
    for line in data_lines:
        vals = [float(v) for v in line.strip().split(",") if v.strip()]
        if vals:
            rows.append(vals)

    data = np.array(rows, dtype=np.float32)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data.flatten()

    return data, fs


def _load_e4_from_zip(zip_path: Path) -> dict[str, np.ndarray]:
    """
    Load EDA, BVP, TEMP, ACC signals from an Empatica E4 zip file.

    Returns:
        dict of signal arrays, ACC already converted to magnitude.
    """
    signals = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for sig_name in ["EDA", "BVP", "TEMP", "ACC"]:
            filename = f"{sig_name}.csv"
            # Some zips have subdirectories
            matches = [n for n in names if n.endswith(filename)]
            if not matches:
                print(f"  [WARNING] {filename} not found in {zip_path.name}")
                continue
            raw = zf.read(matches[0])
            data, fs = _parse_e4_csv(raw, sig_name)
            signals[sig_name] = data
    return signals


def _load_stress_metric(sm_path: Path) -> np.ndarray:
    """Load SM_DrvX.csv — single column 'x' = continuous stress score."""
    df = pd.read_csv(sm_path)
    col = df.columns[0]   # should be 'x' or similar
    return df[col].values.astype(np.float32)


def _binarize_stress(scores: np.ndarray, percentile: float = 75.0) -> np.ndarray:
    """
    Binarize continuous stress scores into binary labels.

    Proposal specification: stress = (Arousal > 0.6) AND (Valence < 0.4)
    Implementation deviation: The SM_Drv*.csv provides a single continuous
    stress metric (not separate Arousal/Valence dimensions). The dual-threshold
    criterion from the proposal cannot be directly applied.

    Pragmatic approximation: score > 75th percentile (per-drive) → stress=1.
    Rationale: The upper quartile captures the most arousing/negative driving
    segments, closely approximating the "high arousal + negative valence" intent
    of the original criterion. This is a documented dataset-specific limitation.

    Reference: El Haouij et al. (2018) — SM is a composite arousal-valence proxy.
    """
    threshold = np.nanpercentile(scores, percentile)
    return (scores > threshold).astype(np.int64)


def _align_label_to_signal(label: np.ndarray, target_len: int) -> np.ndarray:
    """Nearest-neighbour resampling for label arrays."""
    orig_len = len(label)
    if orig_len == target_len:
        return label
    idx = np.round(np.linspace(0, orig_len - 1, target_len)).astype(int)
    return label[idx]


def _sliding_window(X: np.ndarray, y: np.ndarray,
                    window: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    C, T = X.shape
    starts = range(0, T - window + 1, step)
    X_out, y_out = [], []
    for s in starts:
        y_w = y[s:s + window]
        # Majority vote
        label = int(np.bincount(y_w.astype(int)).argmax())
        X_out.append(X[:, s:s + window])
        y_out.append(label)
    if not X_out:
        return np.empty((0, C, window), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(X_out).astype(np.float32), np.array(y_out, dtype=np.int64)


def process_drive(drive_id: str,
                  e4_dir: Path,
                  sm_dir: Path,
                  target_fs: int = TARGET_FS,
                  window: int = WINDOW,
                  step: int = STEP) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a single AffectiveROAD drive.

    Args:
        drive_id: e.g. "Drv1"
        e4_dir:   path to Database/E4/
        sm_dir:   path to Database/Subj_metric/

    Returns:
        X_wins: [N, 4, window]
        y_wins: [N]
    """
    # Find the zip file for this drive
    # Directory pattern: {N}-E4-{drive_id}/Left.zip
    drive_num = drive_id.replace("Drv", "")
    zip_pattern = f"{drive_num}-E4-{drive_id}"
    matches = list(e4_dir.glob(f"{zip_pattern}/Left.zip"))

    if not matches:
        print(f"  [WARNING] No E4 zip found for {drive_id}")
        return np.empty((0, 4, window)), np.empty(0, dtype=np.int64)

    zip_path = matches[0]

    # Load E4 signals
    signals = _load_e4_from_zip(zip_path)
    if len(signals) < 3:
        print(f"  [WARNING] Insufficient channels for {drive_id}")
        return np.empty((0, 4, window)), np.empty(0, dtype=np.int64)

    # Convert ACC 3-axis → magnitude
    if "ACC" in signals:
        acc = signals.pop("ACC")
        if acc.ndim == 2:
            signals["ACC"] = acc_magnitude(acc)
        else:
            # Already 1-D (some E4 exports are pre-processed)
            signals["ACC"] = acc

    # Resample all channels to target_fs
    resampled = resample_signals(signals, E4_FS, target_fs)

    # Load stress metric
    sm_path = sm_dir / f"SM_{drive_id}.csv"
    if not sm_path.exists():
        print(f"  [WARNING] No SM file for {drive_id}")
        return np.empty((0, 4, window)), np.empty(0, dtype=np.int64)

    stress_metric = _load_stress_metric(sm_path)
    binary_label  = _binarize_stress(stress_metric, percentile=50.0)

    # Align label to signal length (use EDA as reference)
    ref_len = len(resampled.get("EDA", next(iter(resampled.values()))))
    aligned_label = _align_label_to_signal(binary_label, ref_len)

    # Z-normalize per drive
    for ch in resampled:
        resampled[ch] = znorm_per_subject(resampled[ch])

    # Stack to [4, T] — fill missing channels with zeros if needed
    channel_order = ["EDA", "BVP", "TEMP", "ACC"]
    for ch in channel_order:
        if ch not in resampled:
            resampled[ch] = np.zeros(ref_len, dtype=np.float32)

    X = build_channel_matrix(resampled, channel_order)

    # Align label to final signal length
    final_len = X.shape[1]
    aligned_label = _align_label_to_signal(aligned_label, final_len)

    return _sliding_window(X, aligned_label, window, step)


def load_all_drives(affectiveroad_root: str,
                    processed_dir: str,
                    force_reprocess: bool = False) -> dict[str, tuple]:
    """
    Process all AffectiveROAD drives.

    Returns:
        dict: {drive_id: (X [N,4,1920], y [N])}
    """
    root          = Path(affectiveroad_root)
    e4_dir        = root / "Database" / "E4"
    sm_dir        = root / "Database" / "Subj_metric"
    processed_dir = Path(processed_dir) / "affectiveroad"
    processed_dir.mkdir(parents=True, exist_ok=True)

    drive_ids = [f"Drv{i}" for i in range(1, 14)]
    all_data  = {}

    print(f"Processing {len(drive_ids)} AffectiveROAD drives...")
    for drive_id in tqdm(drive_ids, desc="AffectiveROAD"):
        out_path = processed_dir / f"{drive_id}_windows.npz"

        if out_path.exists() and not force_reprocess:
            arr = np.load(out_path)
            all_data[drive_id] = (arr["X"], arr["y"])
            continue

        X, y = process_drive(drive_id, e4_dir, sm_dir)

        if X.shape[0] == 0:
            print(f"  [SKIP] {drive_id}: no valid windows")
            continue

        np.savez_compressed(out_path, X=X, y=y)
        all_data[drive_id] = (X, y)
        print(f"  {drive_id}: {X.shape[0]} windows | "
              f"stress={y.sum()} ({100*y.mean():.1f}%)")

    return all_data
