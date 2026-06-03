"""
WESAD Dataset Loader
====================
Loads each subject's .pkl file, extracts wrist (E4) physiological signals,
binarizes labels, resamples to 32 Hz, z-normalizes per subject, and
segments into sliding windows.

Output shape per subject:
    X: [N_windows, 4, 1920]   channels: [EDA, BVP, TEMP, ACC_mag]
    y: [N_windows]             0 = non-stress, 1 = stress

WESAD label mapping (from Schmidt et al. 2018):
    0  = not defined / transient
    1  = baseline
    2  = TSST stress  → binary label 1
    3  = amusement
    4  = meditation
    5,6,7 = other conditions (ignore)
"""
from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .pipeline import (
    resample_signals,
    acc_magnitude,
    znorm_per_subject,
    sliding_window,
    build_channel_matrix,
)


NATIVE_FS = {
    "EDA":  4,
    "BVP":  64,
    "TEMP": 4,
    "ACC":  32,    # 3-axis at 32 Hz
}

TARGET_FS = 32
WINDOW    = 1920   # 60 s × 32 Hz
STEP      = 960    # 50 % overlap

STRESS_LABEL    = 2     # TSST condition
UNDEFINED_LABEL = 0     # transient / not defined


def _load_subject(pkl_path: Path) -> dict | None:
    """Load a single WESAD subject pickle."""
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"  [WARNING] Could not load {pkl_path}: {e}")
        return None


def _extract_wrist(data: dict) -> tuple[dict, np.ndarray]:
    """
    Extract wrist (E4) signals and the label array.

    Returns:
        signals: dict {channel: 1-D array}
        label:   1-D array at chest sensor rate (700 Hz) — resampled below
    """
    wrist = data["signal"]["wrist"]

    # ACC is shape [T, 3]; BVP/EDA/TEMP are [T, 1] or [T]
    def squeeze(arr):
        return arr.squeeze() if arr.ndim > 1 and arr.shape[1] == 1 else arr

    signals = {
        "EDA":  squeeze(wrist["EDA"]).astype(np.float32),
        "BVP":  squeeze(wrist["BVP"]).astype(np.float32),
        "TEMP": squeeze(wrist["TEMP"]).astype(np.float32),
        "ACC":  wrist["ACC"].astype(np.float32),          # [T, 3]
    }
    label = data["label"].flatten().astype(np.int32)
    return signals, label


def _align_label(label: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample label array (originally at 700 Hz) to match the resampled
    signal length (at TARGET_FS Hz).  Uses nearest-neighbour to preserve
    integer class values.
    """
    orig_len = len(label)
    if orig_len == target_len:
        return label
    idx = np.round(np.linspace(0, orig_len - 1, target_len)).astype(int)
    return label[idx]


def _binarize(label: np.ndarray) -> np.ndarray:
    """Map WESAD labels to binary: stress=1, non-stress=1 (others), undefined=0."""
    binary = np.where(label == STRESS_LABEL, 1, label)
    # Mark all non-zero non-stress classes as 0 (non-stress)
    binary = np.where((binary != 1) & (binary != 0), 0, binary)
    # IMPORTANT: keep original 0 (undefined/transient) as 0 so windowing can skip them
    # but re-map baseline(1), amusement(3), meditation(4) → non-stress label 2
    # Then windowing skips windows where all labels == 0 (undefined)
    # Strategy: non-stress defined = baseline(1), amusement(3), meditation(4) → label 2
    binary = np.where(label == STRESS_LABEL, 1, 0)                     # stress → 1
    non_stress_classes = [1, 3, 4]
    for c in non_stress_classes:
        binary = np.where(label == c, 2, binary)                        # non-stress → 2
    # Now binary: 0=undefined, 1=stress, 2=non-stress
    return binary.astype(np.int32)


def process_subject(pkl_path: Path,
                    target_fs: int = TARGET_FS,
                    window: int = WINDOW,
                    step: int = STEP) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Full processing pipeline for one WESAD subject.

    Returns:
        X_wins: [N, 4, window]  float32
        y_wins: [N]             int64  (0=undefined/skipped, 1=stress, 2=non-stress)
                                after windowing: 1=stress, 0=non-stress
        subject_id: str
    """
    subject_id = pkl_path.stem   # e.g. "S2"
    data = _load_subject(pkl_path)
    if data is None:
        return np.empty((0, 4, window)), np.empty(0, dtype=np.int64), subject_id

    signals, label = _extract_wrist(data)

    # Compute ACC magnitude before resampling
    acc_mag = acc_magnitude(signals.pop("ACC"))   # [T]
    signals["ACC"] = acc_mag

    # Resample everything to TARGET_FS
    resampled = resample_signals(signals, NATIVE_FS, target_fs)

    # Align label to resampled length (use EDA length as reference — slowest signal)
    ref_len = len(resampled["EDA"])
    binary_label = _binarize(label)
    aligned_label = _align_label(binary_label, ref_len)

    # Z-normalize each channel independently
    for ch in resampled:
        resampled[ch] = znorm_per_subject(resampled[ch])

    # Stack channels [4, T]
    X = build_channel_matrix(resampled, ["EDA", "BVP", "TEMP", "ACC"])

    # Sliding window — skip undefined (label==0) windows
    # Remap: 1=stress→1, 2=non-stress→0 for binary classification
    remap = np.where(aligned_label == 1, 1,
                     np.where(aligned_label == 2, 0, -1))  # -1 = undefined

    # Replace undefined with a sentinel so windowing can filter
    y_for_window = np.where(remap == -1, 0, remap).astype(np.int32)
    # Build a validity mask so we skip windows with >50% undefined samples
    valid_mask = (remap != -1).astype(np.int32)

    X_wins, y_wins = _window_with_validity(X, y_for_window, valid_mask, window, step)
    return X_wins, y_wins, subject_id


def _window_with_validity(X, y, valid_mask, window, step):
    """Sliding window that skips windows where <50% samples are valid."""
    C, T = X.shape
    starts = range(0, T - window + 1, step)
    X_out, y_out = [], []

    for s in starts:
        x_w = X[:, s:s + window]
        y_w = y[s:s + window]
        v_w = valid_mask[s:s + window]

        if v_w.mean() < 0.5:   # less than 50% valid → skip
            continue

        # Majority vote over valid samples only
        valid_labels = y_w[v_w == 1]
        if len(valid_labels) == 0:
            continue
        label = int(np.bincount(valid_labels.astype(int)).argmax())

        X_out.append(x_w)
        y_out.append(label)

    if not X_out:
        return np.empty((0, C, window), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(X_out).astype(np.float32), np.array(y_out, dtype=np.int64)


def load_subject_raw(pkl_path: Path,
                     target_fs: int = TARGET_FS) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load one WESAD subject and return resampled, z-normed channels WITHOUT windowing.

    Returns:
        channels: np.ndarray [4, T]  — [EDA, BVP, TEMP, ACC_mag] at target_fs
        labels:   np.ndarray [T]     — binary labels (1=stress, 0=non-stress, -1=undefined)
        or None on load failure.
    """
    data = _load_subject(pkl_path)
    if data is None:
        return None

    signals, label = _extract_wrist(data)
    acc_mag = acc_magnitude(signals.pop("ACC"))
    signals["ACC"] = acc_mag
    resampled = resample_signals(signals, NATIVE_FS, target_fs)

    ref_len = len(resampled["EDA"])
    binary_label = _binarize(label)
    aligned_label = _align_label(binary_label, ref_len)

    for ch in resampled:
        resampled[ch] = znorm_per_subject(resampled[ch])

    X = build_channel_matrix(resampled, ["EDA", "BVP", "TEMP", "ACC"])

    # Remap to binary with -1 for undefined
    remap = np.where(aligned_label == 1, 1,
                     np.where(aligned_label == 2, 0, -1)).astype(np.int32)
    return X, remap


def load_all_subjects(wesad_root: str,
                      processed_dir: str,
                      force_reprocess: bool = False) -> dict[str, tuple]:
    """
    Process all WESAD subjects and save/load .npz files.

    Returns:
        dict: {subject_id: (X [N,4,1920], y [N])}
    """
    wesad_root    = Path(wesad_root)
    processed_dir = Path(processed_dir) / "wesad"
    processed_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(wesad_root.glob("S*/S*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No WESAD .pkl files found in {wesad_root}")

    all_data = {}
    print(f"Processing {len(pkl_files)} WESAD subjects...")

    for pkl_path in tqdm(pkl_files, desc="WESAD"):
        subject_id = pkl_path.stem
        out_path   = processed_dir / f"{subject_id}_windows.npz"

        if out_path.exists() and not force_reprocess:
            arr = np.load(out_path)
            all_data[subject_id] = (arr["X"], arr["y"])
            continue

        X, y, sid = process_subject(pkl_path)
        if X.shape[0] == 0:
            print(f"  [SKIP] {sid}: no valid windows")
            continue

        np.savez_compressed(out_path, X=X, y=y)
        all_data[subject_id] = (X, y)
        print(f"  {sid}: {X.shape[0]} windows | "
              f"stress={y.sum()} ({100*y.mean():.1f}%)")

    return all_data


def get_loso_splits(all_data: dict[str, tuple]) -> list[dict]:
    """
    Generate Leave-One-Subject-Out splits.

    Returns:
        list of dicts: [{'test_subject': sid,
                          'X_train': ..., 'y_train': ...,
                          'X_test':  ..., 'y_test':  ...}, ...]
    """
    subjects = sorted(all_data.keys())
    splits = []

    for test_sid in subjects:
        X_test, y_test = all_data[test_sid]
        train_X, train_y = [], []

        for train_sid in subjects:
            if train_sid == test_sid:
                continue
            X_tr, y_tr = all_data[train_sid]
            train_X.append(X_tr)
            train_y.append(y_tr)

        splits.append({
            "test_subject": test_sid,
            "X_train": np.concatenate(train_X, axis=0),
            "y_train": np.concatenate(train_y, axis=0),
            "X_test":  X_test,
            "y_test":  y_test,
        })

    return splits
