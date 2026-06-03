"""Script 42 — EmpaticaE4Stress Preprocessing"""
from __future__ import annotations
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.empatica_e4stress_loader import load_all_subjects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",
                        default="",
                        help="Path to EmpaticaE4Stress root directory")
    parser.add_argument("--out",
                        default="data/processed",
                        help="Base processed-data directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if .npz files already exist")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Dataset root not found: {root}")
        print("Please download from: https://data.mendeley.com/datasets/kb42z77m2g/2")
        print("Extract and point --root to the folder containing subject_01/, subject_02/, …")
        sys.exit(1)

    print(f"Processing EmpaticaE4Stress from: {root}")
    data = load_all_subjects(root, args.out, force_reprocess=args.force)

        print("\n" + "=" * 55)
    print("EMPATICA E4STRESS PREPROCESSING SUMMARY")
    print("=" * 55)
    all_y = []
    total_windows = 0
    for subj_id, (X, y) in sorted(data.items()):
        stress_rate = y.mean()
        print(f"  {subj_id:15s}: {X.shape[0]:4d} windows  "
              f"stress={stress_rate:.2f}  shape={X.shape}")
        all_y.append(y)
        total_windows += len(y)

    if all_y:
        all_y_concat = np.concatenate(all_y)
        print(f"\n  Total windows: {total_windows}")
        print(f"  Overall stress rate: {all_y_concat.mean():.3f}")
        print(f"  Subjects loaded: {len(data)}")
    print("=" * 55)

if __name__ == "__main__":
    main()
