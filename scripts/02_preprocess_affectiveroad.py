"""
Script 02 — AffectiveROAD Preprocessing
==========================================
Extracts E4 wristband signals from per-drive zip archives, aligns subjective
stress metric labels, binarises with 75th-percentile threshold, windows, and
saves processed .npz files.

Usage:
    python scripts/02_preprocess_affectiveroad.py [--config config/config.yaml]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.preprocessing.affectiveroad_loader import load_all_drives

def main():
    parser = argparse.ArgumentParser(description="AffectiveROAD preprocessing")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ar_root      = cfg["data"]["affectiveroad_root"]
    processed_dir = cfg["data"]["processed_dir"]

    print(f"AffectiveROAD root: {ar_root}")
    print(f"Processed dir     : {processed_dir}")
    print()

    all_data = load_all_drives(ar_root, processed_dir, force_reprocess=args.force)

    if not all_data:
        print("No drives processed. Check that AffectiveROAD_Data path is correct.")
        return

    print(f"\nDone! Processed {len(all_data)} drives.")
    total_windows = sum(v[0].shape[0] for v in all_data.values())
    total_stress  = sum(int(v[1].sum()) for v in all_data.values())
    print(f"Total windows : {total_windows}")
    print(f"Stress windows: {total_stress} ({100*total_stress/total_windows:.1f}%)")

if __name__ == "__main__":
    main()
