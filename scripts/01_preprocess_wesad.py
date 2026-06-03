"""
Script 01 — WESAD Preprocessing
=================================
Loads raw WESAD .pkl files, extracts wrist E4 signals, resamples to 32 Hz,
applies sliding window, and saves processed .npz files.

Usage:
    python scripts/01_preprocess_wesad.py [--config config/config.yaml]
"""
import argparse
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.preprocessing.wesad_loader import load_all_subjects

def main():
    parser = argparse.ArgumentParser(description="WESAD preprocessing")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--force", action="store_true", help="Re-process even if cache exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wesad_root   = cfg["data"]["wesad_root"]
    processed_dir = cfg["data"]["processed_dir"]

    print(f"WESAD root    : {wesad_root}")
    print(f"Processed dir : {processed_dir}")
    print()

    all_data = load_all_subjects(wesad_root, processed_dir, force_reprocess=args.force)

    print(f"\nDone! Processed {len(all_data)} subjects.")
    total_windows = sum(v[0].shape[0] for v in all_data.values())
    total_stress  = sum(int(v[1].sum()) for v in all_data.values())
    print(f"Total windows : {total_windows}")
    print(f"Stress windows: {total_stress} ({100*total_stress/total_windows:.1f}%)")

if __name__ == "__main__":
    main()
