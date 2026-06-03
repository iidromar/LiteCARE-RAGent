"""
Script 03 — SWELL Preprocessing
=================================
Loads minute-level physiology features (HR, RMSSD, SCL) from SWELL-KW,
creates 5-minute statistical feature windows, and saves .npz for cross-dataset
evaluation with the Random Forest baseline.

Usage:
    python scripts/03_preprocess_swell.py [--config config/config.yaml]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.preprocessing.swell_loader import load_swell_features

def main():
    parser = argparse.ArgumentParser(description="SWELL feature preprocessing")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    swell_root   = cfg["data"]["swell_root"]
    processed_dir = cfg["data"]["processed_dir"]

    print(f"SWELL root    : {swell_root}")
    print(f"Processed dir : {processed_dir}")
    print()

    X, y, meta = load_swell_features(swell_root, processed_dir, force_reprocess=args.force)

    print(f"\nDone!")
    print(f"Feature matrix shape : {X.shape}")
    print(f"Labels shape         : {y.shape}")
    print(f"Stress windows       : {y.sum()} ({100*y.mean():.1f}%)")
    print(f"Participants         : {len(set(meta[:,0]))}")

if __name__ == "__main__":
    main()
