"""
Script 05 — Cross-Dataset Evaluation
======================================
Evaluates generalisation of the best WESAD-trained model on:
  1. AffectiveROAD (raw time-series, same TCN model)
  2. SWELL-KW (feature-level, Random Forest + comparison)

Usage:
    python scripts/05_cross_dataset_eval.py [--config config/config.yaml]
                                            [--model_dir results/wesad_loso/full]
                                            [--device cpu]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.preprocessing.wesad_loader import load_all_subjects
from src.preprocessing.affectiveroad_loader import load_all_drives
from src.preprocessing.swell_loader import load_swell_features
from src.evaluation.cross_eval import run_all_cross_evals

def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--model_dir", default="results/wesad_loso/full",
                        help="Directory containing saved model checkpoints")
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--out_dir",   default="results/cross_dataset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    wesad_data = load_all_subjects(cfg["data"]["wesad_root"], cfg["data"]["processed_dir"])

    affectiveroad_data = {}
    try:
        affectiveroad_data = load_all_drives(
            cfg["data"]["affectiveroad_root"], cfg["data"]["processed_dir"]
        )
    except Exception as e:
        print(f"[WARNING] AffectiveROAD loading failed: {e}")

    swell_X, swell_y, swell_meta = None, None, None
    try:
        swell_X, swell_y, swell_meta = load_swell_features(
            cfg["data"]["swell_root"], cfg["data"]["processed_dir"]
        )
    except Exception as e:
        print(f"[WARNING] SWELL loading failed: {e}")

    print("\nRunning cross-dataset evaluations...")
    results_df = run_all_cross_evals(
        wesad_data=wesad_data,
        affectiveroad_data=affectiveroad_data,
        swell_feats=(swell_X, swell_y, swell_meta) if swell_X is not None else None,
        cfg=cfg,
        out_dir=out_dir,
    )

    print("\n=== Cross-Dataset Summary ===")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
