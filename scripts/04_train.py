"""
Script 04 — Train Lite-TCN-SE + LOSO Evaluation on WESAD
===========================================================
Runs Leave-One-Subject-Out cross-validation on WESAD and saves:
  - Best model checkpoint per fold
  - Per-fold metrics CSV
  - Aggregated summary CSV

Usage:
    python scripts/04_train.py [--config config/config.yaml] [--variant full]
                               [--device cpu] [--out_dir results/wesad_loso]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.preprocessing.wesad_loader import load_all_subjects
from src.training.train import run_loso_cv

def main():
    parser = argparse.ArgumentParser(description="WESAD LOSO training")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--variant", default="full",
                        choices=["full", "no_se", "fixed_dilation", "tap", "full"],
                        help="Model variant for ablation")
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--out_dir", default="results/wesad_loso")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) / args.variant

    print(f"Variant   : {args.variant}")
    print(f"Device    : {args.device}")
    print(f"Output dir: {out_dir}")
    print()

    # Load data (uses cache if available)
    print("Loading WESAD data...")
    all_data = load_all_subjects(cfg["data"]["wesad_root"], cfg["data"]["processed_dir"])

    print(f"Running LOSO CV on {len(all_data)} subjects...\n")
    results = run_loso_cv(
        all_data=all_data,
        cfg=cfg,
        out_dir=out_dir,
        variant=args.variant,
        device=args.device,
    )

    print("\n=== Final Results ===")
    import numpy as np
    for metric in ["accuracy", "f1", "auroc", "ece"]:
        vals = [r[metric] for r in results if metric in r]
        if vals:
            print(f"  {metric:10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

if __name__ == "__main__":
    main()
