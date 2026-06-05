"""Script 06 — Ablation Study"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.preprocessing.wesad_loader import load_all_subjects
from src.evaluation.ablation import run_ablation_study
from src.utils.visualization import plot_ablation_bars

def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--out_dir", default="results/ablation")
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WESAD data...")
    all_data = load_all_subjects(cfg["data"]["wesad_root"], cfg["data"]["processed_dir"])

    print("\nRunning ablation study (5 variants × LOSO)...")
    summary = run_ablation_study(
        all_data=all_data,
        cfg=cfg,
        out_dir=out_dir,
        device=args.device,
    )

    # Plot ablation bar chart
    plot_ablation_bars(
        ablation_results=summary,
        metric="f1",
        title="Ablation Study — F1 Score",
        out_path=out_dir / "ablation_f1.png",
    )

    print("\n=== Ablation Summary (F1) ===")
    for variant, metrics in summary.items():
        f1   = metrics.get("f1",     0)
        std  = metrics.get("f1_std", 0)
        auroc = metrics.get("auroc", 0)
        print(f"  {variant:20s}  F1={f1:.4f} ± {std:.4f}  AUROC={auroc:.4f}")

if __name__ == "__main__":
    main()
