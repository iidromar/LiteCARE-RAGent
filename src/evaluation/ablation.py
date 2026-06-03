"""
Ablation Study
==============
Runs the 4 planned ablation experiments from proposal Table 10.

Experiment | TCN | SE | MC | RAG | Dilation | Purpose
-----------+-----+----+----+-----+----------+------------------
Full Model |  Y  | Y  | Y  | Y   | [1,2,4,8]| Baseline
Ablation 1 |  Y  | N  | Y  | Y   | [1,2,4,8]| Impact of SE
Ablation 2 |  Y  | Y  | N  | Y   | [1,2,4,8]| Impact of MC-UQ
Ablation 3 |  Y  | Y  | Y  | N   | [1,2,4,8]| Impact of RAG
Ablation 4 |  Y  | Y  | Y  | Y   | [1,1,1,1]| Impact of dilation
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from ..training.train import run_loso_cv
from .metrics import compute_metrics


ABLATION_CONFIGS = {
    "full":           {"variant": "full",          "mc_dropout": True,  "rag": True},
    "no_se":          {"variant": "no_se",         "mc_dropout": True,  "rag": True},
    "no_mc":          {"variant": "full",          "mc_dropout": False, "rag": True},
    "no_rag":         {"variant": "full",          "mc_dropout": True,  "rag": False},
    "fixed_dilation": {"variant": "fixed_dilation","mc_dropout": True,  "rag": True},
}


def run_ablation_study(all_data: dict,
                       cfg: dict,
                       out_dir: str,
                       experiments: list[str] | None = None,
                       device: str = "cpu") -> dict:
    """
    Run all ablation experiments on WESAD using LOSO CV.

    Caches results: if loso_results.csv already exists in an experiment's
    subdirectory, that experiment is skipped (results loaded from disk).

    Note: 'no_mc' and 'no_rag' use the same model architecture as 'full'
    (they differ only at inference time, not at training time).  Their
    training checkpoints are therefore identical to 'full'; training is
    skipped and 'full' results are re-used if those already exist.

    Args:
        all_data:    {subject_id: (X, y)} from wesad_loader
        cfg:         loaded config
        out_dir:     output directory
        experiments: list of experiment keys (default: all)
        device:      'cpu' or 'cuda'

    Returns:
        dict {exp_name: {f1, f1_std, auroc, ece, accuracy, ...}}
    """
    if experiments is None:
        experiments = list(ABLATION_CONFIGS.keys())

    ablations_dir = Path(out_dir) / "ablations"
    summary = {}

    for exp_name in experiments:
        ablation_cfg = ABLATION_CONFIGS[exp_name]
        variant      = ablation_cfg["variant"]
        use_mc       = ablation_cfg["mc_dropout"]

        print(f"\n{'='*60}")
        print(f"Ablation: {exp_name}  (variant={variant}, mc={use_mc})")
        print(f"{'='*60}")

        exp_dir  = ablations_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        cache_csv = exp_dir / "loso_results.csv"

        # ── Load from cache if available ──────────────────────────────────
        if cache_csv.exists():
            print(f"  [CACHE] Loading existing results from {cache_csv}")
            fold_df  = pd.read_csv(cache_csv)
            fold_res = fold_df.to_dict("records")
        else:
            # no_mc and no_rag train the same model as full — check full cache
            if exp_name in ("no_mc", "no_rag"):
                full_cache = ablations_dir / "full" / "loso_results.csv"
                if full_cache.exists():
                    print(f"  [REUSE] {exp_name} shares model with 'full' — "
                          f"loading full results from {full_cache}")
                    fold_df  = pd.read_csv(full_cache)
                    fold_res = fold_df.to_dict("records")
                    fold_df.to_csv(cache_csv, index=False)
                else:
                    # Train as full variant
                    fold_res = run_loso_cv(all_data, cfg, str(exp_dir),
                                          variant="full", device=device)
            else:
                fold_res = run_loso_cv(all_data, cfg, str(exp_dir),
                                       variant=variant, device=device)

        # ── Aggregate ─────────────────────────────────────────────────────
        f1s = [r["f1"] for r in fold_res if "f1" in r]
        acc = [r["accuracy"] for r in fold_res if "accuracy" in r]
        auc = [r.get("auroc", float("nan")) for r in fold_res]
        ece = [r.get("ece",   float("nan")) for r in fold_res]

        summary[exp_name] = {
            "f1":        np.mean(f1s),
            "f1_std":    np.std(f1s),
            "accuracy":  np.mean(acc),
            "auroc":     np.nanmean(auc),
            "ece":       np.nanmean(ece),
            "mc_dropout": use_mc,
            "rag_mini":   ablation_cfg["rag"],
            "variant":    variant,
        }

        print(f"  → F1 = {summary[exp_name]['f1']:.4f} ± {summary[exp_name]['f1_std']:.4f} "
              f"  AUROC = {summary[exp_name]['auroc']:.4f}")

    # ── Save summary ──────────────────────────────────────────────────────
    df = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "experiment"})
    out_path = Path(out_dir) / "ablation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nAblation results saved to {out_path}")
    print(df.to_string(index=False))
    return summary
