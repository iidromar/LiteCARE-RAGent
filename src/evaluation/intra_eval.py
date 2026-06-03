"""
Intra-Dataset Evaluation
=========================
LOSO cross-validation on each dataset individually.
Returns per-fold and aggregate metrics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from ..training.train import run_loso_cv
from .metrics import compute_metrics, measure_inference_time
from ..models.lite_tcn_se import LiteTCNSE


def evaluate_wesad_loso(all_data: dict,
                         cfg: dict,
                         out_dir: str,
                         variants: list[str] | None = None) -> pd.DataFrame:
    """
    Run LOSO CV for one or more model variants on WESAD.

    Args:
        all_data: {subject_id: (X, y)}
        cfg:      loaded config
        out_dir:  output directory for checkpoints and results
        variants: list of variant names to run (default: ['full'])

    Returns:
        DataFrame with per-fold and mean results
    """
    if variants is None:
        variants = ["full"]

    all_rows = []
    for variant in variants:
        variant_dir = Path(out_dir) / "wesad" / variant
        results = run_loso_cv(all_data, cfg, str(variant_dir), variant=variant)

        for r in results:
            r["variant"] = variant
            all_rows.append(r)

    df = pd.DataFrame(all_rows)

    # Add summary rows
    summary_rows = []
    for variant in variants:
        sub = df[df["variant"] == variant]
        row = {col: sub[col].mean() for col in sub.select_dtypes(include=float).columns}
        row["variant"]      = variant
        row["test_subject"] = "MEAN"
        summary_rows.append(row)

    df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    # Save
    out_path = Path(out_dir) / "wesad_loso_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    return df
