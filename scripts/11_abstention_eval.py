"""
Script 11 — MC-Dropout Abstention Evaluation
=============================================
Loads trained per-fold TCN v3 checkpoints, calibrates τ per fold on the
validation subject, then computes abstention rate and F1-after-abstention
on the test subject.

This properly wires the τ calibration pipeline from mc_inference.py into
the LOSO evaluation, computing Ablation 2 (no-MC deterministic comparison).

Usage:
    python scripts/11_abstention_eval.py [--device cpu] [--n_samples 30]
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.preprocessing.wesad_loader import load_all_subjects
from src.models.lite_tcn_se import build_model
from src.models.mc_inference import mc_predict, calibrate_threshold
from src.evaluation.metrics import compute_metrics, compute_abstention_metrics

try:
    from src.preprocessing.hrv_features import N_HRV_FEATURES, extract_hrv_batch
    HAS_HRV = True
except ImportError:
    HAS_HRV = False
    N_HRV_FEATURES = 0

BASE = Path("")

def evaluate_abstention_loso(all_data: dict, cfg: dict,
                              ckpt_dir: Path, device: str,
                              n_samples: int = 30,
                              variant: str = "full",
                              out_suffix: str = "") -> pd.DataFrame:
    """
    For each fold: load checkpoint → calibrate τ on val → evaluate on test.
    """
    mcfg     = cfg["model"]
    subjects = sorted(all_data.keys())
    use_hrv  = mcfg.get("use_hrv", False)
    results  = []

    print(f"\nAbstention LOSO — {len(subjects)} folds | n_samples={n_samples} | variant={variant}")

    for i, test_sid in enumerate(subjects):
        val_sid    = subjects[(i - 1) % len(subjects)]
        train_sids = [s for s in subjects if s != test_sid and s != val_sid]

        ckpt_path = ckpt_dir / f"{variant}_fold_{test_sid}.pt"
        if not ckpt_path.exists():
            print(f"  [MISSING] {ckpt_path} — skipping fold")
            continue

        # Load model
        model = build_model(
            variant=variant,
            input_channels=mcfg["input_channels"],
            channels_per_layer=mcfg["channels_per_layer"],
            dilation_schedule=mcfg["dilation_schedule"],
            kernel_size=mcfg["kernel_size"],
            dropout_rate=mcfg["dropout_rate"],
            se_reduction=mcfg["se_reduction_ratio"],
            hrv_features=N_HRV_FEATURES if use_hrv else 0,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        X_val, y_val = all_data[val_sid]
        X_te,  y_te  = all_data[test_sid]
        X_tr = np.concatenate([all_data[s][0] for s in train_sids])
        y_tr = np.concatenate([all_data[s][1] for s in train_sids])

        # HRV features (if enabled)
        hrv_val = hrv_te = None
        if use_hrv and HAS_HRV:
            fs      = cfg["data"].get("target_fs", 32)
            bvp_ch  = cfg["data"].get("bvp_channel_idx", 1)
            hrv_tr  = extract_hrv_batch(X_tr,  bvp_channel=bvp_ch, fs=fs)
            hrv_val = extract_hrv_batch(X_val, bvp_channel=bvp_ch, fs=fs)
            hrv_te  = extract_hrv_batch(X_te,  bvp_channel=bvp_ch, fs=fs)
            hrv_mean = hrv_tr.mean(0, keepdims=True)
            hrv_std  = hrv_tr.std(0,  keepdims=True) + 1e-8
            hrv_val  = (hrv_val - hrv_mean) / hrv_std
            hrv_te   = (hrv_te  - hrv_mean) / hrv_std

                tau = calibrate_threshold(
            model, X_val, y_val,
            hrv_val=hrv_val,
            n_samples=n_samples,
            percentile=95.0,
            device=device,
        )

                X_te_t   = torch.tensor(X_te,  dtype=torch.float32)
        hrv_te_t = torch.tensor(hrv_te, dtype=torch.float32) if hrv_te is not None else None
        result   = mc_predict(model, X_te_t, n_samples=n_samples, device=device, hrv=hrv_te_t)
        y_pred  = result["pred_class"]
        y_prob  = result["mean_probs"]
        uncerts = result["uncertainty"]

        # Standard metrics (no abstention)
        std_m = compute_metrics(y_te, y_pred, y_prob)

        # Abstention metrics
        abst_m = compute_abstention_metrics(y_te, y_pred, uncerts, tau)

        row = {
            "test_subject":          test_sid,
            "tau":                   round(float(tau), 4),
            "f1":                    round(std_m["f1"], 4),
            "accuracy":              round(std_m["accuracy"], 4),
            "auroc":                 round(std_m.get("auroc", float("nan")), 4),
            "ece":                   round(std_m.get("ece",   float("nan")), 4),
            **{k: round(float(v), 4) if isinstance(v, float) else v
               for k, v in abst_m.items()},
        }
        results.append(row)
        print(f"  [{i+1:2d}/{len(subjects)}] {test_sid}: "
              f"F1={row['f1']:.4f}  τ={row['tau']:.4f}  "
              f"AbstRate={abst_m['abstention_rate']:.3f}  "
              f"F1@abst={abst_m.get('f1_after_abstention', float('nan')):.4f}")

    df = pd.DataFrame(results)

    # Separate metrics excl. S17
    mask_no17 = df["test_subject"].astype(str) != "S17"

    print("\n" + "="*80)
    print(f"MC-DROPOUT ABSTENTION RESULTS — TCN v3 {variant}, LOSO on WESAD")
    print("="*80)
    print(f"  F1 (all):          {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
    print(f"  F1 (excl S17):     {df.loc[mask_no17,'f1'].mean():.4f} ± {df.loc[mask_no17,'f1'].std():.4f}")
    print(f"  Mean τ:            {df['tau'].mean():.4f} ± {df['tau'].std():.4f}")
    print(f"  Abstention rate:   {df['abstention_rate'].mean():.4f} ± {df['abstention_rate'].std():.4f}")
    if "f1_after_abstention" in df.columns:
        print(f"  F1 after abstain:  {df['f1_after_abstention'].mean():.4f} ± {df['f1_after_abstention'].std():.4f}")
    print("="*80)

    out_path = BASE / f"results/abstention_loso_{variant}{out_suffix}_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--variant",   default="full",
                        choices=["full", "no_se", "fixed_dilation", "tap", "full"])
    parser.add_argument("--version",   default="v3",
                        help="Model version folder, e.g. v3 or v5")
    parser.add_argument("--config",    default="config/config.yaml",
                        help="Config file to use (e.g. config/config_v8b.yaml)")
    args = parser.parse_args()

    cfg      = load_config(str(BASE / args.config))
    all_data = load_all_subjects(cfg["data"]["wesad_root"], cfg["data"]["processed_dir"])
    ckpt_dir = BASE / f"results/wesad_loso_{args.version}/{args.variant}"
    out_suffix = f"_{args.version}" if args.version != "v3" else ""

    evaluate_abstention_loso(all_data, cfg, ckpt_dir,
                              device=args.device, n_samples=args.n_samples,
                              variant=args.variant, out_suffix=out_suffix)

if __name__ == "__main__":
    main()
