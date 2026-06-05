"""
Script 09 — Full Thesis Evaluation Table
==========================================
Produces the complete evaluation table with ALL metrics:
  - Accuracy, Precision, Recall, F1, AUROC, ECE
  - Abstention Rate (MC-Dropout, TCN only)
  - Model Size (MB), Inference Time (ms)
  - Comparison: TCN v3 Full vs RF, BiLSTM, DilatedCNN baselines

Usage:
    python scripts/09_full_evaluation.py [--device cpu]
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.preprocessing.wesad_loader import load_all_subjects
from src.models.lite_tcn_se import build_model
from src.models.baselines import (
    build_rf_pipeline, extract_statistical_features,
    BiLSTMClassifier, DilatedCNNClassifier,
)
from src.models.mc_inference import mc_predict
from src.training.losses import CompositeLoss, compute_class_weight
from src.training.train import evaluate
from src.evaluation.metrics import (
    compute_metrics, measure_inference_time, compute_abstention_metrics
)

BASE = Path("")

def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)

def run_bilstm_loso(all_data: dict, cfg: dict, out_dir: Path,
                    device: str = "cpu") -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "loso_results.csv"
    if cache.exists():
        print("  [CACHE] BiLSTM results loaded.")
        return pd.read_csv(cache).to_dict("records")

    subjects = sorted(all_data.keys())
    results  = []
    tcfg     = cfg["training"]

    print(f"\nBiLSTM LOSO — {len(subjects)} folds")
    for i, test_sid in enumerate(subjects):
        val_sid    = subjects[(i - 1) % len(subjects)]
        train_sids = [s for s in subjects if s != test_sid and s != val_sid]

        X_tr = np.concatenate([all_data[s][0] for s in train_sids])
        y_tr = np.concatenate([all_data[s][1] for s in train_sids])
        X_val, y_val = all_data[val_sid]
        X_te,  y_te  = all_data[test_sid]

        model = BiLSTMClassifier(
            input_size=cfg["model"]["input_channels"],
            hidden_size=128, num_layers=2,
            dropout=cfg["model"]["dropout_rate"],
        ).to(device)

        class_wt  = compute_class_weight(y_tr).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_wt)
        optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])

        # DataLoaders — BiLSTM takes [B, C, T] same as TCN (transposes internally)
        tr_ds  = TensorDataset(torch.tensor(X_tr,  dtype=torch.float32),
                               torch.tensor(y_tr,  dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
        tr_ld  = DataLoader(tr_ds,  batch_size=tcfg["batch_size"], shuffle=True,  drop_last=True)
        val_ld = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

        best_f1, best_state, patience_cnt = 0.0, None, 0
        patience = min(tcfg["early_stopping_patience"], 10)   # baselines converge fast
        max_epochs = min(tcfg["max_epochs"], 50)              # cap at 50 for baselines

        for epoch in range(1, max_epochs + 1):
            model.train()
            for X_b, y_b in tr_ld:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            all_p, all_t, all_pr = [], [], []
            with torch.no_grad():
                for X_b, y_b in val_ld:
                    logits = model(X_b.to(device))
                    probs  = torch.softmax(logits, -1).cpu().numpy()
                    all_pr.extend(probs)
                    all_p.extend(probs.argmax(-1))
                    all_t.extend(y_b.numpy())
            val_f1 = compute_metrics(np.array(all_t), np.array(all_p))["f1"]

            if val_f1 > best_f1:
                best_f1, best_state, patience_cnt = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        # Test
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        all_p, all_t, all_pr = [], [], []
        te_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                              torch.tensor(y_te, dtype=torch.long))
        with torch.no_grad():
            for X_b, y_b in DataLoader(te_ds, batch_size=tcfg["batch_size"]):
                logits = model(X_b.to(device))
                probs  = torch.softmax(logits, -1).cpu().numpy()
                all_pr.extend(probs)
                all_p.extend(probs.argmax(-1))
                all_t.extend(y_b.numpy())

        m = compute_metrics(np.array(all_t), np.array(all_p), np.array(all_pr))
        m["test_subject"] = test_sid
        results.append(m)
        print(f"  [{i+1:2d}/{len(subjects)}] {test_sid}: F1={m['f1']:.4f} AUROC={m.get('auroc',0):.4f}")

    pd.DataFrame(results).to_csv(cache, index=False)
    return results

def run_dilatedcnn_loso(all_data: dict, cfg: dict, out_dir: Path,
                        device: str = "cpu") -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "loso_results.csv"
    if cache.exists():
        print("  [CACHE] DilatedCNN results loaded.")
        return pd.read_csv(cache).to_dict("records")

    subjects = sorted(all_data.keys())
    results  = []
    tcfg     = cfg["training"]
    mcfg     = cfg["model"]

    print(f"\nDilatedCNN LOSO — {len(subjects)} folds")
    for i, test_sid in enumerate(subjects):
        val_sid    = subjects[(i - 1) % len(subjects)]
        train_sids = [s for s in subjects if s != test_sid and s != val_sid]

        X_tr = np.concatenate([all_data[s][0] for s in train_sids])
        y_tr = np.concatenate([all_data[s][1] for s in train_sids])
        X_val, y_val = all_data[val_sid]
        X_te,  y_te  = all_data[test_sid]

        model = DilatedCNNClassifier(
            input_channels=mcfg["input_channels"],
            channels_per_layer=mcfg["channels_per_layer"],
            dilation_schedule=mcfg["dilation_schedule"],
            kernel_size=mcfg["kernel_size"],
            dropout=mcfg["dropout_rate"],
        ).to(device)

        class_wt  = compute_class_weight(y_tr).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_wt)
        optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])

        tr_ds  = TensorDataset(torch.tensor(X_tr,  dtype=torch.float32),
                               torch.tensor(y_tr,  dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
        tr_ld  = DataLoader(tr_ds,  batch_size=tcfg["batch_size"], shuffle=True,  drop_last=True)
        val_ld = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

        best_f1, best_state, patience_cnt = 0.0, None, 0
        patience   = min(tcfg["early_stopping_patience"], 10)
        max_epochs = min(tcfg["max_epochs"], 50)

        for epoch in range(1, max_epochs + 1):
            model.train()
            for X_b, y_b in tr_ld:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for X_b, y_b in val_ld:
                    logits = model(X_b.to(device))
                    all_p.extend(logits.argmax(-1).cpu().numpy())
                    all_t.extend(y_b.numpy())
            val_f1 = compute_metrics(np.array(all_t), np.array(all_p))["f1"]

            if val_f1 > best_f1:
                best_f1, best_state, patience_cnt = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        all_p, all_t, all_pr = [], [], []
        te_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                              torch.tensor(y_te, dtype=torch.long))
        with torch.no_grad():
            for X_b, y_b in DataLoader(te_ds, batch_size=tcfg["batch_size"]):
                logits = model(X_b.to(device))
                probs  = torch.softmax(logits, -1).cpu().numpy()
                all_pr.extend(probs)
                all_p.extend(probs.argmax(-1))
                all_t.extend(y_b.numpy())

        m = compute_metrics(np.array(all_t), np.array(all_p), np.array(all_pr))
        m["test_subject"] = test_sid
        results.append(m)
        print(f"  [{i+1:2d}/{len(subjects)}] {test_sid}: F1={m['f1']:.4f} AUROC={m.get('auroc',0):.4f}")

    pd.DataFrame(results).to_csv(cache, index=False)
    return results

def compute_tcn_system_metrics(all_data: dict, cfg: dict, device: str) -> dict:
    """Model size, inference time, abstention rate for TCN v3 Full."""
    mcfg = cfg["model"]

    # Load first available checkpoint
    ckpt_dir = BASE / "results/wesad_loso_v3/full"
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        return {}

    from src.preprocessing.hrv_features import N_HRV_FEATURES
    model = build_model(
        variant="full",
        input_channels=mcfg["input_channels"],
        channels_per_layer=mcfg["channels_per_layer"],
        dilation_schedule=mcfg["dilation_schedule"],
        kernel_size=mcfg["kernel_size"],
        dropout_rate=mcfg["dropout_rate"],
        se_reduction=mcfg["se_reduction_ratio"],
        hrv_features=N_HRV_FEATURES,
    ).to(device)
    model.load_state_dict(torch.load(ckpts[0], map_location=device))

    size_mb   = model_size_mb(model)
    inf_time  = measure_inference_time(model, input_shape=(1, 4, 1920), device=device)

    # Abstention rate via MC-Dropout on all test subjects (concatenated)
    X_all = np.concatenate([v[0] for v in all_data.values()])
    y_all = np.concatenate([v[1] for v in all_data.values()])

    # Sample 500 windows for speed
    idx = np.random.choice(len(X_all), min(500, len(X_all)), replace=False)
    X_s, y_s = X_all[idx], y_all[idx]

    X_t = torch.tensor(X_s, dtype=torch.float32).to(device)
    result = mc_predict(model, X_t, n_samples=cfg["mc_dropout"]["n_samples"])
    tau    = float(np.percentile(result["uncertainty"], 95))
    abst   = compute_abstention_metrics(
        y_s, result["pred_class"],
        result["uncertainty"], tau
    )

    return {
        "model_size_mb":   round(size_mb, 3),
        "inference_ms":    round(inf_time, 2),
        "abstention_rate": round(abst["abstention_rate"], 4),
        "f1_after_abst":   round(abst.get("f1_after_abstention", float("nan")), 4),
        "n_params":        sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

def aggregate(records: list[dict], excl_s17: bool = False) -> dict:
    df = pd.DataFrame(records)
    if excl_s17 and "test_subject" in df.columns:
        df = df[df["test_subject"].astype(str) != "S17"]
    return {
        "accuracy":  f"{np.mean(df.accuracy):.4f}±{np.std(df.accuracy):.4f}",
        "precision": f"{np.nanmean(df.precision):.4f}±{np.nanstd(df.precision):.4f}",
        "recall":    f"{np.nanmean(df.recall):.4f}±{np.nanstd(df.recall):.4f}",
        "f1":        f"{np.mean(df.f1):.4f}±{np.std(df.f1):.4f}",
        "auroc":     f"{np.nanmean(df.auroc):.4f}±{np.nanstd(df.auroc):.4f}",
        "ece":       f"{np.nanmean(df.ece):.4f}±{np.nanstd(df.ece):.4f}",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--skip_nn", action="store_true",
                        help="Skip BiLSTM/DilatedCNN training (use cached)")
    args = parser.parse_args()

    cfg      = load_config(str(BASE / "config/config.yaml"))
    all_data = load_all_subjects(cfg["data"]["wesad_root"], cfg["data"]["processed_dir"])
    out_base = BASE / "results/baselines"

        tcn_v3   = pd.read_csv(BASE / "results/wesad_loso_v3/full/loso_results.csv").to_dict("records")
    no_se    = pd.read_csv(BASE / "results/wesad_loso_v3/no_se/loso_results.csv").to_dict("records")
    fix_dil  = pd.read_csv(BASE / "results/wesad_loso_v3/fixed_dilation/loso_results.csv").to_dict("records")
    rf_res   = pd.read_csv(BASE / "results/rf_loso_wesad/rf_loso_results.csv").to_dict("records")

        bilstm_res  = run_bilstm_loso(all_data, cfg, out_base / "bilstm",  args.device)
    dilcnn_res  = run_dilatedcnn_loso(all_data, cfg, out_base / "dilatedcnn", args.device)

        print("\nMeasuring TCN system metrics (size, speed, abstention)...")
    sys_m = compute_tcn_system_metrics(all_data, cfg, args.device)
    print(f"  Model size   : {sys_m.get('model_size_mb','?')} MB")
    print(f"  Parameters   : {sys_m.get('n_params','?'):,}")
    print(f"  Inference    : {sys_m.get('inference_ms','?')} ms")
    print(f"  Abstention   : {sys_m.get('abstention_rate','?')*100:.1f}%")
    print(f"  F1 post-abst : {sys_m.get('f1_after_abst','?')}")

        models = {
        "Random Forest":         rf_res,
        "BiLSTM":                bilstm_res,
        "DilatedCNN":            dilcnn_res,
        "TCN v3 Full (proposed)":tcn_v3,
        "  No-SE ablation":      no_se,
        "  Fixed-Dilation abla.":fix_dil,
    }

    for title, excl in [("ALL 15 SUBJECTS", False), ("EXCLUDING S17 (14 subjects)", True)]:
        print(f"\n{'='*100}")
        print(f"  {title}")
        print(f"{'='*100}")
        print(f"{'Model':<28} {'Accuracy':>14} {'Precision':>14} {'Recall':>14} {'F1':>14} {'AUROC':>12} {'ECE':>12}")
        print("-"*100)
        for label, records in models.items():
            if not records:
                print(f"{label:<28}  [not available]")
                continue
            a = aggregate(records, excl_s17=excl)
            print(f"{label:<28} {a['accuracy']:>14} {a['precision']:>14} "
                  f"{a['recall']:>14} {a['f1']:>14} {a['auroc']:>12} {a['ece']:>12}")

    print(f"\n{'='*60}")
    print(f"  SYSTEM METRICS — TCN v3 Full")
    print(f"{'='*60}")
    print(f"  Parameters    : {sys_m.get('n_params', '?'):,}")
    print(f"  Model size    : {sys_m.get('model_size_mb', '?')} MB  (budget: ≤5 MB)")
    print(f"  Inference     : {sys_m.get('inference_ms', '?')} ms  (budget: ≤100 ms)")
    print(f"  Abstention    : {sys_m.get('abstention_rate', '?')*100:.1f}%  (τ=95th pct entropy)")
    print(f"  F1 post-abst  : {sys_m.get('f1_after_abst', '?')}")

    # Save
    rows = []
    for label, records in models.items():
        if not records: continue
        a14 = aggregate(records, excl_s17=True)
        a15 = aggregate(records, excl_s17=False)
        rows.append({"model": label, **{f"{k}_all15": v for k, v in a15.items()},
                                      **{f"{k}_excl17": v for k, v in a14.items()}})
    pd.DataFrame(rows).to_csv(BASE / "results/full_evaluation_table.csv", index=False)
    print(f"\nSaved to results/full_evaluation_table.csv")

if __name__ == "__main__":
    main()
