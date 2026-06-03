"""Script 43 — Cross-Dataset Evaluation: WESAD → EmpaticaE4Stress"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.lite_tcn_se import LiteTCNSE
from src.models.baselines import BiLSTMClassifier, DilatedCNNClassifier
from src.preprocessing.empatica_e4stress_loader import load_all_subjects
from src.evaluation.metrics import compute_metrics

try:
    from src.preprocessing.hrv_features import extract_hrv_batch, N_HRV_FEATURES
    HAS_HRV = True
except ImportError:
    HAS_HRV = False
    N_HRV_FEATURES = 0

BASE     = Path(__file__).parent.parent
CKPT_DIR = BASE / "results/wesad_loso/v8"
OUT_DIR  = BASE / "results/cross_dataset_e4stress"
CONFIG   = BASE / "config/config_multidomain.yaml"

def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """
    Extract 16-dim statistical feature vector per window.
    X: (N, 4, T) → features: (N, 16) = [mean, std, min, max] × 4 channels
    """
    feats = []
    for c in range(X.shape[1]):
        ch = X[:, c, :]
        feats.append(ch.mean(axis=1, keepdims=True))
        feats.append(ch.std(axis=1, keepdims=True))
        feats.append(ch.min(axis=1, keepdims=True))
        feats.append(ch.max(axis=1, keepdims=True))
    return np.concatenate(feats, axis=1).astype(np.float32)

def average_wesad_checkpoints(cfg: dict, hrv_feat: int, device: str):
    """Load all 15 WESAD fold checkpoints and average their weights."""
    mcfg = cfg["model"]
    model = LiteTCNSE(
        input_channels=mcfg["input_channels"],
        num_classes=mcfg["num_classes"],
        channels_per_layer=mcfg["channels_per_layer"],
        dilation_schedule=mcfg["dilation_schedule"],
        kernel_size=mcfg["kernel_size"],
        dropout_rate=mcfg["dropout_rate"],
        se_reduction=mcfg.get("se_reduction_ratio", 4),
        hrv_features=hrv_feat,
    ).to(device)

    ckpt_paths = sorted(CKPT_DIR.glob("v8_fold_*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {CKPT_DIR}")

    avg_state = None
    for p in ckpt_paths:
        s = torch.load(p, map_location=device)
        if avg_state is None:
            avg_state = {k: s[k].clone().float() for k in s}
        else:
            avg_state = {k: avg_state[k] + s[k].float() for k in avg_state}
    for k in avg_state:
        avg_state[k] /= len(ckpt_paths)

    model.load_state_dict(avg_state)
    model.eval()
    print(f"  Averaged {len(ckpt_paths)} WESAD fold checkpoints")
    return model

def load_wesad_all(processed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all WESAD windowed subjects."""
    Xs, ys = [], []
    for f in sorted((processed_dir / "wesad").glob("S*_windows.npz")):
        d = np.load(f)
        Xs.append(d["X"]); ys.append(d["y"])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    print(f"  WESAD: {X.shape[0]} windows, stress={y.mean():.3f}")
    return X, y

def predict_tcn(model, X: np.ndarray, hrv: np.ndarray, device: str) -> tuple:
    """Run batched inference, return (pred_labels, probs)."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    h_t = torch.tensor(hrv, dtype=torch.float32)
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X_t), 64):
            xb = X_t[i:i+64].to(device)
            hb = h_t[i:i+64].to(device)
            p  = torch.softmax(model(xb, hb), dim=-1)
            probs_list.extend(p.cpu().numpy())
    probs = np.array(probs_list)
    return probs.argmax(-1), probs

def train_baseline_on_wesad(model_cls, X_wesad: np.ndarray, y_wesad: np.ndarray,
                             device: str) -> nn.Module:
    """Train a baseline model on all WESAD data (zero-shot: no E4Stress labels)."""
    # BiLSTMClassifier uses 'input_size'; DilatedCNNClassifier uses 'input_channels'
    try:
        model = model_cls(input_channels=4, num_classes=2).to(device)
    except TypeError:
        model = model_cls(input_size=4, num_classes=2).to(device)
    model.train()

    dataset = TensorDataset(
        torch.tensor(X_wesad, dtype=torch.float32),
        torch.tensor(y_wesad, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Compute class weight for imbalance
    pos = y_wesad.sum()
    neg = len(y_wesad) - pos
    w = torch.tensor([1.0, neg / (pos + 1e-8)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    for epoch in range(30):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    return model

def predict_baseline(model, X: np.ndarray, device: str) -> tuple:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X_t), 64):
            p = torch.softmax(model(X_t[i:i+64].to(device)), dim=-1)
            probs_list.extend(p.cpu().numpy())
    probs = np.array(probs_list)
    return probs.argmax(-1), probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e4stress_root",
                        default="")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    cfg = load_config(str(CONFIG))
    processed_dir = Path(cfg["data"]["processed_dir"])
    bvp_ch = cfg["data"].get("bvp_channel_idx", 1)
    fs = cfg["preprocessing"]["target_fs"]
    hrv_feat = N_HRV_FEATURES if HAS_HRV else 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

        e4_root = Path(args.e4stress_root)
    if not e4_root.exists():
        print(f"[ERROR] EmpaticaE4Stress root not found: {e4_root}")
        print("Download from: https://data.mendeley.com/datasets/kb42z77m2g/2")
        print("Then re-run: python scripts/43_cross_dataset_e4stress.py --e4stress_root /path/to/data")
        return

    print("\nLoading EmpaticaE4Stress...")
    e4_data = load_all_subjects(args.e4stress_root, str(processed_dir))
    if not e4_data:
        print("[ERROR] No subjects loaded. Check dataset path and structure.")
        return

    X_e4_all = np.concatenate([v[0] for v in e4_data.values()])
    y_e4_all = np.concatenate([v[1] for v in e4_data.values()])
    print(f"  Total: {X_e4_all.shape[0]} windows, stress={y_e4_all.mean():.3f}")

        print("\nLoading WESAD (for baseline training)...")
    X_wesad, y_wesad = load_wesad_all(processed_dir)

        if HAS_HRV:
        print("\nExtracting HRV for EmpaticaE4Stress...")
        hrv_e4 = extract_hrv_batch(X_e4_all, bvp_channel=bvp_ch, fs=fs)
        hrv_wesad = extract_hrv_batch(X_wesad, bvp_channel=bvp_ch, fs=fs)
        # Normalize HRV with WESAD training stats
        mu, sig = hrv_wesad.mean(0), hrv_wesad.std(0) + 1e-8
        hrv_e4_norm = (hrv_e4 - mu) / sig
        hrv_e4_norm_zero = np.zeros((len(X_wesad), hrv_feat))
    else:
        hrv_e4_norm = np.zeros((len(X_e4_all), 0))

    results = []

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Lite-TCN-SE v8b (averaged WESAD checkpoints, zero-shot)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*55)
    print("1. Lite-TCN-SE v8b (zero-shot, WESAD → EmpaticaE4Stress)")
    print("="*55)
    tcn_model = average_wesad_checkpoints(cfg, hrv_feat, args.device)
    preds, probs = predict_tcn(tcn_model, X_e4_all, hrv_e4_norm, args.device)
    m = compute_metrics(y_e4_all, preds, probs)
    m.update({"source": "WESAD", "target": "EmpaticaE4Stress", "model": "Lite-TCN-SE"})
    results.append(m)
    print(f"  F1={m['f1']:.4f}  AUROC={m['auroc']:.4f}  "
          f"Acc={m['accuracy']:.4f}  ECE={m['ece']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Random Forest (handcrafted features, trained on WESAD)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*55)
    print("2. Random Forest (WESAD features → EmpaticaE4Stress)")
    print("="*55)
    X_wesad_feat = extract_stat_features(X_wesad)
    X_e4_feat    = extract_stat_features(X_e4_all)
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42,
                                class_weight="balanced")
    rf.fit(X_wesad_feat, y_wesad)
    rf_preds = rf.predict(X_e4_feat)
    rf_probs = rf.predict_proba(X_e4_feat)
    m_rf = compute_metrics(y_e4_all, rf_preds, rf_probs)
    m_rf.update({"source": "WESAD", "target": "EmpaticaE4Stress", "model": "RandomForest"})
    results.append(m_rf)
    print(f"  F1={m_rf['f1']:.4f}  AUROC={m_rf['auroc']:.4f}  "
          f"Acc={m_rf['accuracy']:.4f}  ECE={m_rf['ece']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. BiLSTM (trained on WESAD, zero-shot on E4Stress)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*55)
    print("3. BiLSTM+Attention (WESAD → EmpaticaE4Stress)")
    print("="*55)
    bilstm = train_baseline_on_wesad(BiLSTMClassifier, X_wesad, y_wesad, args.device)
    bl_preds, bl_probs = predict_baseline(bilstm, X_e4_all, args.device)
    m_bl = compute_metrics(y_e4_all, bl_preds, bl_probs)
    m_bl.update({"source": "WESAD", "target": "EmpaticaE4Stress", "model": "BiLSTM"})
    results.append(m_bl)
    print(f"  F1={m_bl['f1']:.4f}  AUROC={m_bl['auroc']:.4f}  "
          f"Acc={m_bl['accuracy']:.4f}  ECE={m_bl['ece']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Dilated CNN (trained on WESAD, zero-shot on E4Stress)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*55)
    print("4. Dilated CNN (WESAD → EmpaticaE4Stress)")
    print("="*55)
    dcnn = train_baseline_on_wesad(DilatedCNNClassifier, X_wesad, y_wesad, args.device)
    dc_preds, dc_probs = predict_baseline(dcnn, X_e4_all, args.device)
    m_dc = compute_metrics(y_e4_all, dc_preds, dc_probs)
    m_dc.update({"source": "WESAD", "target": "EmpaticaE4Stress", "model": "DilatedCNN"})
    results.append(m_dc)
    print(f"  F1={m_dc['f1']:.4f}  AUROC={m_dc['auroc']:.4f}  "
          f"Acc={m_dc['accuracy']:.4f}  ECE={m_dc['ece']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("CROSS-DATASET EVALUATION SUMMARY: WESAD → EmpaticaE4Stress")
    print("="*65)
    print(f"  {'Model':<20}  {'F1':>6}  {'AUROC':>7}  {'Acc':>6}  {'ECE':>6}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}")
    for r in results:
        print(f"  {r['model']:<20}  "
              f"{r['f1']:.4f}  {r['auroc']:.4f}  "
              f"{r['accuracy']:.4f}  {r['ece']:.4f}")
    print("="*65)

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "cross_dataset_e4stress_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

if __name__ == "__main__":
    main()
