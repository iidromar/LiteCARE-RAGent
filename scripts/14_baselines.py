"""
Script 50 — New Baselines: XGBoost, MiniRocket, InceptionTime
=============================================================
Runs LOSO-CV for three additional baselines under identical conditions
as the existing baselines in scripts/09_full_evaluation.py:
  - Same 15 WESAD subjects (excl S12/S1)
  - Same LOSO split (preceding subject = val, rest = train)
  - Same class weighting (balanced)
  - Same seed (42)

Outputs:
  results/baselines/xgboost/loso_results.csv
  results/baselines/minirocket/loso_results.csv
  results/baselines/inceptiontime/loso_results.csv
"""
from __future__ import annotations
import sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.baselines import extract_statistical_features, build_rf_pipeline
from src.evaluation.metrics import compute_metrics

BASE         = Path(__file__).parent.parent
CONFIG       = BASE / "config/config_multidomain.yaml"
PROCESSED    = BASE / "data/processed/wesad"
RESULTS_BASE = BASE / "results/baselines"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_all_subjects() -> dict:
    data = {}
    for f in sorted(PROCESSED.glob("*.npz")):
        sid = f.stem.replace("_windows", "")
        d = np.load(f)
        data[sid] = (d["X"], d["y"])
    return data

def agg_metrics(records: list[dict], excl_s17: bool = True) -> dict:
    df = pd.DataFrame(records)
    if excl_s17:
        df = df[df.test_subject != "S17"]
    cols = ["accuracy", "precision", "recall", "f1", "auroc", "ece"]
    return {c: float(df[c].mean()) for c in cols if c in df}

def run_xgboost_loso(all_data: dict) -> list[dict]:
    import xgboost as xgb

    out_dir = RESULTS_BASE / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "loso_results.csv"
    if cache.exists():
        print("  [CACHE] XGBoost results loaded.")
        return pd.read_csv(cache).to_dict("records")

    subjects = sorted(all_data.keys())
    results  = []

    print(f"\nXGBoost LOSO — {len(subjects)} folds")
    for i, test_sid in enumerate(subjects):
        val_sid    = subjects[(i - 1) % len(subjects)]
        train_sids = [s for s in subjects if s != test_sid and s != val_sid]

        X_tr = np.concatenate([extract_statistical_features(all_data[s][0]) for s in train_sids])
        y_tr = np.concatenate([all_data[s][1] for s in train_sids])
        X_te = extract_statistical_features(all_data[test_sid][0])
        y_te = all_data[test_sid][1]

        # Class weights
        neg, pos = np.sum(y_tr == 0), np.sum(y_tr == 1)
        scale_pos = neg / pos if pos > 0 else 1.0

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
        clf.fit(X_tr_s, y_tr)

        preds = clf.predict(X_te_s)
        probs = clf.predict_proba(X_te_s)
        m = compute_metrics(y_te, preds, probs)
        m["test_subject"] = test_sid
        results.append(m)
        print(f"  [{i+1:2d}/{len(subjects)}] {test_sid}: F1={m['f1']:.4f} AUROC={m.get('auroc',0):.4f}")

    pd.DataFrame(results).to_csv(cache, index=False)
    return results

class MiniRocketTransformer:
    """
    Simplified MiniRocket: random dilated kernels → PPV features → Ridge.
    Dempster et al. (2021), doi:10.1007/s10618-020-00710-y
    """

    def __init__(self, n_kernels: int = 10_000, max_dilation: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        k_len = 9
        self.n_kernels = n_kernels
        self.kernels   = []
        for _ in range(n_kernels):
            # weights: 2/3 of positions = -1, 1/3 = +2 (zero sum approximately)
            w = rng.choice([-1, -1, 2], size=k_len).astype(np.float32)
            w = w - w.mean()
            dilation = int(rng.choice(range(1, max_dilation + 1)))
            padding  = int(rng.choice([0, (k_len - 1) * dilation // 2]))
            bias     = float(rng.uniform(-1, 1))
            channel  = int(rng.integers(0, 4))  # 4 input channels
            self.kernels.append((w, dilation, padding, bias, channel))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: [N, C, T] → features: [N, n_kernels]"""
        N, C, T = X.shape
        feats = np.zeros((N, self.n_kernels), dtype=np.float32)
        for j, (w, dil, pad, bias, ch) in enumerate(self.kernels):
            from scipy.signal import convolve
            k = np.zeros(w.shape[0] + (w.shape[0] - 1) * (dil - 1))
            k[::dil] = w
            for n in range(N):
                sig = X[n, ch]
                out = np.convolve(sig, k[::-1], mode="full")
                # trim to valid-like region
                start = (k.shape[0] - 1) // 2
                out   = out[start:start + T] + bias
                feats[n, j] = np.mean(out > 0)  # PPV
        return feats

    def transform_fast(self, X: np.ndarray) -> np.ndarray:
        """Vectorized transform using PyTorch conv1d for speed."""
        N, C, T = X.shape
        X_t = torch.tensor(X, dtype=torch.float32)
        feats = []
        batch_size = 500  # process kernels in batches
        for start in range(0, self.n_kernels, batch_size):
            batch = self.kernels[start:start + batch_size]
            batch_ppv = []
            for (w, dil, pad, bias, ch) in batch:
                w_t = torch.tensor(w, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                inp = X_t[:, ch:ch+1, :]
                out = torch.nn.functional.conv1d(inp, w_t, dilation=dil, padding=pad)
                ppv = (out + bias > 0).float().mean(dim=-1)  # [N, 1]
                batch_ppv.append(ppv.squeeze(1).numpy())
            feats.append(np.stack(batch_ppv, axis=1))
        return np.concatenate(feats, axis=1)

def run_minirocket_loso(all_data: dict) -> list[dict]:
    out_dir = RESULTS_BASE / "minirocket"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "loso_results.csv"
    if cache.exists():
        print("  [CACHE] MiniRocket results loaded.")
        return pd.read_csv(cache).to_dict("records")

    subjects = sorted(all_data.keys())
    results  = []
    rocket   = MiniRocketTransformer(n_kernels=10_000, seed=SEED)

    print(f"\nMiniRocket LOSO — {len(subjects)} folds (extracting features first...)")
    # Pre-transform all subjects
    feats_all = {}
    for sid, (X, y) in all_data.items():
        feats_all[sid] = rocket.transform_fast(X)

    for i, test_sid in enumerate(subjects):
        val_sid    = subjects[(i - 1) % len(subjects)]
        train_sids = [s for s in subjects if s != test_sid and s != val_sid]

        X_tr = np.concatenate([feats_all[s] for s in train_sids])
        y_tr = np.concatenate([all_data[s][1] for s in train_sids])
        X_te = feats_all[test_sid]
        y_te = all_data[test_sid][1]

        neg, pos = np.sum(y_tr == 0), np.sum(y_tr == 1)
        cw = {0: 1.0, 1: neg / pos if pos > 0 else 1.0}

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(
            max_iter=1000, C=1.0,
            class_weight=cw,
            random_state=SEED,
            solver="lbfgs",
        )
        clf.fit(X_tr_s, y_tr)

        preds = clf.predict(X_te_s)
        probs = clf.predict_proba(X_te_s)
        m = compute_metrics(y_te, preds, probs)
        m["test_subject"] = test_sid
        results.append(m)
        print(f"  [{i+1:2d}/{len(subjects)}] {test_sid}: F1={m['f1']:.4f} AUROC={m.get('auroc',0):.4f}")

    pd.DataFrame(results).to_csv(cache, index=False)
    return results

class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, n_filters: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, n_filters, kernel_size=1, bias=False)
        self.conv10 = nn.Conv1d(n_filters, n_filters, kernel_size=10, padding=4, bias=False)
        self.conv20 = nn.Conv1d(n_filters, n_filters, kernel_size=20, padding=9, bias=False)
        self.conv40 = nn.Conv1d(n_filters, n_filters, kernel_size=40, padding=19, bias=False)
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, n_filters, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(n_filters * 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        c10 = self.conv10(b)
        c20 = self.conv20(b)
        c40 = self.conv40(b)
        mp  = self.maxpool(x)
        # Trim all to the shortest output (even kernels give T-1 length)
        L = min(c10.shape[-1], c20.shape[-1], c40.shape[-1], mp.shape[-1])
        out = torch.cat([c10[..., :L], c20[..., :L], c40[..., :L], mp[..., :L]], dim=1)
        return self.act(self.bn(out))

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.act(self.shortcut(x) + skip)

class InceptionTime(nn.Module):
    """
    InceptionTime: Ismail Fawaz et al. 2020. Canonical depth=6 with residuals
    every 3 blocks. Residual shortcut uses input to each group of 3.
    """
    def __init__(self, in_ch: int = 4, n_classes: int = 2, n_filters: int = 32, depth: int = 6):
        super().__init__()
        out_ch = n_filters * 4
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()  # one per group of 3

        ch = in_ch
        for g in range(depth // 3):
            group = nn.ModuleList()
            for b in range(3):
                group.append(InceptionBlock(ch if b == 0 else out_ch, n_filters))
            self.blocks.append(group)
            self.shortcuts.append(nn.Sequential(
                nn.Conv1d(ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            ))
            ch = out_ch

        self.act = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(out_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for group, shortcut in zip(self.blocks, self.shortcuts):
            residual = x
            for block in group:
                x = block(x)
            # Align lengths for skip connection
            L = min(x.shape[-1], residual.shape[-1])
            sc = shortcut(residual)[..., :L]
            x  = self.act(x[..., :L] + sc)
        x = self.gap(x).squeeze(-1)
        return self.head(x)

def run_inceptiontime_loso(all_data: dict, device: str = "cpu") -> list[dict]:
    out_dir = RESULTS_BASE / "inceptiontime"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "loso_results.csv"
    if cache.exists():
        print("  [CACHE] InceptionTime results loaded.")
        return pd.read_csv(cache).to_dict("records")

    subjects = sorted(all_data.keys())
    results  = []
    MAX_EPOCHS = 40
    PATIENCE   = 8
    BATCH      = 32
    LR         = 1e-3

    print(f"\nInceptionTime LOSO — {len(subjects)} folds")
    for i, test_sid in enumerate(subjects):
        val_sid    = subjects[(i - 1) % len(subjects)]
        train_sids = [s for s in subjects if s != test_sid and s != val_sid]

        X_tr = np.concatenate([all_data[s][0] for s in train_sids])
        y_tr = np.concatenate([all_data[s][1] for s in train_sids])
        X_val, y_val = all_data[val_sid]
        X_te,  y_te  = all_data[test_sid]

        neg, pos = np.sum(y_tr == 0), np.sum(y_tr == 1)
        wt = torch.tensor([1.0, neg / pos if pos > 0 else 1.0], dtype=torch.float32).to(device)

        model = InceptionTime(in_ch=4, n_classes=2, n_filters=32, depth=6).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        crit  = nn.CrossEntropyLoss(weight=wt)

        tr_ld  = DataLoader(
            TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                          torch.tensor(y_tr, dtype=torch.long)),
            batch_size=BATCH, shuffle=True, drop_last=True)
        val_ld = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.long)),
            batch_size=BATCH, shuffle=False)

        best_f1, best_state, patience_cnt = 0.0, None, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for Xb, yb in tr_ld:
                opt.zero_grad()
                crit(model(Xb.to(device)), yb.to(device)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for Xb, yb in val_ld:
                    all_p.extend(torch.softmax(model(Xb.to(device)), -1).cpu().numpy().argmax(-1))
                    all_t.extend(yb.numpy())
            vf1 = compute_metrics(np.array(all_t), np.array(all_p))["f1"]
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        all_p, all_t, all_pr = [], [], []
        te_ld = DataLoader(
            TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                          torch.tensor(y_te, dtype=torch.long)),
            batch_size=BATCH)
        with torch.no_grad():
            for Xb, yb in te_ld:
                probs = torch.softmax(model(Xb.to(device)), -1).cpu().numpy()
                all_pr.extend(probs)
                all_p.extend(probs.argmax(-1))
                all_t.extend(yb.numpy())

        m = compute_metrics(np.array(all_t), np.array(all_p), np.array(all_pr))
        m["test_subject"] = test_sid
        results.append(m)
        print(f"  [{i+1:2d}/{len(subjects)}] {test_sid}: F1={m['f1']:.4f} AUROC={m.get('auroc',0):.4f}")

    pd.DataFrame(results).to_csv(cache, index=False)
    return results

def print_summary(name: str, records: list[dict]):
    df = pd.DataFrame(records)
    df14 = df[df.test_subject != "S17"]
    print(f"\n{name} (excl S17, n={len(df14)}):")
    for c in ["accuracy", "precision", "recall", "f1", "auroc", "ece"]:
        if c in df14:
            print(f"  {c:12s}: {df14[c].mean():.4f} ± {df14[c].std():.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-inceptiontime", action="store_true",
                        help="Skip InceptionTime (slow)")
    args = parser.parse_args()

    all_data = load_all_subjects()
    print(f"Loaded {len(all_data)} subjects: {sorted(all_data.keys())}")

    xgb_res = run_xgboost_loso(all_data)
    print_summary("XGBoost", xgb_res)

    mr_res = run_minirocket_loso(all_data)
    print_summary("MiniRocket", mr_res)

    if not args.skip_inceptiontime:
        it_res = run_inceptiontime_loso(all_data, device=args.device)
        print_summary("InceptionTime", it_res)

    # Summary JSON for thesis
    summary = {}
    for name, res in [("XGBoost", xgb_res), ("MiniRocket", mr_res)]:
        df = pd.DataFrame(res)
        df14 = df[df.test_subject != "S17"]
        summary[name] = {c: round(float(df14[c].mean()), 4)
                         for c in ["accuracy", "precision", "recall", "f1", "auroc", "ece"]
                         if c in df14}
    if not args.skip_inceptiontime:
        df = pd.DataFrame(it_res)
        df14 = df[df.test_subject != "S17"]
        summary["InceptionTime"] = {c: round(float(df14[c].mean()), 4)
                                     for c in ["accuracy", "precision", "recall", "f1", "auroc", "ece"]
                                     if c in df14}

    out_json = BASE / "results/baselines/new_baselines_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {out_json}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
