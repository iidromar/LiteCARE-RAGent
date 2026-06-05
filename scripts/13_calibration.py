"""Script 48 — Comments 13, 16, 17, 18: Calibration & Statistical Analysis"""
from __future__ import annotations
import sys, numpy as np, torch, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.lite_tcn_se import LiteTCNSE
from src.preprocessing.hrv_features import N_HRV_FEATURES, extract_hrv_batch

BASE     = Path(__file__).parent.parent
CKPT_DIR = BASE / "results/wesad_loso/v8"
CONFIG   = BASE / "config/config_multidomain.yaml"
FIG_DIR  = BASE / "results/figures/calibration"
ABSTENTION_CSV = BASE / "results/abstention_loso_v8_v8b_results.csv"
N_PERMUTATIONS = 100
SEED = 42

def load_wesad_subject(processed_dir: Path, sid: str):
    f = processed_dir / "wesad" / f"{sid}_windows.npz"
    d = np.load(f)
    return d["X"], d["y"]

def load_all_wesad(processed_dir: Path):
    Xs, ys, sids = [], [], []
    for f in sorted((processed_dir / "wesad").glob("S*_windows.npz")):
        sid = f.stem.replace("_windows", "")
        d   = np.load(f)
        Xs.append(d["X"]); ys.append(d["y"]); sids.append(sid)
    return Xs, ys, sids

def extract_stat_features(X: np.ndarray) -> np.ndarray:
    feats = []
    for c in range(X.shape[1]):
        ch = X[:, c, :]
        feats += [ch.mean(1, keepdims=True), ch.std(1, keepdims=True),
                  ch.min(1, keepdims=True),  ch.max(1, keepdims=True)]
    return np.concatenate(feats, axis=1).astype(np.float32)

def build_model(cfg: dict, device: str) -> LiteTCNSE:
    mcfg = cfg["model"]
    return LiteTCNSE(
        input_channels=mcfg["input_channels"],
        num_classes=mcfg["num_classes"],
        channels_per_layer=mcfg["channels_per_layer"],
        dilation_schedule=mcfg["dilation_schedule"],
        kernel_size=mcfg["kernel_size"],
        dropout_rate=mcfg["dropout_rate"],
        se_reduction=mcfg.get("se_reduction_ratio", 4),
        hrv_features=N_HRV_FEATURES,
    ).to(device)

def get_probs(model, X: np.ndarray, hrv: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    h_t = torch.tensor(hrv, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), 128):
            p = torch.softmax(model(X_t[i:i+128].to(device),
                                    h_t[i:i+128].to(device)), -1)
            out.extend(p.cpu().numpy())
    return np.array(out)

def run_permutation_test(Xs, ys, sids, n_perm=N_PERMUTATIONS, seed=SEED):
    print("\n" + "="*55)
    print(f"Comment 13: Permutation test ({n_perm} permutations, RF)")
    print("="*55)

    # Real RF LOSO F1
    real_f1s = []
    for i, sid in enumerate(sids):
        X_test = Xs[i]; y_test = ys[i]
        X_train = np.concatenate([Xs[j] for j in range(len(sids)) if j != i])
        y_train = np.concatenate([ys[j] for j in range(len(sids)) if j != i])
        feat_tr = extract_stat_features(X_train)
        feat_te = extract_stat_features(X_test)
        rf = RandomForestClassifier(100, n_jobs=-1, random_state=seed, class_weight="balanced")
        rf.fit(feat_tr, y_train)
        real_f1s.append(f1_score(y_test, rf.predict(feat_te), zero_division=0))

    real_mean_f1 = np.mean(real_f1s)
    print(f"  Real RF LOSO F1 = {real_mean_f1:.4f}")

    rng = np.random.RandomState(seed)
    perm_f1s = []
    for perm in range(n_perm):
        fold_f1s = []
        for i, sid in enumerate(sids):
            X_test = Xs[i]; y_test = ys[i]
            X_train = np.concatenate([Xs[j] for j in range(len(sids)) if j != i])
            y_train = np.concatenate([ys[j] for j in range(len(sids)) if j != i])
            y_train_perm = rng.permutation(y_train)  # shuffle labels
            feat_tr = extract_stat_features(X_train)
            feat_te = extract_stat_features(X_test)
            rf = RandomForestClassifier(50, n_jobs=-1, random_state=seed+perm,
                                        class_weight="balanced")
            rf.fit(feat_tr, y_train_perm)
            fold_f1s.append(f1_score(y_test, rf.predict(feat_te), zero_division=0))
        perm_f1s.append(np.mean(fold_f1s))
        if (perm + 1) % 10 == 0:
            print(f"  Permutation {perm+1}/{n_perm}  running p-val ≤ "
                  f"{sum(f >= real_mean_f1 for f in perm_f1s)/(perm+1):.3f}")

    perm_arr = np.array(perm_f1s)
    p_value = (perm_arr >= real_mean_f1).mean()
    print(f"\n  Real F1 = {real_mean_f1:.4f}")
    print(f"  Permuted F1: mean={perm_arr.mean():.4f}  std={perm_arr.std():.4f}  "
          f"max={perm_arr.max():.4f}")
    print(f"  p-value = {p_value:.4f}  (fraction of {n_perm} perms ≥ real F1)")

    return {
        "real_f1": real_mean_f1,
        "perm_mean": perm_arr.mean(), "perm_std": perm_arr.std(),
        "perm_max": perm_arr.max(), "p_value": p_value,
        "n_permutations": n_perm, "perm_f1s": perm_f1s,
    }

def run_calibration(cfg, Xs, ys, sids, device):
    print("\n" + "="*55)
    print("Comment 16: Brier score + reliability diagram")
    print("="*55)
    processed_dir = Path(cfg["data"]["processed_dir"])
    bvp_ch = cfg["data"].get("bvp_channel_idx", 1)
    fs = cfg["preprocessing"]["target_fs"]

    brier_rows = []
    all_probs, all_y = [], []

    for sid in sids:
        ckpt = CKPT_DIR / f"v8_fold_{sid}.pt"
        if not ckpt.exists():
            print(f"  [SKIP] No checkpoint for {sid}"); continue

        # Train = all other subjects; Test = this subject
        X_test, y_test = Xs[sids.index(sid)], ys[sids.index(sid)]
        hrv_test = extract_hrv_batch(X_test, bvp_channel=bvp_ch, fs=fs)

        # Normalise HRV with training subjects' stats
        train_idxs = [j for j, s in enumerate(sids) if s != sid]
        X_tr_all = np.concatenate([Xs[j] for j in train_idxs])
        hrv_tr = extract_hrv_batch(X_tr_all, bvp_channel=bvp_ch, fs=fs)
        mu, sg = hrv_tr.mean(0), hrv_tr.std(0) + 1e-8
        hrv_norm = (hrv_test - mu) / sg

        model = build_model(cfg, device)
        s = torch.load(ckpt, map_location=device)
        model.load_state_dict(s); model.eval()

        probs = get_probs(model, X_test, hrv_norm, device)
        p1 = probs[:, 1]
        brier = brier_score_loss(y_test, p1)

        preds = probs.argmax(-1)
        f1 = f1_score(y_test, preds, zero_division=0)

        brier_rows.append({"subject": sid, "brier_score": brier, "f1": f1,
                           "n_windows": len(y_test)})
        all_probs.extend(p1.tolist())
        all_y.extend(y_test.tolist())
        print(f"  {sid}: Brier={brier:.4f}  F1={f1:.4f}")

    all_probs = np.array(all_probs)
    all_y     = np.array(all_y, dtype=int)
    mean_brier = brier_score_loss(all_y, all_probs)
    print(f"  Overall Brier score: {mean_brier:.4f}")

    # Reliability diagram data (10 bins)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    rel_data = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (all_probs >= lo) & (all_probs < hi)
        if mask.sum() == 0:
            rel_data.append({"bin_centre": (lo+hi)/2, "mean_confidence": (lo+hi)/2,
                             "fraction_positive": 0.0, "n": 0})
            continue
        rel_data.append({
            "bin_centre": (lo+hi)/2,
            "mean_confidence": all_probs[mask].mean(),
            "fraction_positive": all_y[mask].mean(),
            "n": int(mask.sum()),
        })

    return brier_rows, mean_brier, rel_data, all_probs, all_y

def coverage_f1_curve(all_probs: np.ndarray, all_y: np.ndarray):
    """Sweep entropy threshold τ → compute (coverage, F1) pairs."""
    # Entropy for binary: H = -p log p - (1-p) log(1-p)
    eps = 1e-10
    p1 = all_probs
    entropy = -(p1 * np.log(p1 + eps) + (1 - p1) * np.log(1 - p1 + eps))
    max_entropy = np.log(2)   # binary max

    taus = np.linspace(0, max_entropy, 200)
    coverages, f1s = [], []
    for tau in taus:
        decided = entropy <= tau
        cov = decided.mean()
        if decided.sum() == 0:
            coverages.append(cov); f1s.append(0.0); continue
        f = f1_score(all_y[decided], (all_probs[decided] >= 0.5).astype(int),
                     zero_division=0)
        coverages.append(cov); f1s.append(f)

    return np.array(coverages), np.array(f1s), taus

def plot_reliability(rel_data: list, mean_brier: float, save_path: Path):
    centres = [r["bin_centre"]     for r in rel_data]
    conf    = [r["mean_confidence"] for r in rel_data]
    frac    = [r["fraction_positive"] for r in rel_data]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(centres, frac, width=0.09, alpha=0.6, color="#2166ac", label="Fraction positive")
    ax.plot(conf, frac, "o-", color="#d73027", lw=2, ms=6, label="Model")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title(f"Reliability Diagram — Lite-TCN-SE v8b\nBrier Score = {mean_brier:.4f}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

def plot_coverage_f1(coverages: np.ndarray, f1s: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(coverages * 100, f1s, color="#2166ac", lw=2)
    ax.axhline(f1s[-1], color="grey", ls="--", lw=1, label=f"Full coverage F1={f1s[-1]:.3f}")
    ax.fill_between(coverages * 100, f1s, f1s[-1], alpha=0.12, color="#2166ac")
    ax.set_xlabel("Coverage (% windows decided)", fontsize=11)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_title("Selective Prediction: Coverage–F1 Curve\n"
                 "Lite-TCN-SE v8b with MC-Dropout Abstention", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

def plot_permutation(perm_f1s: list, real_f1: float, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm_f1s, bins=20, color="#4575b4", alpha=0.75, edgecolor="white",
            label=f"Permuted F1 (n={len(perm_f1s)})")
    ax.axvline(real_f1, color="#d73027", lw=2.5, ls="--",
               label=f"Real RF F1 = {real_f1:.4f}")
    ax.set_xlabel("F1 score (LOSO mean)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Permutation Test — Random Forest LOSO\n"
                 "(labels randomly shuffled on training set)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    cfg = load_config(str(CONFIG))
    processed_dir = Path(cfg["data"]["processed_dir"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all WESAD subjects...")
    Xs, ys, sids = load_all_wesad(processed_dir)
    print(f"  {len(sids)} subjects, "
          f"{sum(len(X) for X in Xs)} total windows")

        perm_results = run_permutation_test(Xs, ys, sids, n_perm=args.permutations)

    pd.DataFrame({"perm_f1": perm_results["perm_f1s"]}).to_csv(
        BASE / "results/permutation_test.csv", index=False)
    with open(BASE / "results/permutation_test_summary.txt", "w") as fh:
        fh.write(f"Permutation test (n={perm_results['n_permutations']})\n")
        fh.write(f"Real RF LOSO F1 : {perm_results['real_f1']:.4f}\n")
        fh.write(f"Permuted F1     : {perm_results['perm_mean']:.4f} ± {perm_results['perm_std']:.4f}\n")
        fh.write(f"p-value         : {perm_results['p_value']:.4f}\n")
    plot_permutation(perm_results["perm_f1s"], perm_results["real_f1"],
                     FIG_DIR / "permutation_test.png")

        brier_rows, mean_brier, rel_data, all_probs, all_y = run_calibration(
        cfg, Xs, ys, sids, args.device)

    pd.DataFrame(brier_rows).to_csv(BASE / "results/brier_scores.csv", index=False)
    pd.DataFrame(rel_data).to_csv(BASE / "results/reliability_diagram_data.csv", index=False)
    plot_reliability(rel_data, mean_brier, FIG_DIR / "reliability_diagram.png")

        print("\n" + "="*55)
    print("Comment 18: Coverage–F1 curve")
    print("="*55)
    coverages, f1s, taus = coverage_f1_curve(all_probs, all_y)
    pd.DataFrame({"tau": taus, "coverage": coverages, "f1": f1s}).to_csv(
        BASE / "results/coverage_f1_curve.csv", index=False)
    plot_coverage_f1(coverages, f1s, FIG_DIR / "coverage_f1_curve.png")

        print("\n" + "="*55)
    print("Summary")
    print("="*55)
    print(f"  Permutation p-value : {perm_results['p_value']:.4f}")
    print(f"  Overall Brier score : {mean_brier:.4f}")
    print(f"  Figures saved to    : {FIG_DIR}/")

if __name__ == "__main__":
    main()
