"""Main LiteCARE-RAGent pipeline: signal → model → abstention → RAG retrieval."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.lite_tcn_se import build_model
from src.models.mc_inference import mc_predict, agent_decision
from src.rag_mini.retrieval import load_knowledge_base, retrieve, format_intervention


# ── Activity inference from ACC ───────────────────────────────────────────────

ACC_CHANNEL = 3   # channel index in [EDA, BVP, TEMP, ACC_mag] layout

def infer_activity_level(window: np.ndarray,
                          acc_channel: int = ACC_CHANNEL,
                          threshold: float = 0.15) -> str:
    """
    Automatically infer activity level from the ACC magnitude channel.

    High ACC variance indicates physical movement (walking, exercise);
    low variance indicates a sedentary state (desk work, seated driving).

    Args:
        window:      [C, T] preprocessed sensor window (z-normalised)
        acc_channel: index of the ACC magnitude channel (default 3)
        threshold:   std threshold separating active from sedentary (default 0.15)
                     Tuned on WESAD wrist ACC at 32 Hz after z-normalisation.

    Returns:
        'active' if std(ACC) > threshold, else 'sedentary'
    """
    acc_std = float(window[acc_channel].std())
    return "active" if acc_std > threshold else "sedentary"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Structured output from one agent invocation."""
    stress_detected: bool
    uncertainty: float
    abstained: bool
    decision: str                          # 'stress' | 'no_stress' | 'abstain'
    interventions: list[dict] = field(default_factory=list)
    formatted_suggestions: list[str] = field(default_factory=list)
    mean_probs: Optional[np.ndarray] = None
    inference_ms: float = 0.0
    context: str = "general"
    activity_level: str = "unknown"

    def summary(self) -> str:
        lines = [
            f"[LiteCARE-RAGent] Decision: {self.decision.upper()}",
            f"  Stress probability : {self.mean_probs[1]:.3f}" if self.mean_probs is not None else "",
            f"  Uncertainty (H)    : {self.uncertainty:.4f}",
            f"  Abstained          : {self.abstained}",
            f"  Inference time     : {self.inference_ms:.1f} ms",
        ]
        if self.interventions:
            lines.append("\n  --- Well-being Suggestions ---")
            for i, s in enumerate(self.formatted_suggestions, 1):
                lines.append(f"  {i}. {s}")
        elif self.decision == "no_stress":
            lines.append("  No stress detected. Keep it up!")
        elif self.abstained:
            lines.append("  Low-confidence reading — no intervention triggered.")
        return "\n".join(l for l in lines if l)


# ── Main Agent ────────────────────────────────────────────────────────────────

class LiteCARE_RAGent:
    """
    End-to-end LiteCARE-RAGent inference pipeline.

    Parameters
    ----------
    model_path : str | Path
        Path to saved model state_dict (.pt file).
    cfg : dict
        Full config dict (loaded from config.yaml).
    tau : float | None
        Abstention threshold τ.  If None, abstention is disabled (τ=∞).
    device : str
        'cpu' or 'cuda'.
    variant : str
        Model variant: 'full' | 'no_se' | 'fixed_dilation' (see lite_tcn_se.py).
    """

    def __init__(
        self,
        model_path: str | Path,
        cfg: dict,
        tau: float | None = None,
        device: str = "cpu",
        variant: str = "full",
    ):
        self.cfg = cfg
        self.tau = tau if tau is not None else float("inf")
        self.device = torch.device(device)
        self.variant = variant

        # Build model and load weights
        mc_cfg = cfg.get("mc_dropout", {})
        self.n_mc_samples = mc_cfg.get("n_samples", 30)

        model_cfg = cfg.get("model", {})
        self.model = build_model(
            variant=variant,
            input_channels=model_cfg.get("input_channels", 4),
        )
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # Load knowledge base once
        kb_path = cfg.get("rag_mini", {}).get("knowledge_base_path", None)
        self.kb = load_knowledge_base(kb_path)
        self.top_k = cfg.get("rag_mini", {}).get("top_k", 3)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        window: np.ndarray,
        context: str = "general",
        activity_level: str = "auto",
    ) -> AgentResult:
        """
        Run the full LiteCARE-RAGent pipeline on a single window.

        Parameters
        ----------
        window : np.ndarray
            Shape [4, 1920] — (EDA, BVP, TEMP, ACC_mag) at 32 Hz for 60 s.
            Should be z-normalised before passing in.
        context : str
            Deployment context: 'wearable' | 'workplace' | 'driving' | 'general'
            Set once at deployment time — not inferred from the signal.
        activity_level : str
            'active' | 'sedentary' | 'unknown' | 'auto' (default).
            When 'auto', activity level is inferred from the ACC channel variance:
            std(ACC) > 0.15 → 'active', else → 'sedentary'.

        Returns
        -------
        AgentResult with all fields populated.
        """
        t0 = time.perf_counter()

        # 1. Auto-infer activity level from ACC if not explicitly provided
        if activity_level == "auto":
            activity_level = infer_activity_level(window)

        # 2. Tensor preparation
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 4, 1920]

        # 3. MC-Dropout inference
        mc_result = mc_predict(self.model, x, n_samples=self.n_mc_samples, device=self.device)
        mean_probs  = mc_result["mean_probs"][0]    # [num_classes]
        pred_class  = int(mc_result["pred_class"][0])
        uncertainty = float(mc_result["uncertainty"][0])

        inference_ms = (time.perf_counter() - t0) * 1000

        # 4. Abstention decision
        dec_result = agent_decision(uncertainty, pred_class, self.tau)
        decision = dec_result["decision"]
        abstained = (decision == "abstain")
        stress_detected = (decision == "stress")

        # 5. RAG-Mini retrieval (only if stress confirmed)
        stress_label = "stress" if stress_detected else "no_stress"
        interventions = retrieve(
            stress_state=stress_label,
            context=context,
            activity_level=activity_level,
            top_k=self.top_k,
            kb=self.kb,
        )
        suggestions = [format_intervention(iv) for iv in interventions]

        return AgentResult(
            stress_detected=stress_detected,
            uncertainty=float(uncertainty),
            abstained=abstained,
            decision=decision,
            interventions=interventions,
            formatted_suggestions=suggestions,
            mean_probs=mean_probs,
            inference_ms=inference_ms,
            context=context,
            activity_level=activity_level,
        )

    # ── Batch evaluation helper ───────────────────────────────────────────────

    def run_batch(
        self,
        X: np.ndarray,
        contexts: list[str] | str = "general",
        activity_levels: list[str] | str = "unknown",
    ) -> list[AgentResult]:
        """
        Run pipeline over multiple windows.

        Parameters
        ----------
        X : np.ndarray
            Shape [N, 4, 1920]
        contexts : list or str
            Per-window context or single context for all windows.
        activity_levels : list or str
            Per-window activity level or single value for all.

        Returns
        -------
        list of AgentResult, one per window.
        """
        n = len(X)
        if isinstance(contexts, str):
            contexts = [contexts] * n
        if isinstance(activity_levels, str):
            activity_levels = [activity_levels] * n

        return [
            self.run(X[i], contexts[i], activity_levels[i])
            for i in range(n)
        ]

    def predict_labels(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Lightweight batch prediction returning only labels + uncertainties.
        No RAG retrieval. Useful for evaluation metrics.

        Returns
        -------
        preds : np.ndarray [N] — 0/1 stress labels (abstained → -1)
        uncertainties : np.ndarray [N]
        """
        preds, uncerts = [], []
        for i in range(len(X)):
            x = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            mc_result = mc_predict(self.model, x, n_samples=self.n_mc_samples, device=self.device)
            pred_class  = int(mc_result["pred_class"][0])
            uncertainty = float(mc_result["uncertainty"][0])
            dec_result  = agent_decision(uncertainty, pred_class, self.tau)
            decision    = dec_result["decision"]
            label = -1 if decision == "abstain" else (1 if decision == "stress" else 0)
            preds.append(label)
            uncerts.append(uncertainty)
        return np.array(preds, dtype=np.int64), np.array(uncerts, dtype=np.float32)

    # ── Threshold calibration ─────────────────────────────────────────────────

    def calibrate_tau(self, X_val: np.ndarray, y_val: np.ndarray, percentile: float = 95.0):
        """
        Calibrate abstention threshold τ on a validation set.
        Sets self.tau in-place and returns the computed value.
        τ = p-th percentile of entropy for correctly classified windows.
        """
        from src.models.mc_inference import calibrate_threshold
        self.tau = calibrate_threshold(
            self.model, X_val, y_val,
            n_samples=self.n_mc_samples,
            percentile=percentile,
            device=self.device,
        )
        print(f"[Agent] Calibrated τ = {self.tau:.4f} (percentile={percentile})")
        return self.tau

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"LiteCARE_RAGent(variant={self.variant}, "
            f"params={n_params:,}, τ={self.tau:.4f}, "
            f"n_mc={self.n_mc_samples}, kb={len(self.kb)} entries)"
        )
