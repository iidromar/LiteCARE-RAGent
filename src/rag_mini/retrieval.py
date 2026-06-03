"""Taxonomy-based retrieval from the local knowledge base. No LLM required."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Literal


CONTEXT_TYPE = Literal["wearable", "workplace", "driving", "general"]
ACTIVITY_TYPE = Literal["active", "sedentary", "unknown"]

# Context → implied probable stress types.
# Driving elicits physical arousal and attentional (mental) load.
# Workplace tasks primarily cause mental and emotional stress.
# Wearable/general covers all types equally.
_CONTEXT_STRESS_TYPES: dict[str, list[str]] = {
    "driving":   ["physical", "mental"],
    "workplace": ["mental", "emotional"],
    "wearable":  ["mental", "emotional", "physical"],
    "general":   ["mental", "emotional", "physical"],
}


def load_knowledge_base(kb_path: str | None = None) -> list[dict]:
    """Load interventions from JSON knowledge base."""
    if kb_path is None:
        kb_path = Path(__file__).parent / "knowledge_base.json"
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(stress_state: str,
             context: CONTEXT_TYPE = "general",
             activity_level: ACTIVITY_TYPE = "unknown",
             top_k: int = 3,
             kb: list[dict] | None = None,
             kb_path: str | None = None) -> list[dict]:
    """
    Retrieve the most relevant interventions for the detected stress context.

    Scoring uses all six knowledge-base attributes:
      +3  exact context match
      +1  intervention tagged 'general' (broad applicability)
      +1  per overlapping stress_type with the context-implied stress profile
      +2  intensity == 'low' when activity_level == 'active'
      −10 type not in safe_types when context == 'driving'

    Args:
        stress_state:   'stress' or 'no_stress'  (if no_stress → no intervention needed)
        context:        deployment context — determines both context score and
                        the implied stress-type profile used for stress_type scoring
        activity_level: 'active' | 'sedentary' | 'unknown' — gates intensity preference;
                        inferred automatically from ACC magnitude variance in agent.py
        top_k:          number of interventions to return
        kb:             pre-loaded knowledge base (optional, avoids re-loading)
        kb_path:        path to knowledge_base.json (optional)

    Returns:
        list of up to top_k intervention dicts, ranked by composite relevance score
    """
    if stress_state != "stress":
        return []

    if kb is None:
        kb = load_knowledge_base(kb_path)

    # Implied stress types for this deployment context
    implied_stress_types = set(_CONTEXT_STRESS_TYPES.get(context, ["mental", "emotional", "physical"]))

    # Determine preferred intensity based on ACC-inferred activity level
    # 'active' → prefer low-intensity to avoid suggesting exercise while already moving
    preferred_intensity = "low" if activity_level == "active" else None

    # Driving context: only safe intervention types (no movement/physical exercises)
    safe_types = {"breathing", "cognitive", "sensory"} if context == "driving" else None

    scored = []
    for item in kb:
        score = 0

        # ── Context match (primary signal) ────────────────────────────────────
        if context in item.get("context", []):
            score += 3
        elif "general" in item.get("context", []):
            score += 1

        # ── Stress-type relevance (secondary signal) ──────────────────────────
        # Each overlapping tag between the intervention's stress_type list and
        # the context-implied profile adds +1, rewarding specificity.
        item_stress_types = set(item.get("stress_type", []))
        score += len(item_stress_types & implied_stress_types)

        # ── Intensity fit (activity-level gate) ───────────────────────────────
        if preferred_intensity and item.get("intensity") == preferred_intensity:
            score += 2

        # ── Safety hard-filter for driving ────────────────────────────────────
        if safe_types and item.get("type") not in safe_types:
            score -= 10

        scored.append((score, item))

    # Sort by score descending, randomize ties for diversity
    scored.sort(key=lambda x: (-x[0], random.random()))
    candidates = [item for score, item in scored if score >= 0]

    return candidates[:top_k]


def format_intervention(intervention: dict) -> str:
    """
    Format a single intervention for display to the user.

    Shows the suggestion text, the scientific evidence rationale, and the
    academic source — all three fields are present in every knowledge-base entry
    to ensure transparency and verifiability of recommendations.
    """
    text     = intervention.get("intervention", "Take a moment to breathe deeply.")
    evidence = intervention.get("evidence", "")
    source   = intervention.get("source", "")
    lines    = [f"[Suggestion] {text}"]
    if evidence:
        lines.append(f"  Evidence: {evidence}")
    if source:
        lines.append(f"  Source: {source}")
    return "\n".join(lines)
