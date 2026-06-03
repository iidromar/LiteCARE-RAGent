from __future__ import annotations
import yaml
from pathlib import Path


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
