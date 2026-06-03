"""Early stopping and model checkpointing."""
import torch
import numpy as np
from pathlib import Path


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.min_delta = min_delta
        self.best      = -np.inf if mode == "max" else np.inf
        self.counter   = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        improved = (
            (self.mode == "max" and metric > self.best + self.min_delta) or
            (self.mode == "min" and metric < self.best - self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelCheckpoint:
    """Save the best model checkpoint to disk."""

    def __init__(self, save_path: str, mode: str = "max"):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf

    def __call__(self, metric: float, model: torch.nn.Module) -> bool:
        improved = (
            (self.mode == "max" and metric > self.best) or
            (self.mode == "min" and metric < self.best)
        )
        if improved:
            self.best = metric
            torch.save(model.state_dict(), self.save_path)
            return True
        return False
