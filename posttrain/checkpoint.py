"""Checkpoint that can resume a run: weights, optimiser, and RNG.

Colab jobs die. A checkpoint that saves only the weights restarts the
optimiser moments and the dropout mask sequence, so the next step is not
the step you would have taken.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path: str | Path, model, optimizer, step: int, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    torch.set_rng_state(blob["torch_rng"])
    np.random.set_state(blob["numpy_rng"])
    random.setstate(blob["python_rng"])
    return blob
