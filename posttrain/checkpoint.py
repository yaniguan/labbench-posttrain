"""Checkpoint that can resume a run: weights, extra modules, optimiser, and CPU/CUDA RNG.

Colab jobs die. A checkpoint that saves only the weights restarts the
optimiser moments, critic head, and the RNG sequence, so the next step is not
the step you would have taken.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    step: int,
    extra: dict[str, Any] | None = None,
    *,
    extra_modules: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_extra_modules = {
        k: (m.state_dict() if hasattr(m, "state_dict") else m)
        for k, m in (extra_modules or {}).items()
    }
    torch.save(
        {
            "model": model.state_dict(),
            "extra_modules": serialized_extra_modules,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    *,
    extra_modules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    saved_modules = blob.get("extra_modules") or {}
    if extra_modules:
        for k, mod in extra_modules.items():
            if k in saved_modules and hasattr(mod, "load_state_dict"):
                mod.load_state_dict(saved_modules[k])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    torch.set_rng_state(blob["torch_rng"])
    if torch.cuda.is_available() and blob.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    np.random.set_state(blob["numpy_rng"])
    random.setstate(blob["python_rng"])
    return blob
