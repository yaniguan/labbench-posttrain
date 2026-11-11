"""Load the reward-ladder grid. Names only. This module does not download models."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = ROOT / "configs" / "posttrain" / "grid.yaml"


def load_yaml(path) -> dict:
    with open(path) as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    return data


def expand_grid(cfg: dict) -> list[dict]:
    """One dict per model x reward x env. Nothing is launched."""
    runs = []
    for model in cfg["models"]:
        for reward_name in cfg["rewards"]:
            for env in cfg["envs"]:
                runs.append({
                    "model_name": model["name"],
                    "family": model["family"],
                    "reward_name": reward_name,
                    "env": env,
                    "method": cfg.get("method", "grpo"),
                    "download_in_ci": False,
                    "status": "not run",
                })
    return runs


def load_grid(path=None) -> list[dict]:
    return expand_grid(load_yaml(path or GRID_PATH))
