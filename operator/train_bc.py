"""Behavior cloning: MLP maps robot state -> action.

Usage:
    python operator/train_bc.py configs/operator/bc_smoke.yaml [--smoke]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def build_bc_model(state_dim: int, action_dim: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, action_dim),
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    from labpt.config import check_gpu, training_config
    from labpt.profiling import StepTimer

    cfg = training_config("configs/operator/bc_a100.yaml", argv=argv)
    check_gpu(cfg)
    torch.manual_seed(0)
    dev = "cpu" if cfg.get("device") == "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(cfg["frames"])
    df = df[df.repo == cfg["repo"]]  # one robot per run (state dims differ across robots)
    sd_dim, ad_dim = len(df.state.iloc[0]), len(df.action.iloc[0])

    def tensors(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        d = df[df.split == split]
        return (
            torch.tensor(np.stack(d.state.to_numpy()), dtype=torch.float32),
            torch.tensor(np.stack(d.action.to_numpy()), dtype=torch.float32),
        )

    x_tr, y_tr = tensors("train")
    x_te, y_te = tensors("test")
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    model = build_bc_model(sd_dim, ad_dim, int(cfg["hidden"])).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    timer = StepTimer(out_dir, "bc")
    for _step in range(cfg["steps"]):
        timer.start()
        idx = torch.randint(0, len(x_tr), (cfg["batch_size"],))
        x, y = ((x_tr[idx] - mu) / sd).to(dev), y_tr[idx].to(dev)
        timer.mark("data")
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        timer.mark("fwd_bwd_opt")
        timer.end(len(idx), loss=round(loss.item(), 5))
    prof = timer.close()

    with torch.no_grad():
        pred = model(((x_te - mu) / sd).to(dev)).cpu()
        mse = nn.functional.mse_loss(pred, y_te).item()
        base = nn.functional.mse_loss(y_tr.mean(0).expand_as(y_te), y_te).item()  # predict-the-mean baseline

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mu": mu.cpu(),
            "sd": sd.cpu(),
            "state_dim": sd_dim,
            "action_dim": ad_dim,
            "hidden": int(cfg["hidden"]),
        },
        ckpt_path,
    )
    res = {
        "test_mse": round(mse, 5),
        "mean_baseline_mse": round(base, 5),
        "n_train": len(x_tr),
        "n_test": len(x_te),
        "checkpoint": str(ckpt_path),
        "profile": prof,
    }
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
