"""Simple world model: predict next robot state from (state, action). Used for cheap rollouts.

Reports 1-step MSE and k-step open-loop rollout error on held-out episodes vs a 'state stays the same' baseline.
Usage:
    python operator/world_model.py configs/operator/wm_smoke.yaml [--smoke]
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


class WM(nn.Module):
    """State-delta MLP world model with self-contained `(mu, sd)` normalization buffers."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 128,
        *,
        mu: torch.Tensor | None = None,
        sd: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.state_dim),
        )
        self.register_buffer("mu", mu.clone().float() if mu is not None else torch.zeros(self.state_dim))
        self.register_buffer("sd", sd.clone().float() if sd is not None else torch.ones(self.state_dim))

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        norm_s = (s - self.mu) / self.sd
        return s + self.net(torch.cat([norm_s, a], dim=-1)) * self.sd


def _sort_episode_frames(df_ep: pd.DataFrame) -> pd.DataFrame:
    for col in ("frame_index", "frame", "step", "t", "timestamp"):
        if col in df_ep.columns:
            return df_ep.sort_values(col, kind="mergesort")
    return df_ep.sort_index()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    from labpt.config import check_gpu, training_config
    from labpt.profiling import StepTimer

    cfg = training_config("configs/operator/wm_a100.yaml", argv=argv)
    check_gpu(cfg)
    torch.manual_seed(0)
    dev = "cpu" if cfg.get("device") == "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(cfg["frames"])
    df = df[df.repo == cfg["repo"]]  # one robot per run (state dims differ across robots)
    sd_dim, ad_dim = len(df.state.iloc[0]), len(df.action.iloc[0])

    tr = df[df.split == "train"]
    S = torch.tensor(np.stack(tr.state.to_numpy()), dtype=torch.float32)
    A = torch.tensor(np.stack(tr.action.to_numpy()), dtype=torch.float32)
    N = torch.tensor(np.stack(tr.next_state.to_numpy()), dtype=torch.float32)
    mu, sd = S.mean(0), S.std(0) + 1e-6

    wm = WM(sd_dim, ad_dim, int(cfg["hidden"]), mu=mu, sd=sd).to(dev)
    opt = torch.optim.AdamW(wm.parameters(), lr=cfg["lr"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = StepTimer(out_dir, "world_model")
    for _step in range(cfg["steps"]):
        timer.start()
        i = torch.randint(0, len(S), (cfg["batch_size"],))
        s, a, n = S[i].to(dev), A[i].to(dev), N[i].to(dev)
        timer.mark("data")
        loss = nn.functional.mse_loss((wm(s, a) - s) / wm.sd, (n - s) / wm.sd)
        opt.zero_grad()
        loss.backward()
        opt.step()
        timer.mark("fwd_bwd_opt")
        timer.end(len(i), loss=round(loss.item(), 5))
    prof = timer.close()

    # Open-loop rollouts on chronologically sorted held-out episodes
    te = df[df.split == "test"]
    k = int(cfg.get("rollout_k", 10))
    errs: list[float] = []
    base: list[float] = []
    wm.eval()
    with torch.no_grad():
        for (_repo, _ep), g_raw in te.groupby(["repo", "episode"], sort=False):
            g = _sort_episode_frames(g_raw)
            if len(g) <= k:
                continue
            st_arr = np.stack(g.state.to_numpy())
            ac_arr = np.stack(g.action.to_numpy())
            for t0 in range(0, len(g) - k, k):
                s = torch.tensor(st_arr[t0], dtype=torch.float32, device=dev)
                for j in range(k):
                    s = wm(s[None], torch.tensor(ac_arr[t0 + j], dtype=torch.float32, device=dev)[None])[0]
                true = torch.tensor(st_arr[t0 + k], dtype=torch.float32, device=dev)
                errs.append(float((((s - true) / wm.sd) ** 2).mean().item()))
                base.append(float((((torch.tensor(st_arr[t0], dtype=torch.float32, device=dev) - true) / wm.sd) ** 2).mean().item()))

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": wm.state_dict(),
            "mu": mu.cpu(),
            "sd": sd.cpu(),
            "state_dim": sd_dim,
            "action_dim": ad_dim,
            "hidden": int(cfg["hidden"]),
        },
        ckpt_path,
    )
    res = {
        "rollout_k": k,
        "rollout_norm_mse": round(float(np.mean(errs)), 5) if errs else None,
        "copy_state_baseline": round(float(np.mean(base)), 5) if base else None,
        "n_rollouts": len(errs),
        "checkpoint": str(ckpt_path),
        "profile": prof,
    }
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
