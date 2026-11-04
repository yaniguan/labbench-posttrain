"""Simple world model: predict next robot state from (state, action). Used for cheap rollouts.
Reports 1-step MSE and k-step open-loop rollout error on held-out episodes vs a 'state stays the same' baseline.
Usage: python operator/world_model.py configs/operator/wm_smoke.yaml"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, "src")
from labpt.config import check_gpu, training_config
from labpt.profiling import StepTimer

cfg = training_config('configs/operator/wm_a100.yaml')
check_gpu(cfg)
torch.manual_seed(0)
dev = "cpu" if cfg.get("device") == "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_parquet(cfg["frames"])
df = df[df.repo == cfg["repo"]]  # one robot per run (state dims differ across robots)
SD, AD = len(df.state.iloc[0]), len(df.action.iloc[0])
tr = df[df.split == "train"]
S = torch.tensor(np.stack(tr.state.to_numpy()), dtype=torch.float32)
A = torch.tensor(np.stack(tr.action.to_numpy()), dtype=torch.float32)
N = torch.tensor(np.stack(tr.next_state.to_numpy()), dtype=torch.float32)
mu, sd = S.mean(0), S.std(0) + 1e-6
D = (N - S) / sd  # predict normalized delta


class WM(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(SD + AD, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, SD))

    def forward(self, s, a):
        return s + self.net(torch.cat([(s - mu.to(s.device)) / sd.to(s.device), a], -1)) * sd.to(s.device)


wm = WM(cfg["hidden"]).to(dev)
opt = torch.optim.AdamW(wm.parameters(), lr=cfg["lr"])
timer = StepTimer(cfg["output_dir"], "world_model")
for step in range(cfg["steps"]):
    timer.start()
    i = torch.randint(0, len(S), (cfg["batch_size"],))
    s, a, n = S[i].to(dev), A[i].to(dev), N[i].to(dev)
    timer.mark("data")
    loss = nn.functional.mse_loss((wm(s, a) - s) / sd.to(dev), (n - s) / sd.to(dev))
    opt.zero_grad()
    loss.backward()
    opt.step()
    timer.mark("fwd_bwd_opt")
    timer.end(len(i), loss=round(loss.item(), 5))
prof = timer.close()

# open-loop rollouts on held-out episodes
te = df[df.split == "test"]
k = cfg.get("rollout_k", 10)
errs, base = [], []
with torch.no_grad():
    for (repo, ep), g in te.groupby(["repo", "episode"]):
        st = np.stack(g.state.to_numpy())
        ac = np.stack(g.action.to_numpy())
        for t0 in range(0, len(g) - k, k):
            s = torch.tensor(st[t0], dtype=torch.float32, device=dev)
            for j in range(k):
                s = wm(s[None], torch.tensor(ac[t0 + j], dtype=torch.float32, device=dev)[None])[0]
            true = torch.tensor(st[t0 + k], dtype=torch.float32, device=dev)
            errs.append(float((((s - true) / sd.to(dev)) ** 2).mean()))
            base.append(float((((torch.tensor(st[t0], device=dev) - true) / sd.to(dev)) ** 2).mean()))
res = {"rollout_k": k, "rollout_norm_mse": round(float(np.mean(errs)), 5), "copy_state_baseline": round(float(np.mean(base)), 5),
       "n_rollouts": len(errs), "profile": prof}
json.dump(res, open(Path(cfg["output_dir"]) / "metrics.json", "w"), indent=1)
print(json.dumps(res))
