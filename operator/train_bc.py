"""Behavior cloning: MLP maps robot state -> action. Usage: python operator/train_bc.py configs/operator/bc_smoke.yaml"""
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

cfg = training_config('configs/operator/bc_a100.yaml')
check_gpu(cfg)
torch.manual_seed(0)
dev = "cpu" if cfg.get("device") == "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_parquet(cfg["frames"])
df = df[df.repo == cfg["repo"]]  # one robot per run (state dims differ across robots)
SD, AD = len(df.state.iloc[0]), len(df.action.iloc[0])


def tensors(split):
    d = df[df.split == split]
    return (torch.tensor(np.stack(d.state.to_numpy()), dtype=torch.float32),
            torch.tensor(np.stack(d.action.to_numpy()), dtype=torch.float32))


Xtr, Ytr = tensors("train")
Xte, Yte = tensors("test")
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
model = nn.Sequential(nn.Linear(SD, cfg["hidden"]), nn.ReLU(), nn.Linear(cfg["hidden"], cfg["hidden"]), nn.ReLU(),
                      nn.Linear(cfg["hidden"], AD)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
timer = StepTimer(cfg["output_dir"], "bc")
for step in range(cfg["steps"]):
    timer.start()
    idx = torch.randint(0, len(Xtr), (cfg["batch_size"],))
    x, y = ((Xtr[idx] - mu) / sd).to(dev), Ytr[idx].to(dev)
    timer.mark("data")
    loss = nn.functional.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    timer.mark("fwd_bwd_opt")
    timer.end(len(idx), loss=round(loss.item(), 5))
prof = timer.close()
with torch.no_grad():
    pred = model(((Xte - mu) / sd).to(dev)).cpu()
    mse = nn.functional.mse_loss(pred, Yte).item()
    base = nn.functional.mse_loss(Ytr.mean(0).expand_as(Yte), Yte).item()  # predict-the-mean baseline
res = {"test_mse": round(mse, 5), "mean_baseline_mse": round(base, 5), "n_train": len(Xtr), "n_test": len(Xte), "profile": prof}
Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
json.dump(res, open(Path(cfg["output_dir"]) / "metrics.json", "w"), indent=1)
print(json.dumps(res))
