"""Representation learning per modality.
  molecules: supervised multi-task regression on QM9 DFT targets (compute provenance)
  spectra:   self-supervised masked-curve reconstruction (XRD/HPLC/IR/growth curves)
  images:    self-supervised denoising autoencoder on microscopy patches
Usage: python scientist/train_encoders.py configs/scientist/encoders_smoke.yaml"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "src")
sys.path.insert(0, "scientist")
import data
from encoders import ImageEncoder, MolEncoder, SpectrumEncoder
from labpt.config import check_gpu, training_config
from labpt.profiling import StepTimer

cfg = training_config('configs/scientist/encoders_a100.yaml')
check_gpu(cfg)
torch.manual_seed(0)
dev = "cpu" if cfg.get("device") == "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
out = Path(cfg["output_dir"])
out.mkdir(parents=True, exist_ok=True)
df = data.load()
res = {}

# ---- molecules
X, Y, S, _ = data.molecules(df)
mu, sd = Y[S == "train"].mean(0), Y[S == "train"].std(0)
Yn = (Y - mu) / sd
enc, head = MolEncoder(cfg["d"]).to(dev), nn.Linear(cfg["d"], Y.shape[1]).to(dev)
opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=cfg["lr"])
Xt, Yt = torch.tensor(X[S == "train"]), torch.tensor(Yn[S == "train"])
timer = StepTimer(out / "mol", "mol_encoder")
for step in range(cfg["steps"]):
    timer.start()
    i = torch.randint(0, len(Xt), (cfg["batch_size"],))
    x, y = Xt[i].to(dev), Yt[i].to(dev)
    timer.mark("data")
    loss = nn.functional.mse_loss(head(enc(x)), y)
    opt.zero_grad(); loss.backward(); opt.step()
    timer.mark("fwd_bwd_opt")
    timer.end(len(i), loss=round(loss.item(), 4))
prof = timer.close()
with torch.no_grad():
    p = head(enc(torch.tensor(X[S == "test"]).to(dev))).cpu().numpy()
mae = (np.abs(p - Yn[S == "test"]) * sd).mean(0)
base = (np.abs(Yn[S == "test"]) * sd).mean(0)  # predict train mean
res["molecules"] = {"n_train": int((S == "train").sum()), "n_test": int((S == "test").sum()),
                    "final_loss": float(loss.item()), "test_mae": dict(zip(["gap", "homo", "lumo", "mu", "u0"], np.round(mae.astype(float), 4).tolist())),
                    "mean_baseline_mae": dict(zip(["gap", "homo", "lumo", "mu", "u0"], np.round(base.astype(float), 4).tolist())), "profile": prof}
torch.save(enc.state_dict(), out / "mol_encoder.pt")

# ---- spectra: masked reconstruction
X, T, S, K = data.spectra(df)
Xt = torch.tensor(X[S == "train"])
enc, dec = SpectrumEncoder(cfg["d"]).to(dev), nn.Linear(cfg["d"], 256).to(dev)
opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=cfg["lr"])
timer = StepTimer(out / "spec", "spectrum_encoder")
for step in range(cfg["steps"]):
    timer.start()
    x = Xt[torch.randint(0, len(Xt), (min(cfg["batch_size"], len(Xt)),))].to(dev)
    mask = (torch.rand_like(x) < 0.3).float()
    timer.mark("data")
    rec = dec(enc(x * (1 - mask)))
    loss = ((rec - x) ** 2 * mask).sum() / mask.sum()
    opt.zero_grad(); loss.backward(); opt.step()
    timer.mark("fwd_bwd_opt")
    timer.end(len(x), loss=round(loss.item(), 4))
prof = timer.close()
with torch.no_grad():
    xt = torch.tensor(X[S == "test"]).to(dev)
    torch.manual_seed(1)
    mask = (torch.rand_like(xt) < 0.3).float()
    test_loss = float((((dec(enc(xt * (1 - mask))) - xt) ** 2 * mask).sum() / mask.sum()))
res["spectra"] = {"n_train": int(len(Xt)), "n_test": int((S == "test").sum()), "test_masked_mse": round(test_loss, 4),
                  "final_loss": float(loss.item()), "zero_baseline_mse": round(float((xt.square() * mask).sum() / mask.sum()), 4), "profile": prof}
torch.save(enc.state_dict(), out / "spectrum_encoder.pt")

# ---- images: denoising AE
X, V, S = data.images(df)
Xt = torch.tensor(X)
enc, dec = ImageEncoder(cfg["d"]).to(dev), nn.Linear(cfg["d"], 32 * 32).to(dev)
opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=cfg["lr"])
timer = StepTimer(out / "img", "image_encoder")
for step in range(cfg["steps"] // 2):
    timer.start()
    x = Xt[torch.randint(0, len(Xt), (32,))].to(dev)
    timer.mark("data")
    loss = nn.functional.mse_loss(dec(enc(x + 0.3 * torch.randn_like(x))).view_as(x), x)
    opt.zero_grad(); loss.backward(); opt.step()
    timer.mark("fwd_bwd_opt")
    timer.end(len(x), loss=round(loss.item(), 4))
res["images"] = {"n_patches": int(len(X)), "final_train_mse": round(loss.item(), 4), "profile": timer.close(),
                 "note": "only 64 patches from one real image in v0; demo of the pipeline, not a useful encoder"}
torch.save(enc.state_dict(), out / "image_encoder.pt")
json.dump(res, open(out / "metrics.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "profile"} for k, v in res.items()}, indent=1))
