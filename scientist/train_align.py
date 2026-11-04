"""Contrastive alignment (CLIP-style) between a non-text modality and text.
  pair 'mol_text':      ChEMBL/PubChem molecule <-> assay/target text
  pair 'spectrum_text': XRD/HPLC/IR/growth curve <-> method + outcome text
Reports held-out retrieval recall@1/@5 (random = k/N).
Usage: python scientist/train_align.py configs/scientist/align_smoke.yaml"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "src")
sys.path.insert(0, "scientist")
import data
from encoders import MolEncoder, SpectrumEncoder, TextEncoder
from labpt.config import check_gpu, training_config
from labpt.profiling import StepTimer

cfg = training_config('configs/scientist/align_a100.yaml')
check_gpu(cfg)
torch.manual_seed(0)
dev = "cpu" if cfg.get("device") == "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
out = Path(cfg["output_dir"])
out.mkdir(parents=True, exist_ok=True)
df = data.load()
res = {}
for pair in cfg["pairs"]:
    if pair == "mol_text":
        X, T, S = data.molecule_text_pairs(df, cfg.get("max_n", 6000))
        enc = MolEncoder(cfg["d"])
    else:
        X, T, S, _ = data.spectra(df)
        enc = SpectrumEncoder(cfg["d"])
    Tx = data.texts_to_array(T)
    tenc = TextEncoder(cfg["d"])
    enc, tenc = enc.to(dev), tenc.to(dev)
    temp = nn.Parameter(torch.tensor(2.0, device=dev))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(tenc.parameters()) + [temp], lr=cfg["lr"])
    Xa, Ta = torch.tensor(X[S == "train"]), torch.tensor(Tx[S == "train"])
    timer = StepTimer(out / pair, pair)
    for step in range(cfg["steps"]):
        timer.start()
        i = torch.randint(0, len(Xa), (min(cfg["batch_size"], len(Xa)),))
        x, t = Xa[i].to(dev), Ta[i].to(dev)
        timer.mark("data")
        a = nn.functional.normalize(enc(x), dim=-1)
        b = nn.functional.normalize(tenc(t), dim=-1)
        logits = a @ b.T * temp.exp()
        lab = torch.arange(len(x), device=dev)
        loss = (nn.functional.cross_entropy(logits, lab) + nn.functional.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
        timer.mark("fwd_bwd_opt")
        timer.end(len(x), loss=round(loss.item(), 4))
    prof = timer.close()
    with torch.no_grad():
        xt, tt = torch.tensor(X[S == "test"]).to(dev), torch.tensor(Tx[S == "test"]).to(dev)
        n = min(len(xt), 256)
        a = nn.functional.normalize(enc(xt[:n]), dim=-1)
        b = nn.functional.normalize(tenc(tt[:n]), dim=-1)
        rank = (a @ b.T).argsort(dim=1, descending=True)
        hit = rank == torch.arange(n, device=dev)[:, None]
        r1, r5 = float(hit[:, :1].any(1).float().mean()), float(hit[:, :5].any(1).float().mean())
    res[pair] = {"n_train": int(len(Xa)), "n_test_eval": n, "recall@1": round(r1, 4), "recall@5": round(r5, 4),
                 "random_recall@1": round(1 / n, 4), "random_recall@5": round(5 / n, 4), "final_loss": round(loss.item(), 4), "profile": prof}
    torch.save({"enc": enc.state_dict(), "text": tenc.state_dict()}, out / f"{pair}.pt")
json.dump(res, open(out / "metrics.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "profile"} for k, v in res.items()}, indent=1))
