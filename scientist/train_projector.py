"""Multimodal post-training: frozen molecule encoder -> projector -> k soft tokens prepended to an LLM.
The LLM learns to answer questions about the molecule (QM9 DFT properties) from the soft tokens only
(the SMILES is NOT in the prompt). Trains the projector (+ optional LoRA).
Usage: python scientist/train_projector.py configs/scientist/projector_smoke.yaml"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
sys.path.insert(0, "scientist")
import data
from encoders import MolEncoder
from labpt.config import check_gpu, training_config
from labpt.profiling import StepTimer

cfg = training_config('configs/scientist/projector_a100.yaml')
check_gpu(cfg)
torch.manual_seed(0)
dev = "cpu" if cfg.get("device") == "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (cfg.get("bf16") and dev == "cuda") else torch.float32
out = Path(cfg["output_dir"])
out.mkdir(parents=True, exist_ok=True)

X, Y, S, smiles = data.molecules(data.load())
props = ["gap", "homo", "lumo", "mu", "u0"]
enc = MolEncoder(cfg["d"])
if Path(cfg.get("encoder_ckpt", "")).is_file():
    enc.load_state_dict(torch.load(cfg["encoder_ckpt"]))
    print("loaded encoder", cfg["encoder_ckpt"])
enc = enc.to(dev).eval().requires_grad_(False)

tok = AutoTokenizer.from_pretrained(cfg["model"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype).to(dev)
if cfg.get("lora"):
    from peft import LoraConfig, get_peft_model
    llm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
else:
    llm.requires_grad_(False)
H, K = llm.get_input_embeddings().weight.shape[1], cfg["n_soft_tokens"]
proj = nn.Sequential(nn.Linear(cfg["d"], H), nn.GELU(), nn.Linear(H, H * K)).to(dev, dtype)
params = list(proj.parameters()) + [p for p in llm.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=cfg["lr"])


def batch(idx):
    qs, ans = [], []
    for i in idx:
        j = np.random.randint(len(props))
        qs.append(f"Question: what is the DFT {props[j]} of this molecule? Answer:")
        ans.append(f" {Y[i, j]:.4f}{tok.eos_token}")
    q = tok(qs, add_special_tokens=False)["input_ids"]
    a = tok(ans, add_special_tokens=False)["input_ids"]
    L = max(len(x) + len(y) for x, y in zip(q, a))
    ids = torch.full((len(idx), L), tok.pad_token_id or 0)
    lab = torch.full((len(idx), L), -100)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    for b, (x, y) in enumerate(zip(q, a)):
        ids[b, :len(x) + len(y)] = torch.tensor(x + y)
        lab[b, len(x):len(x) + len(y)] = torch.tensor(y)
        att[b, :len(x) + len(y)] = 1
    return ids.to(dev), lab.to(dev), att.to(dev)


def forward(idx):
    ids, lab, att = batch(idx)
    soft = proj(enc(torch.tensor(X[idx]).to(dev)).to(dtype)).view(len(idx), K, H)
    emb = torch.cat([soft, llm.get_input_embeddings()(ids)], 1)
    att = torch.cat([torch.ones(len(idx), K, dtype=torch.long, device=dev), att], 1)
    lab = torch.cat([torch.full((len(idx), K), -100, device=dev), lab], 1)
    return llm(inputs_embeds=emb, attention_mask=att, labels=lab).loss


tr, te = np.where(S == "train")[0], np.where(S == "test")[0]
timer = StepTimer(out, "projector")
for step in range(cfg["steps"]):
    timer.start()
    idx = np.random.choice(tr, cfg["batch_size"])
    timer.mark("data")
    loss = forward(idx)
    opt.zero_grad(); loss.backward(); opt.step()
    timer.mark("fwd_bwd_opt")
    timer.end(len(idx), loss=round(loss.item(), 4))
prof = timer.close()
with torch.no_grad():
    test_loss = float(np.mean([forward(te[i:i + 16]).item() for i in range(0, min(len(te), 64), 16)]))
res = {"model": cfg["model"], "n_soft_tokens": K, "steps": cfg["steps"], "first_loss": timer.rows[0]["loss"],
       "last_loss": timer.rows[-1]["loss"], "test_answer_nll": round(test_loss, 4), "profile": prof}
torch.save(proj.state_dict(), out / "projector.pt")
if cfg.get("lora"):
    llm.save_pretrained(out / "adapter")
json.dump(res, open(out / "metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "profile"}))
