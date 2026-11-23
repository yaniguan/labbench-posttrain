"""Multimodal post-training: frozen molecule encoder -> projector -> k soft tokens prepended to an LLM.

The LLM learns to answer questions about the molecule (QM9 DFT properties) from the soft tokens only
(the SMILES is NOT in the prompt). Trains the projector (+ optional LoRA).
Usage:
    python scientist/train_projector.py configs/scientist/projector_smoke.yaml [--smoke]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scientist")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROPS = ["gap", "homo", "lumo", "mu", "u0"]


def build_projector(in_dim: int, hidden_dim: int, n_soft_tokens: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim * n_soft_tokens),
    )


def make_batch(
    idx: np.ndarray,
    Y: np.ndarray,
    tok: Any,
    dev: str | torch.device,
    props: list[str] = PROPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qs, ans = [], []
    for i in idx:
        j = int(np.random.randint(len(props)))
        qs.append(f"Question: what is the DFT {props[j]} of this molecule? Answer:")
        ans.append(f" {Y[i, j]:.4f}{tok.eos_token}")
    q = tok(qs, add_special_tokens=False)["input_ids"]
    a = tok(ans, add_special_tokens=False)["input_ids"]
    L = max(len(x) + len(y) for x, y in zip(q, a))
    ids = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long)
    lab = torch.full((len(idx), L), -100, dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    for b, (x, y) in enumerate(zip(q, a)):
        ids[b, : len(x) + len(y)] = torch.tensor(x + y, dtype=torch.long)
        lab[b, len(x) : len(x) + len(y)] = torch.tensor(y, dtype=torch.long)
        att[b, : len(x) + len(y)] = 1
    return ids.to(dev), lab.to(dev), att.to(dev)


batch = make_batch


def forward(
    idx: np.ndarray,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    tok: Any,
    enc: nn.Module,
    proj: nn.Module,
    llm: nn.Module,
    K: int,
    H: int,
    dev: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    ids, lab, att = make_batch(idx, Y, tok, dev)
    soft = proj(enc(torch.tensor(X[idx]).to(dev)).to(dtype)).view(len(idx), K, H)
    emb = torch.cat([soft, llm.get_input_embeddings()(ids)], 1)
    att = torch.cat([torch.ones(len(idx), K, dtype=torch.long, device=dev), att], 1)
    lab = torch.cat([torch.full((len(idx), K), -100, dtype=torch.long, device=dev), lab], 1)
    return llm(inputs_embeds=emb, attention_mask=att, labels=lab).loss


def main(argv: list[str] | None = None) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from labpt.config import check_gpu, training_config
    from labpt.profiling import StepTimer
    from scientist import data
    from scientist.encoders import MolEncoder

    cfg = training_config("configs/scientist/projector_a100.yaml", argv=argv)
    check_gpu(cfg)
    torch.manual_seed(0)
    dev = "cpu" if cfg.get("device") == "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (cfg.get("bf16") and dev == "cuda") else torch.float32
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    X, Y, S, _smiles = data.molecules(data.load())
    enc = MolEncoder(cfg["d"])
    if Path(cfg.get("encoder_ckpt", "")).is_file():
        enc.load_state_dict(torch.load(cfg["encoder_ckpt"], map_location="cpu", weights_only=False))
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
    H, K = llm.get_input_embeddings().weight.shape[1], int(cfg["n_soft_tokens"])
    proj = build_projector(int(cfg["d"]), H, K).to(dev, dtype)
    params = list(proj.parameters()) + [p for p in llm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])

    def forward_step(idx: np.ndarray) -> torch.Tensor:
        ids, lab, att = make_batch(idx, Y, tok, dev)
        soft = proj(enc(torch.tensor(X[idx]).to(dev)).to(dtype)).view(len(idx), K, H)
        emb = torch.cat([soft, llm.get_input_embeddings()(ids)], 1)
        att = torch.cat([torch.ones(len(idx), K, dtype=torch.long, device=dev), att], 1)
        lab = torch.cat([torch.full((len(idx), K), -100, dtype=torch.long, device=dev), lab], 1)
        return llm(inputs_embeds=emb, attention_mask=att, labels=lab).loss

    tr, te = np.where(S == "train")[0], np.where(S == "test")[0]
    timer = StepTimer(out, "projector")
    for _step in range(cfg["steps"]):
        timer.start()
        idx = np.random.choice(tr, cfg["batch_size"])
        timer.mark("data")
        loss = forward_step(idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        timer.mark("fwd_bwd_opt")
        timer.end(len(idx), loss=round(loss.item(), 4))
    prof = timer.close()
    with torch.no_grad():
        test_loss = float(np.mean([forward_step(te[i : i + 16]).item() for i in range(0, min(len(te), 64), 16)]))
    res = {
        "model": cfg["model"],
        "n_soft_tokens": K,
        "steps": cfg["steps"],
        "first_loss": timer.rows[0]["loss"],
        "last_loss": timer.rows[-1]["loss"],
        "test_answer_nll": round(test_loss, 4),
        "profile": prof,
    }
    torch.save(proj.state_dict(), out / "projector.pt")
    if cfg.get("lora"):
        llm.save_pretrained(out / "adapter")
    (out / "metrics.json").write_text(json.dumps(res, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "profile"}))
    return res


if __name__ == "__main__":
    main()
