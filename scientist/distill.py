"""Distillation: big teacher LLM -> small student LLM on our task texts (completion-masked KL + CE).

Fixes over legacy implementation:
1. Completion-only masking: prompt tokens receive zero CE and zero KL gradient.
2. Vocabulary safety: forbids silent `[:min(V_t, V_s)]` slicing across mismatched tokenizers
   unless `vocab_projection` (`[V_s, V_t]` X-Token alignment matrix) is explicitly provided
   or the difference is pure LM-head alignment padding (`allow_padding_diff=512`).
3. Supports `forward`, `reverse`, and `mixed` KL plus optional `topk` logit distillation.

Usage: python scientist/distill.py [configs/scientist/distill_a100.yaml] [--smoke]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def encode_chat_with_completion_mask(tokenizer, prompt_msgs: list[dict[str, str]], answer_text: str, max_len: int) -> dict[str, list[int]]:
    """Encode `(prompt_msgs, answer_text)` with `completion_mask = 0` on prompt and `1` on assistant answer."""
    prompt_str = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_str = tokenizer.apply_chat_template(
        prompt_msgs + [{"role": "assistant", "content": answer_text}],
        tokenize=False,
    )
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    full_ids = tokenizer.encode(full_str, add_special_tokens=False)[:max_len]
    p_len = min(len(prompt_ids), max(len(full_ids) - 1, 0))
    comp_mask = [0] * p_len + [1] * (len(full_ids) - p_len)
    return {"input_ids": full_ids, "completion_mask": comp_mask}


def collate_distill_batch(tokenizer, encoded_rows: list[dict[str, list[int]]], device: str | torch.device) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    width = max(len(r["input_ids"]) for r in encoded_rows)
    ids, attn, comp = [], [], []
    for r in encoded_rows:
        pad = width - len(r["input_ids"])
        ids.append(r["input_ids"] + [pad_id] * pad)
        attn.append([1] * len(r["input_ids"]) + [0] * pad)
        comp.append(r["completion_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
        "completion_mask": torch.tensor(comp, dtype=torch.float32, device=device),
    }


def distill_step_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
    kl_type: str = "forward",
    mixed_kl_weight: float = 0.5,
    topk: int | None = None,
    vocab_projection: torch.Tensor | None = None,
    same_tokenizer_family: bool = True,
    allow_padding_diff: int = 512,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute completion-masked logit distillation + CE loss."""
    s_next = student_logits[:, :-1, :].float()
    t_next = teacher_logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    mask = completion_mask[:, 1:].bool()

    if not bool(mask.any()):
        zero = student_logits.sum() * 0.0
        return zero, {"loss": 0.0, "kl": 0.0, "ce": 0.0, "n_tokens": 0.0}

    ce_per_tok = F.cross_entropy(s_next.reshape(-1, s_next.shape[-1]), targets.reshape(-1), reduction="none").view_as(targets)
    ce_loss = ce_per_tok[mask].mean()

    v_s, v_t = s_next.shape[-1], t_next.shape[-1]
    T = float(temperature)

    if vocab_projection is not None:
        # Cross-tokenizer X-Token alignment projection: M has shape [V_s, V_t]
        if vocab_projection.shape != (v_s, v_t):
            raise ValueError(f"vocab_projection shape {tuple(vocab_projection.shape)} must match (V_s={v_s}, V_t={v_t})")
        s_probs = torch.softmax(s_next[mask] / T, dim=-1) @ vocab_projection.to(s_next.device, dtype=torch.float32)
        s_logp = torch.log(s_probs.clamp(min=1e-10))
        t_logp = F.log_softmax(t_next[mask] / T, dim=-1)
    else:
        if not same_tokenizer_family or abs(v_t - v_s) > allow_padding_diff:
            raise ValueError(
                f"Incompatible vocab sizes (student={v_s}, teacher={v_t}, same_tokenizer_family={same_tokenizer_family}) "
                "without an explicit `vocab_projection` alignment matrix."
            )
        v_shared = min(v_s, v_t)
        s_masked = s_next[mask][..., :v_shared]
        t_masked = t_next[mask][..., :v_shared]
        if topk is not None and 0 < topk < v_shared:
            t_top_vals, top_idx = torch.topk(t_masked, k=topk, dim=-1)
            s_top_vals = torch.gather(s_masked, dim=-1, index=top_idx)
            s_logp = F.log_softmax(s_top_vals / T, dim=-1)
            t_logp = F.log_softmax(t_top_vals / T, dim=-1)
        else:
            s_logp = F.log_softmax(s_masked / T, dim=-1)
            t_logp = F.log_softmax(t_masked / T, dim=-1)

    fwd_kl = F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean") * (T * T)
    rev_kl = F.kl_div(t_logp, s_logp, log_target=True, reduction="batchmean") * (T * T)
    if kl_type == "forward":
        kl_loss = fwd_kl
    elif kl_type == "reverse":
        kl_loss = rev_kl
    elif kl_type == "mixed":
        kl_loss = float(mixed_kl_weight) * fwd_kl + (1.0 - float(mixed_kl_weight)) * rev_kl
    else:
        raise ValueError(f"Unknown kl_type: {kl_type!r}")

    loss = float(alpha) * kl_loss + (1.0 - float(alpha)) * ce_loss
    return loss, {
        "loss": float(loss.detach()),
        "kl": float(kl_loss.detach()),
        "ce": float(ce_loss.detach()),
        "n_tokens": float(mask.sum().item()),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from labpt.config import check_gpu, training_config, training_rows
    from labpt.profiling import StepTimer

    cfg = training_config("configs/scientist/distill_a100.yaml", argv=argv)
    check_gpu(cfg)
    torch.manual_seed(0)
    dev = "cpu" if cfg.get("device") == "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (cfg.get("bf16") and dev == "cuda") else torch.float32
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["student"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher_tok = AutoTokenizer.from_pretrained(cfg["teacher"])
    same_family = type(tok).__name__ == type(teacher_tok).__name__

    teacher = AutoModelForCausalLM.from_pretrained(cfg["teacher"], dtype=dtype).to(dev).eval().requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(cfg["student"], dtype=dtype).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"])

    rows = [json.loads(line) for line in open(cfg["train_file"], encoding="utf-8")]
    rows = [r for r in rows if r["task"] in cfg["tasks"]]
    rows = training_rows(rows, cfg)
    encoded = [encode_chat_with_completion_mask(tok, r["prompt"], r["answer"], cfg.get("max_len", 512)) for r in rows]

    T, alpha = cfg.get("temperature", 2.0), cfg.get("alpha", 0.5)
    kl_type = cfg.get("kl_type", "forward")
    topk = cfg.get("topk_logits_k", None)
    timer = StepTimer(out, "distill")

    for step in range(cfg["steps"]):
        timer.start()
        sub = [encoded[(step * cfg["batch_size"] + j) % len(encoded)] for j in range(cfg["batch_size"])]
        batch = collate_distill_batch(tok, sub, dev)
        timer.mark("data")
        with torch.no_grad():
            tl = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        timer.mark("teacher_fwd")
        sl = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss, stats = distill_step_loss(
            sl,
            tl,
            batch["input_ids"],
            batch["completion_mask"],
            temperature=T,
            alpha=alpha,
            kl_type=kl_type,
            topk=topk,
            same_tokenizer_family=same_family,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        timer.mark("student_fwd_bwd_opt")
        timer.end(
            int(stats["n_tokens"]),
            loss=round(stats["loss"], 4),
            kl=round(stats["kl"], 4),
            ce=round(stats["ce"], 4),
        )

    prof = timer.close()
    student.save_pretrained(out / "student")
    res = {
        "teacher": cfg["teacher"],
        "student": cfg["student"],
        "steps": cfg["steps"],
        "first": {k: timer.rows[0][k] for k in ("loss", "kl", "ce")},
        "last": {k: timer.rows[-1][k] for k in ("loss", "kl", "ce")},
        "profile": prof,
    }
    json.dump(res, open(out / "metrics.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "profile"}))
    return res


if __name__ == "__main__":
    main()
