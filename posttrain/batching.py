"""Token packing helpers shared by the trainers.

Packing several completions into one causal sequence is not implemented.
A GPT-2 attention mask is a padding mask, not a block mask, so example B
would attend into example A. We pad and mask instead.
"""
from __future__ import annotations

import copy

import torch


def clone_frozen(model: torch.nn.Module) -> torch.nn.Module:
    """A reference policy. No gradient, no dropout difference (dropout is 0)."""
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def snapshot_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def states_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def encode_example(tokenizer, prompt: str, completion: str, max_length: int):
    """Prompt tokens then completion tokens, then eos. Truncate the prompt first."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    comp_ids = tokenizer.encode(completion, add_special_tokens=False)
    comp_ids = comp_ids + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(comp_ids) > max_length:
        comp_budget = min(len(comp_ids), max(1, max_length - min(len(prompt_ids), max(1, max_length // 2))))
        comp_ids = comp_ids[:comp_budget]
        prompt_budget = max(1, max_length - len(comp_ids))
        prompt_ids = prompt_ids[-prompt_budget:]
    ids = prompt_ids + comp_ids
    # 1 on completion tokens, including eos. 0 on the prompt.
    comp_mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
    return ids, comp_mask


def model_device(model: torch.nn.Module) -> torch.device:
    """Device of the first parameter or buffer; CPU if parameterless."""
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def model_max_length(model: torch.nn.Module, tokenizer=None, fallback: int = 4096) -> int:
    """Context window across GPT-2, Qwen2.5/3, Llama, and OLMo configs."""
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and 0 < val < 10_000_000:
            return val
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 10_000_000:
        return tok_max
    return fallback


def model_hidden_size(model: torch.nn.Module) -> int:
    """Hidden dimension across GPT-2 (n_embd) and Qwen/Llama (hidden_size)."""
    cfg = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    raise AttributeError("Cannot infer hidden size from model.config")


def collate(tokenizer, pairs: list[tuple[list[int], list[int]]], device="cpu"):
    width = max(len(ids) for ids, _ in pairs)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids, attn, comp = [], [], []
    for ids, mask in pairs:
        pad_n = width - len(ids)
        input_ids.append(ids + [pad] * pad_n)
        attn.append([1] * len(ids) + [0] * pad_n)
        comp.append(mask + [0] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
        "completion_mask": torch.tensor(comp, dtype=torch.float32, device=device),
    }


def forward_logprobs(model, batch):
    """Token log-probs of the completion, with gradient, plus full logits."""
    from posttrain.losses import completion_mask_shifted, token_logprobs

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    lp = token_logprobs(out.logits, batch["input_ids"])
    mask = completion_mask_shifted(batch["completion_mask"])
    return out, lp, mask


@torch.no_grad()
def frozen_logprobs(model, batch):
    was = model.training
    model.eval()
    _, lp, mask = forward_logprobs(model, batch)
    if was:
        model.train()
    return lp.detach(), mask


def sequence_logprob(token_lp: torch.Tensor, mask: torch.Tensor, average: bool) -> torch.Tensor:
    """Sum (or mean) of completion token log-probs. One number per row.

    The sum gets more negative as the completion gets longer, even when
    every token is equally likely. That is the DPO length bias. `average`
    True divides by the length so a long answer is not punished just for
    being long. The DPO trainer exposes this as `average_logprobs`.
    """
    mask = mask.to(token_lp.dtype)
    total = (token_lp * mask).sum(dim=-1)
    if not average:
        return total
    return total / mask.sum(dim=-1).clamp(min=1.0)


def maybe_lora(model, use_lora: bool, r: int | None = None, lora_alpha: int | None = None):
    if not use_lora:
        return model
    from peft import LoraConfig, get_peft_model

    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if "c_attn" in module_names:
        # GPT-2's attention projection is a Conv1D. peft wants fan_in_fan_out=True.
        target_modules = ["c_attn"]
        fan_in_fan_out = True
        default_r, default_alpha = 4, 8
    else:
        linear_candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        matched = [m for m in linear_candidates if m in module_names]
        target_modules = matched if matched else "all-linear"
        fan_in_fan_out = False
        default_r, default_alpha = 8, 16

    cfg = LoraConfig(
        r=r if r is not None else default_r,
        lora_alpha=lora_alpha if lora_alpha is not None else default_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=fan_in_fan_out,
    )
    return get_peft_model(model, cfg)


def trainable_count(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
