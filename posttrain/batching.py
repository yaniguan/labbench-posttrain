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
    room = max_length - 1
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]
    comp_ids = comp_ids[: max_length - len(prompt_ids)]
    ids = prompt_ids + comp_ids
    # 1 on completion tokens, including eos. 0 on the prompt.
    comp_mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
    return ids, comp_mask


def collate(tokenizer, pairs: list[tuple[list[int], list[int]]], device="cpu"):
    width = max(len(ids) for ids, _ in pairs)
    pad = tokenizer.pad_token_id
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


def maybe_lora(model, use_lora: bool):
    if not use_lora:
        return model
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        # GPT-2's attention projection is a Conv1D. peft wants this flag or it warns.
        fan_in_fan_out=True,
    )
    return get_peft_model(model, cfg)


def trainable_count(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
