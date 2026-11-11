"""Padding and generation helpers shared by the trainers.

Generation is left-padded so a batched causal model does not continue from a pad token.
Training batches are right-padded. The two are separate on purpose.
"""
from __future__ import annotations

import torch


def pad_right(seqs: list[list[int]], pad_id: int):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = 1
    return ids, mask


def encode_pair(tok, prompt: str, completion: str, max_len: int):
    """Token ids for prompt+completion, and labels with the prompt set to -100.

    The label at the last prompt position is the first completion token, not -100.
    That position is what teaches the model to start the answer. Masking it would
    drop the first answer token from the loss. See docs/posttrain/losses_and_gradients.md.
    """
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    comp_ids = tok(completion, add_special_tokens=False)["input_ids"]
    comp_ids = comp_ids + [tok.eos_token_id]
    # Keep at least one completion token if the prompt is very long.
    room = max_len - 1
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]
    room = max_len - len(prompt_ids)
    comp_ids = comp_ids[:room]
    input_ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + comp_ids
    return input_ids, labels, len(prompt_ids)


def collate_labeled(pairs, tok, max_len: int = 64):
    encoded = [encode_pair(tok, p, c, max_len) for p, c in pairs]
    ids, mask = pad_right([e[0] for e in encoded], tok.pad_token_id)
    label_rows = []
    width = ids.shape[1]
    for e in encoded:
        row = e[1] + [-100] * (width - len(e[1]))
        label_rows.append(row)
    labels = torch.tensor(label_rows, dtype=torch.long)
    prompt_lens = [e[2] for e in encoded]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels, "prompt_lens": prompt_lens}


def completion_logprob_mask(prompt_lens, attention_mask):
    """Mask over the shifted log-prob axis: 1 when the predicted token is a completion token."""
    width = attention_mask.shape[1] - 1
    positions = torch.arange(width).unsqueeze(0)
    prompt = torch.tensor(prompt_lens, dtype=torch.long).unsqueeze(1)
    valid = attention_mask[:, 1:] > 0
    # log-prob index t predicts token t+1. Token t+1 is a completion token when t+1 >= prompt_len.
    is_completion = (positions + 1) >= prompt
    return (is_completion & valid).to(dtype=torch.float)


@torch.no_grad()
def generate_texts(model, tok, prompts: list[str], max_new_tokens: int = 4, temperature: float = 1.0, do_sample: bool = True):
    """Return one dict per prompt: text, prompt_ids, completion_ids.

    The model is set back to train mode by the caller if needed. This function leaves it in eval.
    """
    was_training = model.training
    model.eval()
    previous_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        kwargs = dict(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            kwargs["temperature"] = temperature
        out = model.generate(**kwargs)
        seqs = out.sequences if hasattr(out, "sequences") else out
        start = enc["input_ids"].shape[1]
        fresh = seqs[:, start:]
        rows = []
        for i, prompt in enumerate(prompts):
            prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            comp = fresh[i].tolist()
            if tok.pad_token_id in comp:
                comp = comp[: comp.index(tok.pad_token_id)]
            rows.append({
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "completion_ids": comp,
                "text": tok.decode(comp, skip_special_tokens=True),
            })
        return rows
    finally:
        tok.padding_side = previous_side
        if was_training:
            model.train()
