"""Checks for memorisation and benchmark contamination.

These functions answer "had the model already seen this question?" They do not train.

Shao et al. use AIME 2025 as a control because it was released after the models they
study (Figure 13 caption). A high score on a post-cutoff set is harder to explain as
memorisation of those items. A gain that shows up only on older items is the suspicious
one. The holdout list itself is a config input (configs/posttrain/aime_holdout.yaml).
This file does not ship AIME questions and does not claim an AIME score.

LAB-Bench text is evaluation only (CC-BY-SA). Do not pass those strings to a trainer.
The canary check refuses a training corpus that contains the LAB-Bench canary.

The CPU demo overfits a tiny random model on a toy sentence. That is a real run of
the metric code. It is not a statement about any lab model.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from posttrain.hacking_checks import CANARY_NEEDLES, find_canaries

LABBENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "labbench"
PREFIX_FRACS = (0.10, 0.25, 0.50, 0.75)


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+|\S", text or "")


def lcs_length(a: list, b: list) -> int:
    if not a or not b:
        return 0
    # One row of the DP is enough, and short strings are all we score in tests.
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def rouge_l(pred_tokens: list, gold_tokens: list) -> float:
    """F1 of the longest common subsequence. This is the simple ROUGE-L."""
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    match = lcs_length(pred_tokens, gold_tokens)
    prec = match / len(pred_tokens)
    rec = match / len(gold_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def continuation_metrics(predicted: str, gold: str) -> dict:
    """How much of the true continuation the model reproduced.

    exact: full string match after stripping.
    normalized_edit: token edit distance / max(length). 0 is a perfect copy.
    rouge_l: token LCS F1.
    lcs_ratio: LCS length / gold length.
    token_overlap: |intersection| / |gold tokens|, a bag-of-tokens score.
    """
    pred_s = (predicted or "").strip()
    gold_s = (gold or "").strip()
    pred_t = tokens(pred_s)
    gold_t = tokens(gold_s)
    edit = levenshtein(pred_t, gold_t)
    denom = max(len(pred_t), len(gold_t), 1)
    lcs = lcs_length(pred_t, gold_t)
    gold_set = set(gold_t)
    overlap = (len(gold_set & set(pred_t)) / len(gold_set)) if gold_set else 0.0
    return {
        "exact": float(pred_s == gold_s and gold_s != ""),
        "normalized_edit": edit / denom,
        "rouge_l": rouge_l(pred_t, gold_t),
        "lcs_ratio": lcs / max(len(gold_t), 1),
        "token_overlap": overlap,
    }


def shuffle_text(text: str, seed: int = 0) -> str:
    """A control passage: same tokens, wrong order. Easy text should not pass this."""
    parts = text.split()
    rng = random.Random(seed)
    rng.shuffle(parts)
    return " ".join(parts)


def perturb_text(text: str, seed: int = 0) -> str:
    """Drop and swap a few tokens. A memorised copy should do worse on this than on the original."""
    parts = text.split()
    rng = random.Random(seed)
    if len(parts) > 4:
        i, j = rng.randrange(len(parts)), rng.randrange(len(parts))
        parts[i], parts[j] = parts[j], parts[i]
        del parts[rng.randrange(len(parts))]
    return " ".join(parts)


def min_k_mean(token_logprobs, k_frac: float = 0.2) -> float:
    """Mean of the lowest k% token log-probs (Shi et al. style, the simple version).

    Members of the training set tend to have higher (less negative) values. The
    weakness: a hard paraphrase is also more negative, and a short easy sentence is
    less negative, whether or not it was memorised. Without a threshold fit on known
    non-members, a single score is not a membership decision.
    """
    vals = sorted(float(x) for x in token_logprobs)
    if not vals:
        return float("nan")
    k = max(1, int(round(len(vals) * k_frac)))
    chosen = vals[:k]
    return sum(chosen) / len(chosen)


def membership_gap(original_logprobs, paraphrase_logprobs, k_frac: float = 0.2) -> dict:
    a = min_k_mean(original_logprobs, k_frac)
    b = min_k_mean(paraphrase_logprobs, k_frac)
    return {"min_k_original": a, "min_k_paraphrase": b, "gap": a - b}


def option_order_effect(rows: list[dict]) -> dict:
    """rows: {correct_original: bool, correct_shuffled: bool, letter_same: bool, content_same: bool}.

    A sharp drop after shuffling options, with the letter staying put, fits a memorised
    letter. Accuracy that stays, with the content followed to its new letter, does not.
    """
    if not rows:
        return {"n": 0, "acc_original": None, "acc_shuffled": None, "drop": None, "letter_stuck_rate": None}
    n = len(rows)
    acc_o = sum(bool(r["correct_original"]) for r in rows) / n
    acc_s = sum(bool(r["correct_shuffled"]) for r in rows) / n
    stuck = sum(bool(r.get("letter_same")) and not bool(r.get("content_same")) for r in rows) / n
    return {
        "n": n,
        "acc_original": acc_o,
        "acc_shuffled": acc_s,
        "drop": acc_o - acc_s,
        "letter_stuck_rate": stuck,
    }


def canary_in_outputs(texts) -> dict:
    found = find_canaries(texts)
    return {"hit": bool(found), "n_needles": len(found), "n_texts": len(list(texts))}


def assert_training_corpus_clean(texts, where: str = "training corpus") -> None:
    found = find_canaries(texts)
    if found:
        raise RuntimeError(f"{where} contains a LAB-Bench canary ({len(found)} needles). Refusing to train.")


@torch.no_grad()
def token_logprobs(model, tok, text: str) -> list[float]:
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.shape[1] < 2:
        return [0.0]
    logits = model(ids).logits[:, :-1, :]
    logp = F.log_softmax(logits, dim=-1)
    target = ids[:, 1:]
    gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in gathered[0].tolist()]


@torch.no_grad()
def greedy_continue(model, tok, prefix: str, max_new_tokens: int) -> str:
    model.eval()
    ids = tok(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    seqs = out.sequences if hasattr(out, "sequences") else out
    fresh = seqs[0, ids.shape[1]:].tolist()
    return tok.decode(fresh, skip_special_tokens=True)


def prefix_report(model, tok, texts: list[str], fracs=PREFIX_FRACS, max_new_tokens: int = 32) -> dict:
    """Give the model the first X% of each text and score the continuation.

    Also scores a shuffled control and a perturbed control built from the same text.
    A high score on the original and a low score on both controls is the memorisation
    pattern. A high score on all three is just easy text.
    """
    rows = []
    for text_i, text in enumerate(texts):
        words = text.split()
        if len(words) < 4:
            continue
        for frac in fracs:
            cut = max(1, min(len(words) - 1, int(round(len(words) * frac))))
            prefix = " ".join(words[:cut])
            rest = " ".join(words[cut:])
            pred = greedy_continue(model, tok, prefix, max_new_tokens=max(8, min(max_new_tokens, len(rest.split()) + 4)))
            metrics = continuation_metrics(pred, rest)
            rows.append({"kind": "original", "frac": frac, "text": text_i, **metrics})
        for kind, alt in (
            ("shuffled", shuffle_text(text, seed=text_i)),
            ("perturbed", perturb_text(text, seed=text_i)),
        ):
            alt_words = alt.split()
            cut = max(1, len(alt_words) // 2)
            prefix = " ".join(alt_words[:cut])
            rest = " ".join(alt_words[cut:])
            pred = greedy_continue(model, tok, prefix, max_new_tokens=max(8, len(rest.split()) + 4))
            metrics = continuation_metrics(pred, rest)
            rows.append({"kind": kind, "frac": 0.5, "text": text_i, **metrics})
    return {"rows": rows, "summary": _summarise(rows)}


def _summarise(rows: list[dict]) -> dict:
    out = {}
    for kind in ("original", "shuffled", "perturbed"):
        subset = [r for r in rows if r["kind"] == kind]
        if not subset:
            out[kind] = None
            continue
        out[kind] = {
            "n": len(subset),
            "exact": sum(r["exact"] for r in subset) / len(subset),
            "rouge_l": sum(r["rouge_l"] for r in subset) / len(subset),
            "normalized_edit": sum(r["normalized_edit"] for r in subset) / len(subset),
            "token_overlap": sum(r["token_overlap"] for r in subset) / len(subset),
        }
    return out


def assert_not_labbench_training_path(path) -> None:
    """LAB-Bench files are eval only. A trainer must not open them as a training set."""
    resolved = Path(path).resolve()
    root = LABBENCH_DIR.resolve()
    if resolved == root or root in resolved.parents:
        raise RuntimeError(f"{path} is LAB-Bench. It is eval only and must not be used as training data.")


def load_labbench_questions(limit: int | None = None) -> list[dict]:
    """Public LAB-Bench question text, for evaluation only.

    Do not pass these strings to a trainer. The source lines contain the canary.
    """
    rows = []
    for path in sorted(LABBENCH_DIR.glob("*-public.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("question") or ""
            if row.get("protocol"):
                text = str(row["protocol"]) + "\n" + text
            rows.append({
                "id": row.get("id"),
                "file": path.name,
                "text": text,
                "ideal": row.get("ideal"),
                "has_canary_in_source_line": any(n in line for n in CANARY_NEEDLES),
            })
            if limit and len(rows) >= limit:
                return rows
    return rows


def overfit_toy_sentence(model, tok, text: str, steps: int = 40, lr: float = 3e-3):
    """Fit the tiny model to one sentence so the prefix demo has a positive control.

    This is a real optimisation on CPU. The sentence is not from a benchmark.
    """
    from posttrain.batching import collate_labeled
    from posttrain.sft import sft_nll

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # One-token prompt so the rest of the sentence is the completion the loss trains.
    words = text.split()
    prompt = words[0]
    completion = " " + " ".join(words[1:])
    batch = collate_labeled([(prompt, completion)], tok, max_len=96)
    last = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss, _ = sft_nll(logits, batch["labels"], normalize="token")
        loss.backward()
        opt.step()
        last = float(loss.detach().item())
    model.eval()
    return last


def toy_demo(seed: int = 0) -> dict:
    """CPU demo. Returns measured metrics. Does not claim a benchmark result."""
    from posttrain.tiny import build_tiny

    torch.manual_seed(seed)
    text = "mix buffer then warm it well"
    model, tok = build_tiny(seed=seed, n_positions=64)
    # A control model that was not fit to the sentence.
    cold, _ = build_tiny(seed=seed + 1, n_positions=64)
    loss = overfit_toy_sentence(model, tok, text, steps=40, lr=5e-3)
    fitted = prefix_report(model, tok, [text], fracs=(0.5,), max_new_tokens=24)
    cold_report = prefix_report(cold, tok, [text], fracs=(0.5,), max_new_tokens=24)
    paraphrase = "heat the solution up to body temperature and stir it briefly"
    gap = membership_gap(token_logprobs(model, tok, text), token_logprobs(model, tok, paraphrase))
    return {
        "status": "toy demo, not a benchmark",
        "overfit_loss": loss,
        "fitted": fitted["summary"],
        "unfitted": cold_report["summary"],
        "membership_gap": gap,
        "canary_in_toy_text": canary_in_outputs([text])["hit"],
    }
