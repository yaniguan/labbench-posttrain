"""Contamination and memorization checks.

LAB-Bench text is evaluation only. The CPU demo below uses toy sentences
and the tiny model. A later Colab run can point the same functions at
public LAB-Bench text and BioProBench questions. Do not train on LAB-Bench.

Shao et al. use AIME 2025 as a contamination control because it was
released after the models they trained (Appendix D, Figure 13 caption).
A high score on a post-cutoff set is not evidence the set was in
pretraining. A gap that appears only on pre-cutoff sets can be
contamination, or it can be a harder post-cutoff set. The numbers have
to be measured. This file does not invent them.
"""
from __future__ import annotations

import math
import random

import torch

from posttrain.hacking_checks import CANARY_NEEDLES, canary_in_texts


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """0 means the strings match. 1 means they share nothing of that length."""
    denom = max(len(a), len(b), 1)
    return levenshtein(a, b) / denom


def lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def rouge_l(hypothesis: str, reference: str) -> float:
    """LCS length divided by the reference length. This is ROUGE-L recall."""
    if not reference:
        return 0.0
    return lcs_length(hypothesis, reference) / len(reference)


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of whitespace tokens."""
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def continuation_metrics(predicted: str, gold: str) -> dict:
    exact = float(predicted == gold)
    return {
        "exact_match": exact,
        "normalized_edit_distance": normalized_edit_distance(predicted, gold),
        "rouge_l": rouge_l(predicted, gold),
        "lcs_ratio": rouge_l(predicted, gold),
        "token_overlap": token_overlap(predicted, gold),
    }


def prefix_fractions(text: str, fractions=(0.10, 0.25, 0.50, 0.75)) -> list[tuple[float, str, str]]:
    """Split a question into a prefix and the gold continuation."""
    out = []
    for frac in fractions:
        cut = max(1, min(len(text) - 1, int(round(len(text) * frac))))
        out.append((frac, text[:cut], text[cut:]))
    return out


def score_prefixes(text: str, generate_fn, fractions=(0.10, 0.25, 0.50, 0.75)) -> list[dict]:
    rows = []
    for frac, prefix, gold in prefix_fractions(text, fractions):
        pred = generate_fn(prefix, len(gold))
        metrics = continuation_metrics(pred, gold)
        metrics.update({"fraction": frac, "prefix": prefix, "gold": gold, "predicted": pred})
        rows.append(metrics)
    return rows


def shuffled_control(text: str, seed: int = 0) -> str:
    """A control string with the same characters in a different order.

    A high continuation score on the original and a low score on this
    control is harder to explain as 'the text was easy'. A high score on
    both means the metric is not specific.
    """
    chars = list(text)
    random.Random(seed).shuffle(chars)
    return "".join(chars)


def option_order(options: list[str], answer: str, seed: int = 0) -> dict:
    """Shuffle options. Return the new letter of the same answer text.

    Letters are A, B, C, ... in the new order. A policy that memorised the
    letter will keep the old letter. A policy that read the text will follow
    `new_letter`.
    """
    if answer not in options:
        raise ValueError("answer must be one of the options")
    order = list(range(len(options)))
    random.Random(seed).shuffle(order)
    shuffled = [options[i] for i in order]
    old_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[options.index(answer)]
    new_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[shuffled.index(answer)]
    return {
        "options": shuffled,
        "old_letter": old_letter,
        "new_letter": new_letter,
        "letter_changed": old_letter != new_letter,
    }


def letter_was_memorised(predicted_letter: str, old_letter: str, new_letter: str) -> bool:
    """True when the prediction kept the old letter after the answer moved."""
    return predicted_letter == old_letter and old_letter != new_letter


@torch.no_grad()
def sequence_token_logprobs(model, tokenizer, text: str) -> list[float]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return []
    batch = torch.tensor([ids], dtype=torch.long)
    logits = model(batch).logits
    logp = torch.log_softmax(logits[0, :-1], dim=-1)
    target = batch[0, 1:]
    return [float(logp[i, target[i]]) for i in range(target.shape[0])]


def mean_logprob(token_logprobs: list[float]) -> float | None:
    if not token_logprobs:
        return None
    return sum(token_logprobs) / len(token_logprobs)


def min_k_logprob(token_logprobs: list[float], percent: float = 0.2) -> float | None:
    """Mean log-prob of the least likely `percent` of tokens.

    This is a simple min-k% score (Shi et al.). Lower (more negative) means
    those tokens surprised the model. It is a weak membership signal: a
    paraphrase can be rarer English even when the model never saw the
    original. A gap by itself is not proof of contamination.
    """
    if not token_logprobs:
        return None
    k = max(1, int(math.ceil(len(token_logprobs) * percent)))
    worst = sorted(token_logprobs)[:k]
    return sum(worst) / len(worst)


def membership_gap(original_score: float | None, paraphrase_score: float | None) -> float | None:
    """Positive when the original is more likely than the paraphrase."""
    if original_score is None or paraphrase_score is None:
        return None
    return original_score - paraphrase_score


def interpret_holdout(
    model_release: str | None,
    benchmark_name: str,
    benchmark_release: str | None,
    score_precutoff: float | None,
    score_postcutoff: float | None,
) -> dict:
    """Say what a pair of scores would mean. Do not fill in missing scores.

    Shao et al. (Appendix D, Figure 13 caption): AIME 2025 was released
    after the models, so it is a contamination control. They also note that
    AIME has 30 questions, so gaps under about 2 points may not be significant.
    """
    base = {
        "benchmark": benchmark_name,
        "model_release": model_release,
        "benchmark_release": benchmark_release,
        "score_precutoff": score_precutoff,
        "score_postcutoff": score_postcutoff,
        "status": "not run",
    }
    if score_precutoff is None or score_postcutoff is None:
        base["reading"] = (
            "No scores were passed in. A post-cutoff benchmark can only work "
            "as a control after it is actually run."
        )
        return base
    base["status"] = "scores supplied by caller"
    gap = score_precutoff - score_postcutoff
    base["gap_pre_minus_post"] = gap
    if benchmark_release and model_release and benchmark_release > model_release:
        timing = "The benchmark was released after the model, so a high post-cutoff score is not explained by that benchmark sitting in pretraining."
    else:
        timing = "The release dates do not put the benchmark after the model. A high score is harder to read as a clean control."
    if gap > 0.1:
        gap_text = "The pre-cutoff score is higher by more than 0.10. That is consistent with contamination, and also with the post-cutoff set being harder. It is not a verdict."
    elif gap < -0.1:
        gap_text = "The post-cutoff score is higher. That argues against 'the model only memorised the older set'."
    else:
        gap_text = "The two scores are close. On a 30-question set that closeness is inside the noise. See docs/posttrain/evaluation.md."
    base["reading"] = timing + " " + gap_text
    return base


@torch.no_grad()
def greedy_continue(model, tokenizer, prefix: str, n_chars: int) -> str:
    """Greedy decode, capped so a CPU demo stays small."""
    ids = tokenizer.encode(prefix, add_special_tokens=False) or [tokenizer.eos_token_id]
    # A character budget, not a claim about the text. Byte tokens are ~1 char.
    n_new = max(1, min(8, n_chars))
    room = model.config.n_positions - 1
    ids = ids[-max(1, room - n_new):]
    for _ in range(n_new):
        logits = model(torch.tensor([ids], dtype=torch.long)).logits[0, -1]
        nxt = int(torch.argmax(logits))
        ids.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
    new_ids = ids[len(tokenizer.encode(prefix, add_special_tokens=False)[-max(1, room - n_new):]):]
    # Decode only the generated tail. Re-encode length can differ; use the count we added.
    # Simpler: decode full ids and strip the prefix string if it is a prefix of the decode.
    full = tokenizer.decode(ids, skip_special_tokens=True)
    if full.startswith(prefix):
        return full[len(prefix):]
    return tokenizer.decode(ids[-n_new:], skip_special_tokens=True)


def cpu_demo() -> dict:
    """Tiny model, toy sentence. The numbers are whatever this run produced."""
    from posttrain.tiny import build_tiny_lm

    model, tok = build_tiny_lm(seed=0)
    model.eval()
    text = "Heat the sample to 37 C and then spin."
    control = shuffled_control(text, seed=1)
    rows = score_prefixes(text, lambda prefix, n: greedy_continue(model, tok, prefix, n), fractions=(0.5,))
    control_rows = score_prefixes(control, lambda prefix, n: greedy_continue(model, tok, prefix, n), fractions=(0.5,))
    original_lp = sequence_token_logprobs(model, tok, text)
    paraphrase = "Warm the tube to thirty seven degrees, then centrifuge."
    para_lp = sequence_token_logprobs(model, tok, paraphrase)
    return {
        "status": "ran on tiny model",
        "note": "This is a CPU demo of the metrics. It is not a contamination finding about a lab model.",
        "original": rows,
        "shuffled_control": control_rows,
        "mean_logprob_original": mean_logprob(original_lp),
        "mean_logprob_paraphrase": mean_logprob(para_lp),
        "min_k_original": min_k_logprob(original_lp),
        "min_k_paraphrase": min_k_logprob(para_lp),
        "membership_gap_mean_logprob": membership_gap(mean_logprob(original_lp), mean_logprob(para_lp)),
        "canary": canary_in_texts([text, paraphrase]),
        "weakness": (
            "A min-k or perplexity gap can come from wording. The paraphrase may be "
            "rarer English. The shuffled control is there so a high continuation "
            "score has something to be compared with."
        ),
    }
