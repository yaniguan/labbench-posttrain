"""Reward ladder from Shao et al., Spurious Rewards, §2.2, plus the extras asked for here.

One name swaps the reward. The prompts and the trainer stay the same.

    ground_truth     the env's checked score (rewards.ground_truth)
    majority_vote    1 if the extracted answer equals the most common answer in the rollouts
    format_only      1 if the env can parse an answer (the lab analogue of a non-empty \\boxed{})
    incorrect_label  1 if the extracted answer equals a wrong label
    random_p50       1 about half the time, seeded, independent of the answer
    constant_one     always 1
    constant_zero    always 0
    length_reward    longer text scores higher, up to a cap
    no_repetition    0 if an obvious string repeats, else 1 (Appendix G)
    prior_only       not a reward. The trainer skips the update. Eval still uses ground truth.

What a result would mean
    If random_p50 or format_only picks up most of the ground_truth gain, the gain is the
    pretraining prior or the format, not the task. Shao et al. saw that pattern on
    Qwen2.5-Math and not on Llama3.1-8B-Instruct or OLMo2-7B (§3, Figure 1). Run at
    least two families before trusting a curve. See docs/posttrain/reward_baselines.md.

Nothing here downloads a model. Majority vote needs meta["rollouts"]. The paper used 64
samples. A missing rollout list scores 0 and says so. It does not fall back to the true
label, which would quietly turn the control into ground truth.

BioProBench is CC BY-NC 4.0. These functions only read the existing checkers.
"""
from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter

from bioenvs.base import Sample, extract_answer
from bioenvs.err import parse as parse_err
from bioenvs.gen import split_steps
from bioenvs.pqa import parse as parse_pqa

from posttrain.rewards import get_env, ground_truth

NAMES = (
    "ground_truth",
    "majority_vote",
    "format_only",
    "incorrect_label",
    "random_p50",
    "constant_one",
    "constant_zero",
    "length_reward",
    "no_repetition",
    "prior_only",
)

# Appendix G: reward 0 when a string is repeated more than 10 times, else 1.
# The paper does not publish the matcher. We treat a line, or a token of length
# at least 4, as "an obvious string". Short common words are not enough.
REPEAT_TIMES = 10
REPEAT_TOKEN_MIN = 4


def extract_task_answer(env_name: str, completion: str, info: dict):
    """A comparable answer string, or None when the completion has no extractable answer.

    GEN's answer is the normalised step text. Exact match on that string is sparse.
    That is intentional: majority and incorrect labels stay binary, like the paper's
    0/1 rewards. The dense score stays in ground_truth.
    """
    if env_name == "pqa":
        text = extract_answer(completion)
        if text is None:
            return None
        pick, _conf = parse_pqa(text, info.get("choices") or [])
        return None if pick is None else pick
    if env_name == "ord":
        text = extract_answer(completion)
        if text is None:
            return None
        try:
            pred = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
        n = int(info.get("n") or 0)
        if not isinstance(pred, (list, tuple)) or not all(isinstance(x, int) and not isinstance(x, bool) for x in pred):
            return None
        if sorted(pred) != list(range(n)):
            return None
        return str(list(pred))
    if env_name == "err":
        text = extract_answer(completion)
        if text is None:
            return None
        verdict, _fix = parse_err(text)
        if verdict is True:
            return "True"
        if verdict is False:
            return "False"
        return None
    if env_name == "gen":
        steps = split_steps(completion)
        if not steps:
            return None
        return "\n".join(step.strip().lower() for step in steps)
    return None


def majority_answer(answers: list) -> str | None:
    """Most common non-empty answer. Ties go to whichever appeared first."""
    cleaned = [a for a in answers if a]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    best = max(counts.values())
    for answer in cleaned:
        if counts[answer] == best:
            return answer
    return None


def gold_answer(env_name: str, info: dict) -> str | None:
    if env_name == "pqa":
        return info.get("answer")
    if env_name == "ord":
        return str(list(info.get("gold") or []))
    if env_name == "err":
        return "True" if info.get("is_correct") else "False"
    if env_name == "gen":
        ref = info.get("ref") or []
        return "\n".join(step.strip().lower() for step in ref) if ref else None
    return None


def wrong_label(env_name: str, info: dict, rollouts: list[str] | None, prompt: str) -> tuple[str | None, str]:
    """A label that is not the gold answer.

    Prefer a majority rollout that is already wrong (the paper's construction, §2.2).
    If every rollout is right, or there are no rollouts, use the env's corrupt().
    """
    gold = gold_answer(env_name, info)
    if rollouts:
        extracted = [extract_task_answer(env_name, text, info) for text in rollouts]
        maj = majority_answer(extracted)
        if maj and maj != gold:
            return maj, "incorrect_majority_rollout"
    env = get_env(env_name, "test", 0)
    other = info
    if len(env):
        neighbour = env.sample(1).info
        if neighbour.get("id") == info.get("id") and len(env) > 1:
            neighbour = env.sample(2).info
        other = neighbour
    sample_info = env.corrupt(dict(info), other)
    label = gold_answer(env_name, sample_info)
    if label == gold or label is None:
        return None, "no_wrong_label"
    return label, "env_corrupt"


def has_obvious_repetition(text: str, times: int = REPEAT_TIMES) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and max(Counter(lines).values()) > times:
        return True
    tokens = [tok for tok in text.split() if len(tok) >= REPEAT_TOKEN_MIN]
    if tokens and max(Counter(tokens).values()) > times:
        return True
    return False


def _boxed_nonempty(text: str) -> bool:
    """Paper's format reward: a non-empty \\boxed{...}. Used when meta says format_style=boxed."""
    return re.search(r"\\boxed\{[^{}]+\}", text or "") is not None


def _parsed(env_name: str, prompt: str, completion: str, info: dict, split: str, seed: int) -> bool:
    env = get_env(env_name, split, seed)
    sample = Sample(id=str(info.get("id", "")), prompt=prompt, info=info)
    return bool(env.score(sample, completion).parsed)


def compute_baseline(name: str, prompt: str, completion: str, meta: dict) -> tuple[float, dict]:
    if name not in NAMES:
        raise KeyError(f"unknown reward {name}. known: {NAMES}")
    env_name = meta.get("env")
    info = meta.get("info") or {}
    diag = {"reward_name": name, "env": env_name}

    if name == "constant_one":
        diag["constant"] = 1.0
        return 1.0, diag
    if name == "constant_zero":
        diag["constant"] = 0.0
        return 0.0, diag
    if name == "length_reward":
        cap = float(meta.get("length_cap", 256))
        n = len((completion or "").split())
        value = min(1.0, n / cap) if cap else 0.0
        diag["tokens"] = n
        return value, diag
    if name == "no_repetition":
        bad = has_obvious_repetition(completion or "")
        diag["repeated"] = bad
        return (0.0 if bad else 1.0), diag
    if name == "random_p50":
        seed = int(meta.get("seed", 0))
        digest = hashlib.sha256(f"{seed}|{prompt}|{completion}".encode()).hexdigest()
        # First 8 hex digits. Even -> 1. This is about half, and stable if you score twice.
        bit = int(digest[:8], 16) % 2
        diag["p"] = 0.5
        diag["seed"] = seed
        return float(bit), diag

    if not env_name:
        raise KeyError(f"{name} needs meta['env']")

    if name == "format_only":
        if meta.get("format_style") == "boxed":
            ok = _boxed_nonempty(completion)
        else:
            ok = _parsed(env_name, prompt, completion, info, meta.get("split", "train"), meta.get("seed", 0))
        diag["parsed"] = ok
        return float(ok), diag

    extracted = extract_task_answer(env_name, completion, info)
    diag["extracted"] = extracted

    if name == "majority_vote":
        rollouts = list(meta.get("rollouts") or [])
        answers = [extract_task_answer(env_name, text, info) for text in rollouts]
        label = majority_answer(answers)
        diag["label"] = label
        diag["n_rollouts"] = len(rollouts)
        if label is None:
            diag["status"] = "no_rollouts"
            return 0.0, diag
        return float(extracted is not None and extracted == label), diag

    if name == "incorrect_label":
        label = meta.get("wrong_label")
        source = "given"
        if not label:
            label, source = wrong_label(env_name, info, meta.get("rollouts"), prompt)
        diag["label"] = label
        diag["label_source"] = source
        if not label:
            diag["status"] = "no_wrong_label"
            return 0.0, diag
        return float(extracted is not None and extracted == label), diag

    if name in ("ground_truth", "prior_only"):
        return ground_truth(prompt, completion, meta)

    raise KeyError(name)
