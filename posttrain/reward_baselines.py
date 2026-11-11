"""The spurious-reward ladder from Shao et al., Section 2 and Appendix G.

One flag picks the training signal. The prompts and the trainer stay the
same. If a random or format reward later matches most of the ground-truth
gain, the gain came from the pretraining prior or from formatting, not from
the task. That comparison has not been run. See
docs/posttrain/reward_baselines.md.

KL on and KL off are a second axis. Clipping and the KL anchor both limit
how far a random reward can move the policy (Shao et al., Section 4, the
no-clip panels of Figure 4). The grid is in configs/posttrain/kl_grid/.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter

from posttrain.rewards import RewardOutput

# Paper Section 2, plus the controls we need for a lab run.
LADDER = (
    "ground_truth",
    "majority_vote",
    "format_only",
    "incorrect_label",
    "random_p50",
    "constant_1",
    "constant_0",
    "length_reward",
    "no_repetition",
    "prior_only",
)

# Shao et al. Section 2: reward 1 when the response contains a non-empty \boxed{}.
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_RE = re.compile(r"\[ANSWER_START\](.*?)\[ANSWER_END\]", re.S)


def extract_answer_text(completion: str) -> str | None:
    """Last answer span. Boxed (the paper) or the lab's answer tags."""
    if not completion:
        return None
    boxed = BOXED_RE.findall(completion)
    if boxed and boxed[-1].strip():
        return boxed[-1].strip()
    tags = ANSWER_RE.findall(completion)
    if tags and tags[-1].strip():
        return tags[-1].strip()
    return None


def format_only_reward(completion: str) -> float:
    """1 if there is a non-empty \\boxed{}. 0 otherwise.

    This is the paper's format reward, not the lab's [ANSWER_START] tag.
    A lab completion that is well formed for BioProBench still scores 0 here.
    That is intentional: the control asks whether the boxed habit, which
    Qwen-Math was trained to emit, is doing the work.
    """
    found = BOXED_RE.findall(completion or "")
    return 1.0 if any(part.strip() for part in found) else 0.0


def no_repetition_reward(completion: str, limit: int = 10) -> float:
    """Appendix G: 0 if any string is repeated more than 10 times, else 1.

    The paper's wording is "any string repeated more than 10 times". We
    implement two checks that match that sentence:
      * a whitespace-separated word that occurs more than `limit` times
      * an 8-character span that occurs more than `limit` times
    A normal short answer scores 1. "ok" eleven times scores 0.
    """
    text = completion or ""
    words = text.split()
    if any(c > limit for c in Counter(words).values()):
        return 0.0
    if len(text) >= 8:
        spans = Counter(text[i:i + 8] for i in range(len(text) - 7))
        if any(c > limit for c in spans.values()):
            return 0.0
    return 1.0


def length_reward(completion: str, target_words: int = 32) -> float:
    """A reward that looks only at length. It ignores the task.

    Score is 1 when the completion has `target_words` words, and it falls
    off as the length moves away. This is a control for length bias, not a
    length penalty on a correct answer. The binary group length penalty
    lives in advantages.group_length_penalty.
    """
    n = len((completion or "").split())
    return 1.0 / (1.0 + abs(n - target_words) / max(target_words, 1))


def _stable_rng(*parts, base_seed: int = 0) -> random.Random:
    raw = repr(parts).encode()
    digest = hashlib.sha256(raw).hexdigest()
    seed = (base_seed ^ int(digest[:8], 16)) & 0xFFFFFFFF
    return random.Random(seed)


def majority_label(answers: list[str | None]) -> str | None:
    """Most common non-empty answer. Ties keep the first one we saw.

    The paper uses 64 rollouts (Section 2). We use whatever group the
    caller passed and we record the count. We do not pretend it was 64.
    """
    counts: Counter[str] = Counter()
    first_index: dict[str, int] = {}
    for i, ans in enumerate(answers):
        if ans is None or ans == "":
            continue
        counts[ans] += 1
        first_index.setdefault(ans, i)
    if not counts:
        return None
    return sorted(counts, key=lambda a: (-counts[a], first_index[a]))[0]


def score_baseline(
    mode: str,
    *,
    prompt: str,
    completion: str,
    meta: dict | None = None,
    group_completions: list[str] | None = None,
    ground_truth: float | None = None,
    seed: int = 0,
    index: int = 0,
) -> RewardOutput:
    """One completion, one mode. `ground_truth` is the env's checked reward.

    prior_only does not change the number: the trainer skips the update.
    The logged reward is still the ground-truth score, so the report can
    show what the untouched policy already gets.
    """
    if mode not in LADDER:
        raise ValueError(f"unknown reward baseline {mode!r}; expected one of {LADDER}")
    meta = meta or {}
    diag = {"baseline": mode}

    if mode in ("ground_truth", "prior_only"):
        if ground_truth is None:
            raise ValueError(f"{mode} needs the env reward")
        return RewardOutput(float(ground_truth), {f"reward/{mode}": float(ground_truth)}, diag)

    if mode == "format_only":
        v = format_only_reward(completion)
        return RewardOutput(v, {"reward/format_only": v}, diag)

    if mode == "random_p50":
        rng = _stable_rng(seed, index, prompt, completion, base_seed=seed)
        v = 1.0 if rng.random() < 0.5 else 0.0
        return RewardOutput(v, {"reward/random_p50": v}, diag)

    if mode == "constant_1":
        return RewardOutput(1.0, {"reward/constant_1": 1.0}, diag)
    if mode == "constant_0":
        return RewardOutput(0.0, {"reward/constant_0": 0.0}, diag)

    if mode == "length_reward":
        v = length_reward(completion)
        return RewardOutput(v, {"reward/length": v}, diag)

    if mode == "no_repetition":
        v = no_repetition_reward(completion)
        return RewardOutput(v, {"reward/no_repetition": v}, diag)

    if mode == "majority_vote":
        group = group_completions if group_completions is not None else [completion]
        answers = [extract_answer_text(c) for c in group]
        label = majority_label(answers)
        mine = extract_answer_text(completion)
        v = 1.0 if label is not None and mine == label else 0.0
        diag.update({"majority": label, "n_group": len(group), "n_parsed": sum(a is not None for a in answers)})
        return RewardOutput(v, {"reward/majority_vote": v}, diag)

    if mode == "incorrect_label":
        # Section 2: reward one specific wrong answer, not "any wrong answer".
        # The caller puts that string in meta["incorrect_label"].
        label = meta.get("incorrect_label")
        mine = extract_answer_text(completion)
        if label is None:
            v = 0.0
            diag["missing_incorrect_label"] = True
        else:
            v = 1.0 if mine is not None and mine.strip() == str(label).strip() else 0.0
        return RewardOutput(v, {"reward/incorrect_label": v}, diag)

    raise AssertionError(mode)
