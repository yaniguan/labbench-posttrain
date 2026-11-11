"""Build SFT and DPO rows from a BioProBench env's own gold fields.

The chosen text is the reference the checker scores as correct. The rejected text is
a wrong label from env.corrupt, or an empty answer. These are checker preferences,
not human preferences. Say that when you report a DPO run.

Holdout ids are dropped. The monitor's gold split must not appear here.

LAB-Bench is not a source. The canary guard runs on the strings before they return.
BioProBench is CC BY-NC 4.0.
"""
from __future__ import annotations

from bioenvs import make_env

from posttrain.contamination import assert_training_corpus_clean
from posttrain.hacking_checks import wrap_answer
from posttrain.rewards import get_env


def gold_completion(env_name: str, sample) -> str:
    info = sample.info
    if env_name == "pqa":
        return wrap_answer(f"{info['answer']} & 80")
    if env_name == "ord":
        return wrap_answer(str(list(info["gold"])))
    if env_name == "err":
        if info["is_correct"]:
            return wrap_answer("True")
        return wrap_answer(f"False || {info['gold']}")
    if env_name == "gen":
        return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(info["ref"]))
    raise KeyError(env_name)


def rejected_completion(env_name: str, sample) -> str:
    """A completion the true checker should score below the gold completion."""
    info = sample.info
    env = get_env(env_name, "test", 0)
    other = env.sample(1).info
    if other.get("id") == info.get("id") and len(env) > 1:
        other = env.sample(2).info
    wrong = env.corrupt(dict(info), other)
    if env_name == "pqa":
        return wrap_answer(f"{wrong['answer']} & 100")
    if env_name == "ord":
        return wrap_answer(str(list(wrong["gold"])))
    if env_name == "err":
        # Flip the verdict and do not repair.
        if info["is_correct"]:
            return wrap_answer("False || not the step")
        return wrap_answer("True")
    if env_name == "gen":
        return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(wrong["ref"]))
    raise KeyError(env_name)


def filter_by_type(samples, allowed: list[str] | None):
    """Keep samples whose info['type'] is in the list.

    GEN uses type as easy / standard / difficulty. The other envs use type for the
    kind of error or the kind of blank, which is not an order. A curriculum that
    moves from easy to hard on a schedule is not implemented. This is only a filter.
    """
    if not allowed:
        return list(samples)
    allowed_set = set(allowed)
    return [s for s in samples if s.info.get("type") in allowed_set]


def iter_samples(env_name: str, n: int, split: str, seed: int, holdout_ids: set | None):
    env = make_env(env_name, split=split, seed=seed)
    kept = []
    for i in range(len(env)):
        sample = env.sample(i)
        if holdout_ids and sample.id in holdout_ids:
            continue
        kept.append(sample)
        if len(kept) >= n:
            break
    return kept


def gold_sft_pairs(env_name: str, n: int = 32, split: str = "train", seed: int = 0, holdout_ids: set | None = None):
    samples = iter_samples(env_name, n, split, seed, holdout_ids)
    pairs = [(s.prompt, gold_completion(env_name, s)) for s in samples]
    assert_training_corpus_clean([part for pair in pairs for part in pair], where=f"{env_name} sft pairs")
    return pairs


def gold_dpo_pairs(env_name: str, n: int = 32, split: str = "train", seed: int = 0, holdout_ids: set | None = None):
    samples = iter_samples(env_name, n, split, seed, holdout_ids)
    rows = []
    for sample in samples:
        rows.append({
            "prompt": sample.prompt,
            "chosen": gold_completion(env_name, sample),
            "rejected": rejected_completion(env_name, sample),
            "id": sample.id,
            "info": sample.info,
        })
    texts = [row[key] for row in rows for key in ("prompt", "chosen", "rejected")]
    assert_training_corpus_clean(texts, where=f"{env_name} dpo pairs")
    return rows
