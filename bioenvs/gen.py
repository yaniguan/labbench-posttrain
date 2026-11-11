"""GEN env: write a protocol from a title/goal.

BioProBench scores GEN with BLEU / embedding similarity; neither is cheap or dense enough for RL, and
neither checks quantities. This reward is model-free and runs in ~ms:
  step F1   : soft one-to-one alignment (Hungarian) of generated vs. reference steps on content-token
              F1; a pair counts as matched at >= 0.3. Precision penalises padding / hallucinated steps,
              recall penalises skipped ones.
  order     : Kendall tau (rescaled to [0,1]) over the matched pairs - rewards correct procedure order.
  numeric   : fraction of the reference's quantities (number+unit, e.g. '37 °C', '10 min') that appear in the
              aligned generated step - the "quantitative precision" BioProBench shows models lack.
reward = 0.6*F1 + 0.2*order + 0.2*numeric   (0.8*F1 + 0.2*order if the reference has no quantities)
Non-list / empty outputs get 0; more than 3x the reference length is clipped (anti-padding).

Known limit: repeating one reference step on every line still scores about 0.40
on average (100 test items, max 0.60). Soft token F1 gives that line credit
against several reference steps that share words. We do not drop duplicate
lines, because at least one real reference repeats a step verbatim and the
oracle check requires a mean of 1. See the xfail test in tests/test_posttrain_hacking.py.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from bioenvs.base import BioEnv, Reward, Sample, register, strip_thinking
from bioenvs.ordering import kendall_tau

MATCH_THRESHOLD = 0.3
DUP_THRESHOLD = 0.92
STOP = set("the a an of to and or in on at for with by from then is are be as it this that into using use each all".split())
_word = re.compile(r"[a-z0-9µμ°%.]+")
_qty = re.compile(r"(\d+(?:\.\d+)?)\s*(°\s*c|µ[a-z]+|μ[a-z]+|[a-z]{1,4}|%|x|×\s*g|rpm)?", re.I)
_step_prefix = re.compile(r"^\s*(?:step\s*)?\d+[.):]\s*", re.I)


def split_steps(text: str) -> list[str]:
    lines = [_step_prefix.sub("", l).strip() for l in strip_thinking(text).splitlines()]
    return [l for l in lines if len(l) > 2]


def content_tokens(s: str) -> set[str]:
    return {w.strip(".") for w in _word.findall(s.lower()) if w.strip(".") and w.strip(".") not in STOP}


def tok_f1(a: str, b: str) -> float:
    A, B = content_tokens(a), content_tokens(b)
    if not A or not B:
        return 0.0
    i = len(A & B)
    if i == 0:
        return 0.0
    p, r = i / len(A), i / len(B)
    return 2 * p * r / (p + r)


def quantities(s: str) -> set[str]:
    out = set()
    for num, unit in _qty.findall(s.lower()):
        unit = re.sub(r"\s+", "", unit or "")
        out.add(f"{num}{unit}")
    return out


def align(ref: list[str], gen: list[str]):
    S = np.array([[tok_f1(r, g) for g in gen] for r in ref])
    ri, gi = linear_sum_assignment(-S)
    return [(r, g, S[r, g]) for r, g in zip(ri, gi) if S[r, g] >= MATCH_THRESHOLD], S


def _cluster_steps(steps: list[str], threshold: float = DUP_THRESHOLD) -> list[int]:
    """Connected-component cluster IDs for near-duplicate steps (tok_f1 >= threshold)."""
    n = len(steps)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if steps[i].strip().lower() == steps[j].strip().lower() or tok_f1(steps[i], steps[j]) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return [find(i) for i in range(n)]


def align_dedup(ref: list[str], gen: list[str]):
    """Anti-hack alignment: a near-duplicate cluster of generated steps may align to at most
    one near-duplicate cluster of reference steps, preserving verbatim repeats in `ref` while
    blocking the `repeat ref[0] N times` exploit.
    """
    raw_pairs, S = align(ref, gen)
    if len(raw_pairs) <= 1:
        return raw_pairs, S
    ref_clusters = _cluster_steps(ref)
    gen_clusters = _cluster_steps(gen)
    claimed_ref_cluster: dict[int, int] = {}
    kept = []
    for r, g, sc in sorted(raw_pairs, key=lambda x: (-x[2], x[0], x[1])):
        gc = gen_clusters[g]
        rc = ref_clusters[r]
        if gc not in claimed_ref_cluster:
            claimed_ref_cluster[gc] = rc
            kept.append((r, g, sc))
        elif claimed_ref_cluster[gc] == rc:
            kept.append((r, g, sc))
    kept.sort(key=lambda x: x[0])
    return kept, S


@register
class GEN(BioEnv):
    name, task = "gen", "GEN"

    def keep(self, row):
        return 3 <= len(row["output"]) <= 25

    def make_sample(self, row):
        prompt = (f"{row['system_prompt']}\n{row['instruction']}\nFormat requirements:\n"
                  f"- Each step must be on a separate line.\n\n{row['input']}")
        return Sample(row["id"], prompt, {"ref": [_step_prefix.sub("", s) for s in row["output"]],
                                          "type": row["type"], "id": row["id"]})

    def corrupt(self, info, other):
        return {**info, "ref": other["ref"]}   # the reference protocol of a different task

    def score(self, sample, completion):
        ref = sample.info["ref"]
        gen = split_steps(completion)
        if not gen:
            return Reward(0.0, False, {"f1": 0.0, "order": 0.0, "numeric": 0.0})
        gen = gen[: 3 * len(ref)]
        pairs, _ = align_dedup(ref, gen) if self.reward_version == "v2" else align(ref, gen)
        if not pairs:
            return Reward(0.0, True, {"f1": 0.0, "order": 0.0, "numeric": 0.0})
        # One line that pastes every reference step still overlaps the reference
        # on tokens, so soft F1 used to pay about 0.38 for a format the prompt
        # forbids. A single step longer than 1.5x the longest reference step is
        # that dump. A normal one-step answer is about as long as one step.
        if len(gen) == 1 and len(ref) >= 3 and len(gen[0]) > 1.5 * max(len(s) for s in ref):
            return Reward(0.0, True, {"f1": 0.0, "order": 0.0, "numeric": 0.0})
        # soft precision / recall: matched pairs weighted by their similarity
        tp = sum(s for _, _, s in pairs)
        prec, rec = tp / len(gen), tp / len(ref)
        f1 = 2 * prec * rec / (prec + rec)
        if len(pairs) > 1:
            order = (
                kendall_tau(
                    [r for r, _, _ in sorted(pairs)],
                    [r for r, _, _ in sorted(pairs, key=lambda x: x[1])],
                )
                + 1
            ) / 2
        elif self.reward_version == "v1" or len(ref) <= 1:
            order = 1.0
        else:
            order = 0.0
        if self.reward_version == "v2":
            ref_q = [(r_idx, quantities(s)) for r_idx, s in enumerate(ref)]
        else:
            ref_q = [(r, quantities(ref[r])) for r, _, _ in pairs]
        n_q = sum(len(q) for _, q in ref_q)
        gen_of = {r: g for r, g, _ in pairs}
        numeric = (
            sum(len(q & quantities(gen[gen_of[r]])) for r, q in ref_q if r in gen_of) / n_q
        ) if n_q else None
        if numeric is None:
            value = 0.8 * f1 + 0.2 * order
        else:
            value = 0.6 * f1 + 0.2 * order + 0.2 * numeric
        return Reward(value, True, {"f1": f1, "order": order, "numeric": numeric if numeric is not None else -1.0})
