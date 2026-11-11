"""ORD env: put shuffled protocol steps (or section titles) back in order.

Signal: dense, order-aware. 0.5 * exact-order + 0.5 * max(0, Kendall tau). A random permutation scores
~0.2 on average, a reversed one 0, so the gradient is informative even when exact match is rare.
Steps are shown with explicit [i] indices (BioProBench prints a raw Python list, which adds a counting
confound: long steps made models miscount). Malformed answers (not a permutation of 0..n-1) get 0 - there is no partial credit for formatting.
"""
from __future__ import annotations

import ast
import random
from itertools import combinations

from bioenvs.base import BioEnv, Reward, Sample, extract_answer, register

PROMPT = """{question}
The steps are (each prefixed with its original index):
{steps}

- Give me the correct order of the steps as a list of their original indices (start from 0), no other words.
- Output your answer *wrapped exactly* between the tags [ANSWER_START] and [ANSWER_END].
- The format of your response must be:
[ANSWER_START]a list of the original indices[ANSWER_END]
"""


def gold_permutation(wrong: list[str], correct: list[str]) -> list[int] | None:
    """Indices into `wrong` that spell out `correct` (duplicate-safe). None if the two are not permutations."""
    used, out = set(), []
    for step in correct:
        for i, w in enumerate(wrong):
            if i not in used and w == step:
                used.add(i)
                out.append(i)
                break
        else:
            return None
    return out if len(out) == len(wrong) else None


def kendall_tau(gold: list[int], pred: list[int]) -> float:
    rg = {x: i for i, x in enumerate(gold)}
    rp = {x: i for i, x in enumerate(pred)}
    pairs = list(combinations(gold, 2))
    if not pairs:
        return 1.0
    conc = sum((rg[a] - rg[b]) * (rp[a] - rp[b]) > 0 for a, b in pairs)
    return (2 * conc - len(pairs)) / len(pairs)


@register
class ORD(BioEnv):
    name, task = "ord", "ORD"

    def keep(self, row):
        w, c = row["wrong_steps"], row["correct_steps"]
        return 2 <= len(w) <= 12 and gold_permutation(w, c) is not None

    def make_sample(self, row):
        gold = gold_permutation(row["wrong_steps"], row["correct_steps"])
        return Sample(row["id"], PROMPT.format(question=row["question"], steps="\n".join(f"[{i}] {t}" for i, t in enumerate(row["wrong_steps"]))),
                      {"gold": gold, "n": len(gold), "type": row["type"], "id": row["id"]})

    def corrupt(self, info, other):
        gold, rng = info["gold"], random.Random(info["id"])
        wrong = gold[::-1] if len(gold) == 2 else gold[:]
        while wrong == gold:
            rng.shuffle(wrong)
        return {**info, "gold": wrong}

    def make_counterfactual(self, sample, other_row):
        """Step-permutation counterfactual: re-shuffle displayed steps and recompute exact gold indices."""
        row = self.rows[self._index[sample.id]] if sample.id in self._index else None
        if row is None:
            return super().make_counterfactual(sample, other_row)
        orig_steps = list(row["wrong_steps"])
        correct_steps = list(row["correct_steps"])
        rng = random.Random(f"cf:{sample.id}")
        cf_steps = orig_steps[::-1] if len(orig_steps) == 2 else orig_steps[:]
        while cf_steps == orig_steps:
            rng.shuffle(cf_steps)
        cf_gold = gold_permutation(cf_steps, correct_steps)
        prompt = PROMPT.format(
            question=row["question"],
            steps="\n".join(f"[{i}] {t}" for i, t in enumerate(cf_steps)),
        )
        return Sample(f"{sample.id}:cf", prompt, {**sample.info, "gold": cf_gold})

    def score(self, sample, completion):
        gold, n = sample.info["gold"], sample.info["n"]
        ans = extract_answer(completion)
        zero = Reward(0.0, False, {"exact": 0.0, "tau": 0.0})
        # ast.literal_eval ignores Python comments, so "[0, 1] # notes" used to
        # score as a clean permutation. The prompt says no other words.
        if ans is None or "#" in ans:
            return zero
        try:
            pred = ast.literal_eval(ans)
        except Exception:  # noqa: BLE001
            return zero
        if not isinstance(pred, (list, tuple)) or sorted(pred) != list(range(n)) or not all(isinstance(x, int) for x in pred):
            return zero
        pred = list(pred)
        exact = float(pred == gold)
        tau = kendall_tau(gold, pred)
        return Reward(0.5 * exact + 0.5 * max(0.0, tau), True, {"exact": exact, "tau": tau})
