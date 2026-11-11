"""PQA env: fill-in-the-blank multiple choice about a protocol step (parameter / reagent / operation).

Signal: verifiable correctness + a proper-scoring-rule calibration term (the model must also report a
0-100 confidence, as in BioProBench's PQA prompt). Calibration matters for wet-lab use: a confidently
wrong quantity is worse than a hedged one.
"""
from __future__ import annotations

import random
import re

from bioenvs.base import BioEnv, Reward, Sample, extract_answer, register

PROMPT = """You will be given a multiple-choice question related to a biological protocol. The blank in the question (represented as '____') indicates where the correct choice should be filled in.

Question:
{question}

Choices:
{choices}

Your task:
- Choose the most likely correct answer from the given choices.
- You must always select *one* answer, even if you are unsure.
- The selected answer must match one of the choices exactly (including case and punctuation).
- Assign a confidence score between 0 and 100 based on your certainty.
- Output your answer *wrapped exactly* between the tags [ANSWER_START] and [ANSWER_END].
- The format of your response must be:
[ANSWER_START]your selected choice & your confidence score[ANSWER_END]
"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip("'\"").lower())


_SPLIT = re.compile(r"^(.*?)[\s,;&|:\-]*\(?(\d+(?:\.\d+)?)\s*%?\)?\s*$", re.S)
_ABSTAIN_HEADS = frozenset({"abstain", "[abstain]"})


def parse(ans: str, choices: list[str]) -> tuple[str | None, float | None]:
    """'0.3 & 80' / '0.3, 80' / 'Pour 85' -> (choice, confidence in [0,1]).

    The whole string is tried as a choice first, so choices that end in a number ('0.3') are not mangled.
    The delimiter between choice and confidence is tolerated (models often write ',' or a space instead of '&');
    the [ANSWER_START]/[ANSWER_END] tags and the choice text itself are still required to be exact.
    """
    norm = {_norm(c): c for c in choices}
    if _norm(ans) in norm:
        return norm[_norm(ans)], None
    m = _SPLIT.match(ans)
    if not m:
        return None, None
    return norm.get(_norm(m.group(1))), min(float(m.group(2)), 100.0) / 100.0


def _is_abstain(ans: str) -> bool:
    head = ans.split("&")[0].split(",")[0].strip()
    return _norm(ans) in _ABSTAIN_HEADS or _norm(head) in _ABSTAIN_HEADS


@register
class PQA(BioEnv):
    name, task = "pqa", "PQA"

    def __init__(self, *a, calibration_weight: float = 0.2, abstain_reward: float = 0.10, **kw):
        super().__init__(*a, **kw)
        self.cw = calibration_weight
        self.abstain_reward = abstain_reward

    def make_sample(self, row):
        return Sample(row["id"], PROMPT.format(question=row["question"], choices=row["choices"]),
                      {"answer": row["answer"], "choices": row["choices"], "type": row["type"], "id": row["id"]})

    def corrupt(self, info, other):
        wrong = [c for c in info["choices"] if _norm(c) != _norm(info["answer"])]
        return {**info, "answer": random.Random(info["id"]).choice(wrong)}

    def make_counterfactual(self, sample, other_row):
        """Option-order counterfactual: shuffle displayed choices while keeping the true answer text fixed."""
        inf = sample.info
        choices = list(inf["choices"])
        if len(choices) >= 2:
            rng = random.Random(f"cf:{sample.id}")
            shuffled = choices[::-1] if len(choices) == 2 else choices[:]
            while shuffled == choices:
                rng.shuffle(shuffled)
        else:
            shuffled = choices
        row = self.rows[self._index[sample.id]] if sample.id in self._index else {"question": sample.prompt}
        q = row.get("question", "")
        prompt = PROMPT.format(question=q, choices=shuffled) if q else sample.prompt
        return Sample(f"{sample.id}:cf", prompt, {**inf, "choices": shuffled})

    def score(self, sample, completion):
        inf = sample.info
        ans = extract_answer(completion)
        if ans is None:
            return Reward(0.0, False, {"correct": 0.0, "calibration": 0.0})
        if self.reward_version == "v2" and _is_abstain(ans):
            # Honest abstention beats any wrong guess (<= 0.05) while staying far below any right answer (>= 0.80).
            return Reward(self.abstain_reward, True, {"correct": 0.0, "calibration": 0.5, "abstained": 1.0, "confidence": 0.0})
        pick, conf = parse(ans, inf["choices"])
        if pick is None:  # tags present but the choice is not one of the options
            return Reward(0.0, False, {"correct": 0.0, "calibration": 0.0})
        correct = float(_norm(pick) == _norm(inf["answer"]))
        # Brier-style: 1 when confidence == outcome. Missing confidence is treated as maximal hedging (0.5).
        c = 0.5 if conf is None else conf
        calib = 1.0 - (c - correct) ** 2
        # Right answers get (1-cw) + cw*calib. Wrong answers only get a quarter of the calibration term, so a
        # well-hedged miss (~0.05) is still strictly worse than any correct answer (>= 0.8) but is not
        # indistinguishable from a confidently wrong one (0.0).
        value = (1 - self.cw) * correct + self.cw * calib * (1.0 if correct else 0.25)
        calib_comp = calib * (1.0 if correct else 0.25) if self.reward_version == "v2" else calib
        return Reward(value, True, {"correct": correct, "calibration": calib_comp, "confidence": c})
