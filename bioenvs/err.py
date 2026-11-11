"""ERR env: audit a protocol step; if it is wrong, say so AND repair it.

BioProBench only asks for a True/False verdict. Here the policy must also emit a corrected step, which
turns a 1-bit signal into a verifiable *edit* signal:
  * correct step  -> reward 1 for "True", 0 for flagging it (punishes paranoid over-flagging)
  * corrupted step-> 0.4 * detect  +  0.6 * repair, where repair is the normalised improvement in token-level
                     similarity to the gold step vs. the corrupted step (copying the corrupted step = 0,
                     recovering the gold step = 1). Repair only counts if the verdict was "False".
The corpus is ~50/50 correct/corrupted, so constant policies ("always True" / "always False") average ~0.5 / ~0.2 (see tests).
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from bioenvs.base import BioEnv, Reward, Sample, extract_answer, register

PROMPT = """Determine whether the following target step in a protocol is True (correct) or False (contains an error in an operation, a reagent or a parameter such as a number or unit):
{step}

You may use the following context, which includes the purpose of the step, as well as the preceding and following steps, to inform your decision:
{context}

Carefully evaluate if the step is logically consistent, necessary, and accurate in the context.

- Output your answer *wrapped exactly* between the tags [ANSWER_START] and [ANSWER_END].
- If the step is correct, the answer text is True.
- If the step contains an error, the answer text is False, then ||, then the minimally-corrected step.
- Do not leave the example words in the answer. A copied instruction is not a verdict.
"""

_tok = re.compile(r"\w+(?:[.,]\w+)*|[^\w\s]")


def tokens(s: str) -> list[str]:
    return _tok.findall(s.lower())


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, tokens(a), tokens(b), autojunk=False).ratio()


def parse(ans: str) -> tuple[bool | None, str]:
    """Verdict must be exactly true or false.

    startswith("false") treated "Falsehood || <gold step>" as a real False
    verdict and gave it a full repair score. The prompt asks for the token
    True, or False || <step>, so the head has to be that word and nothing else.
    A trailing period is still accepted ("True.").
    """
    head, sep, tail = ans.partition("||")
    h = head.strip().lower().strip(".!")
    if h == "true":
        return True, ""
    if h == "false":
        return False, tail.strip() if sep == "||" else ""
    return None, ""


@register
class ERR(BioEnv):
    name, task = "err", "ERR"

    def keep(self, row):
        if not row.get("corrected_text"):
            return False
        if row["is_correct"]:
            return True
        # ~5% of corrupted rows in BioProBench are no-ops (corrupted == corrected): effectively a correct step
        # labelled as an error. They are unlearnable and would punish correct "True" verdicts, so drop them.
        return bool(row.get("corrupted_text")) and tokens(row["corrupted_text"]) != tokens(row["corrected_text"])

    def make_sample(self, row):
        step = row["corrected_text"] if row["is_correct"] else row["corrupted_text"]
        ctx = row["context"] if isinstance(row["context"], str) else json.dumps(row["context"], ensure_ascii=False, indent=1)
        return Sample(row["id"], PROMPT.format(step=step, context=ctx),
                      {"is_correct": bool(row["is_correct"]), "shown": step, "gold": row["corrected_text"],
                       "type": row["type"], "id": row["id"]})

    def corrupt(self, info, other):
        return {**info, "is_correct": not info["is_correct"]}   # flip the verdict the policy is rewarded for

    def make_counterfactual(self, sample, other_row):
        """Verdict-flip counterfactual: swap corrupted step with its repaired gold step when available."""
        row = self.rows[self._index[sample.id]] if sample.id in self._index else None
        if row is None:
            return super().make_counterfactual(sample, other_row)
        ctx = row["context"] if isinstance(row["context"], str) else json.dumps(row["context"], ensure_ascii=False, indent=1)
        if not sample.info["is_correct"]:
            # Flip corrupted -> gold so the true verdict becomes True
            step = row["corrected_text"]
            return Sample(
                f"{sample.id}:cf",
                PROMPT.format(step=step, context=ctx),
                {**sample.info, "is_correct": True, "shown": step},
            )
        if row.get("corrupted_text") and tokens(row["corrupted_text"]) != tokens(row["corrected_text"]):
            step = row["corrupted_text"]
            return Sample(
                f"{sample.id}:cf",
                PROMPT.format(step=step, context=ctx),
                {**sample.info, "is_correct": False, "shown": step},
            )
        return super().make_counterfactual(sample, other_row)

    def score(self, sample, completion):
        inf = sample.info
        ans = extract_answer(completion)
        zero = Reward(0.0, False, {"detect": 0.0, "repair": 0.0})
        if ans is None:
            return zero
        verdict, fix = parse(ans)
        if verdict is None:
            return zero
        if inf["is_correct"]:
            ok = float(verdict is True)
            return Reward(ok, True, {"detect": ok, "repair": -1.0})  # -1 = n/a (ignored by summaries)
        detect = float(verdict is False)
        repair = 0.0
        if detect and fix:
            base = sim(inf["shown"], inf["gold"])
            got = sim(fix, inf["gold"])
            repair = max(0.0, (got - base) / max(1e-6, 1.0 - base))
        return Reward(0.4 * detect + 0.6 * repair, True, {"detect": detect, "repair": repair})
