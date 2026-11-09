"""Counterfactual probe derived from ProtocolQA (no new labels).

ProtocolQA asks "which of these changes fixes the problem?". Here each candidate is judged on its own:
would applying *this single change* fix the problem? The benchmark's `ideal` is the YES, each distractor is a NO.
That turns choose-one (which allows elimination) into verifying a hypothetical edit against the protocol.

    python -m labeval.counterfactual --models haiku-5.5 gpt-6-luna glm-5.3-flash
-> results/protocol_reasoning/counterfactual_<model>.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from tqdm import tqdm

from common.llm import LLM
from labeval.data import DATA

RESULTS = Path(__file__).resolve().parents[1] / "results" / "protocol_reasoning"
PROMPT = """{protocol}

Question: {question}

A colleague proposes making exactly this one change and nothing else:
"{option}"

Counterfactual check: if this change were made to the protocol above, would it fix the problem described in the question? Answer YES only if the change addresses the actual cause of the problem; answer NO if it would not help or would make things worse. Think step by step, then answer with exactly [ANSWER]YES[/ANSWER] or [ANSWER]NO[/ANSWER]."""
ANS = re.compile(r"\[ANSWER\]\s*(YES|NO)\s*\[/ANSWER\]", re.I)


def items():
    for l in (DATA / "protocolqa-v1-public.jsonl").read_text().splitlines():
        r = json.loads(l)
        yield r["id"], r["protocol"], r["question"], [(r["ideal"], True)] + [(d, False) for d in r["distractors"]]


async def run(model: str, max_tokens: int, conc: int):
    out = RESULTS / f"counterfactual_{model}.jsonl"
    done = {(json.loads(l)["id"], json.loads(l)["option"]) for l in out.read_text().splitlines()} if out.exists() else set()
    llm = LLM(model, max_tokens=max_tokens, concurrency=conc)
    jobs = [(i, p, q, o, y) for i, p, q, opts in items() for o, y in opts if (i, o) not in done]
    pbar = tqdm(total=len(jobs), desc=model, ncols=80)

    async def one(i, p, q, o, y):
        r = await llm.complete(PROMPT.format(protocol=p, question=q, option=o))
        m = ANS.findall(r["text"])
        pred = (m[-1].upper() == "YES") if m else None
        row = {"id": i, "option": o, "gold": y, "pred": pred, "finish_reason": r["finish_reason"],
               "error": r["error"], "tokens": r["usage"].get("completion", 0), "model": model}
        pbar.update(1)
        if r["error"] is None:
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")

    await asyncio.gather(*[one(*j) for j in jobs])
    print(model, "done, cost $%.2f" % llm.cost)
    await llm.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku-5.5", "gpt-6-luna", "glm-5.3-flash"])
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--concurrency", type=int, default=48)
    a = ap.parse_args()

    async def go():
        await asyncio.gather(*[run(m, a.max_tokens, a.concurrency) for m in a.models])

    asyncio.run(go())
