"""Run models on the LAB-Bench protocol-reasoning subset.

    python -m labeval.run --models haiku-5.5 gpt-6-luna glm-5.3-flash --seeds 0 1
Writes results/protocol_reasoning/labbench_<model>_seed<k>.jsonl (one row per question, raw output included).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from tqdm import tqdm

from common.llm import LLM
from labeval.data import Item, load

import os

RESULTS = Path(os.environ.get("LABEVAL_RESULTS", Path(__file__).resolve().parents[1] / "results" / "protocol_reasoning"))

# LAB-Bench's zero-shot template (labbench/zero_shot.py) with its CoT suffix.
TEMPLATE = """The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.
Think step by step before answering.

Question: {question}

Options:
{answers}

You MUST include the letter of the correct answer within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is the correct letter.
Always answer in exactly this format of a single letter between the two tags, even if you are unsure.
We require this because we use automatic parsing."""

ANS_RE = re.compile(r"\[ANSWER\]\s*\(?([A-Z])\)?\s*\[/ANSWER\]", re.I)


def build_prompt(it: Item) -> str | list[dict]:
    text = TEMPLATE.format(question=it.prompt_question, answers="\n".join(it.choices))
    if not it.images:
        return text
    parts = []
    for path in it.images:
        mime, b64 = encode_image(path)
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return [{"role": "user", "content": parts + [{"type": "text", "text": text}]}]


def encode_image(path: str, max_side: int = 2000) -> tuple[str, str]:
    """Downscale very large figures so request size / cost stay bounded; always re-encode as PNG/JPEG."""
    import base64
    from io import BytesIO

    from PIL import Image
    im = Image.open(path)
    fmt = "JPEG" if im.format == "JPEG" else "PNG"
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    if fmt == "JPEG" and im.mode != "RGB":
        im = im.convert("RGB")
    buf = BytesIO()
    im.save(buf, format=fmt)
    return f"image/{fmt.lower()}", base64.b64encode(buf.getvalue()).decode()


def parse(text: str, n_choices: int) -> str | None:
    m = ANS_RE.findall(text or "")
    if not m:
        return None
    a = m[-1].upper()
    return a if ord(a) - 65 < n_choices else None


async def run_model(model: str, seed: int, args) -> None:
    out = RESULTS / f"labbench_{model}_seed{seed}.jsonl"
    if args.redo_truncated and out.exists():
        # keep the original (small-budget) rows for the sensitivity analysis, then re-run only truncated questions
        old = [json.loads(l) for l in out.read_text().splitlines()]
        bak = RESULTS / f"labbench_{model}_seed{seed}.budget{old[0].get('max_tokens', 'orig')}.jsonl"
        if not bak.exists():
            bak.write_text("\n".join(json.dumps(r) for r in old) + "\n")
        fams = set(args.families or [])   # only the selected families are re-run; others keep their rows
        out.write_text("".join(json.dumps(r) + "\n" for r in old
                               if r["finish_reason"] != "length" or (fams and r["family"] not in fams)))
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    items = [i for i in load(seed, args.families, args.limit) if i.id not in done]
    llm = LLM(model, max_tokens=args.max_tokens, concurrency=args.concurrency)
    pbar = tqdm(total=len(items), desc=f"{model} s{seed}", ncols=80)

    async def one(it: Item):
        r = await llm.complete(build_prompt(it))
        pred = parse(r["text"], len(it.choices))
        row = {"id": it.id, "family": it.family, "subset": it.subset, "seed": seed, "model": model,
               "n_choices": len(it.choices), "answer": it.answer, "unsure": it.unsure, "pred": pred,
               "correct": pred == it.answer, "sure": pred is not None and pred != it.unsure,
               "parse_fail": pred is None, "max_tokens": args.max_tokens, "finish_reason": r["finish_reason"], "error": r["error"],
               "usage": r["usage"], "output": r["text"]}
        pbar.update(1)
        if row["error"] is None:   # persist as we go; API failures are retried on the next invocation
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    rows = await asyncio.gather(*[one(i) for i in items])
    pbar.close()
    print(f"{model} seed{seed}: {len(rows)} new rows, errors={sum(r['error'] is not None for r in rows)}, "
          f"cost=${llm.cost:.3f}")
    await llm.aclose()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku-5.5", "gpt-6-luna", "glm-5.3-flash"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--families", nargs="*", default=None, help="default: all 8 LAB-Bench families")
    ap.add_argument("--limit", type=int, default=None, help="max questions per subset (smoke tests)")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--redo-truncated", action="store_true",
                    help="re-run only questions whose output hit the token limit (use with a larger --max-tokens)")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    async def go():
        await asyncio.gather(*[run_model(m, s, args) for m in args.models for s in args.seeds])

    asyncio.run(go())


if __name__ == "__main__":
    main()
