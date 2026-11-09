"""Loads the public LAB-Bench subsets used for protocol reasoning (text-only, no figures).

Mirrors LAB-Bench's own `randomize_choices` (adds an 'Insufficient information' option so that
precision/coverage can be computed) but uses a seeded RNG so runs are reproducible.
"""
from __future__ import annotations

import ast
import os
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "labbench"
# FigQA / TableQA images are not in this repo. Clone Future-House/LAB-Bench and set LABBENCH_REPO.
REPO = Path(os.environ.get("LABBENCH_REPO", Path(__file__).resolve().parents[1] / "data" / "labbench" / "repo"))
REFUSE = "Insufficient information to answer the question"

# What each LAB-Bench family contributes to "biological protocol reasoning".
FAMILIES = {
    "ProtocolQA": "protocol-troubleshooting",   # given a written protocol + an anomalous outcome, what to change
    "CloningScenarios": "protocol-design",      # multi-step molecular cloning planning
    "SeqQA": "protocol-calculation",            # PCR / restriction digest / ORF calculations a protocol depends on
    # remaining LAB-Bench families (not protocol reasoning; used for the "all of LAB-Bench" comparison)
    "LitQA2": "literature", "DbQA": "database", "SuppQA": "supplementary-info",
    "FigQA": "figure", "TableQA": "table",
}
PROTOCOL_FAMILIES = ["ProtocolQA", "CloningScenarios", "SeqQA"]
ALL_FAMILIES = list(FAMILIES)


@dataclass
class Item:
    id: str
    family: str
    subset: str
    prompt_question: str
    choices: list[str]      # already lettered "(A) ..."
    answer: str             # letter
    unsure: str             # letter of the refuse option
    images: list[str] | None = None   # absolute paths (FigQA / TableQA)


def _family(path: Path) -> str:
    n = path.name
    for prefix, fam in [("protocolqa", "ProtocolQA"), ("cloningscenarios", "CloningScenarios"), ("litqa", "LitQA2"),
                        ("suppqa", "SuppQA"), ("figqa", "FigQA"), ("tableqa", "TableQA")]:
        if n.startswith(prefix):
            return fam
    if n.startswith(("ORF", "PCR", "Prop", "RE-")):
        return "SeqQA"
    return "DbQA"   # dga_task, gene_location_task, ... (13 subtasks)


def load(seed: int = 0, families: list[str] | None = None, limit_per_subset: int | None = None) -> list[Item]:
    items: list[Item] = []
    for p in sorted(DATA.glob("*-public.jsonl")):
        fam = _family(p)
        if families and fam not in families:
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if limit_per_subset:
            rows = rows[:limit_per_subset]
        for r in rows:
            q = r["question"]
            if fam == "ProtocolQA":  # same as LAB-Bench task.py: protocol text is prepended to the question
                q = r["protocol"] + q
            images = None
            if fam == "FigQA":
                images = [str(REPO / "FigQA" / r["figure-path"])]
            elif fam == "TableQA":
                tp = r["table-path"]
                tp = ast.literal_eval(tp) if isinstance(tp, str) else tp   # stored as a stringified list
                images = [str(REPO / "TableQA" / t) for t in tp]
            ideal, dis = r["ideal"], r["distractors"]
            if isinstance(dis, str):
                dis = ast.literal_eval(dis) if dis.startswith("[") else [dis]
            rng = random.Random(f"{seed}:{r['id']}")
            cs = [ideal, REFUSE, *dis]
            perm = list(range(len(cs)))
            rng.shuffle(perm)
            letters = string.ascii_uppercase
            choices = [f"({letters[i]}) {cs[s]}" for i, s in enumerate(perm)]
            items.append(Item(r["id"], fam, p.stem.replace("-v1-public", ""), q, choices,
                              letters[perm.index(0)], letters[perm.index(1)], images))
    return items
