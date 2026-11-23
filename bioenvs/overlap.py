"""Protocol-level split grouping and 13-gram / exact decontamination helpers for BioProBench & LAB-Bench."""
from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BIOPROBENCH_DIR = REPO_ROOT / "data" / "bioprobench"
LABBENCH_DIR = REPO_ROOT / "data" / "labbench"

_WORD_RE = re.compile(r"[a-z0-9]+")
_ORD_TITLE_RE = re.compile(r"'([^']{6,})'")


def normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def word_ngrams(text: str, n: int = 13) -> set[str]:
    toks = _WORD_RE.findall((text or "").lower())
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def protocol_group_key(task: str, row: dict[str, Any]) -> str:
    """Return a deterministic source-protocol key so train, val, and test never share a protocol."""
    t = task.upper()
    for explicit in ("protocol_id", "source_id", "source_protocol", "doi", "article_id"):
        val = row.get(explicit)
        if val:
            return f"{t}:id:{str(val).strip().lower()}"

    if t == "ORD":
        q = str(row.get("question") or "")
        m = _ORD_TITLE_RE.search(q)
        key_src = m.group(1) if m else q
    elif t == "GEN":
        key_src = str(row.get("input") or "").strip()
    elif t == "ERR":
        ctx = row.get("context")
        if isinstance(ctx, dict):
            key_src = f"{ctx.get('purpose', '')} | {ctx.get('prior_step', '')}"
        elif isinstance(ctx, str) and ctx.strip():
            key_src = ctx
        else:
            key_src = str(row.get("corrected_text") or "")
    else:  # PQA or generic
        q = str(row.get("question") or "").replace("____", " ")
        key_src = q

    norm = normalize_text(key_src)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    return f"{t}:proto:{digest}"


def row_full_text(task: str, row: dict[str, Any]) -> str:
    t = task.upper()
    if t == "PQA":
        return f"{row.get('question', '')} {row.get('answer', '')}"
    if t == "ORD":
        steps = " ".join(str(x) for x in (row.get("correct_steps") or []))
        return f"{row.get('question', '')} {steps}"
    if t == "ERR":
        ctx = row.get("context", "")
        ctx_str = ctx if isinstance(ctx, str) else json.dumps(ctx, ensure_ascii=False, sort_keys=True)
        return f"{row.get('corrected_text') or ''} {row.get('corrupted_text') or ''} {ctx_str}"
    if t == "GEN":
        out = " ".join(str(x) for x in (row.get("output") or []))
        return f"{row.get('input', '')} {out}"
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


@functools.lru_cache(maxsize=4)
def load_labbench_index(labbench_dir: str | None = None, n: int = 13) -> tuple[frozenset[str], frozenset[str], dict[str, str]]:
    """Return `(exact_norms, ngrams_set, ngram_to_first_id)` across all public LAB-Bench JSONL files."""
    root = Path(labbench_dir) if labbench_dir else LABBENCH_DIR
    exacts: set[str] = set()
    grams: set[str] = set()
    gram_owner: dict[str, str] = {}
    if not root.exists():
        return frozenset(), frozenset(), {}
    for p in sorted(root.glob("*-public.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = str(obj.get("id", p.name))
            combined = f"{obj.get('protocol', '')} {obj.get('question', '')} {obj.get('ideal', '')}"
            norm = normalize_text(combined)
            if norm:
                exacts.add(norm)
            for g in word_ngrams(combined, n=n):
                grams.add(g)
                gram_owner.setdefault(g, item_id)
    return frozenset(exacts), frozenset(grams), gram_owner


@functools.lru_cache(maxsize=16)
def load_bioprobench_test_index(
    task: str,
    bioprobench_dir: str | None = None,
    n: int = 13,
) -> tuple[frozenset[str], frozenset[str], frozenset[str], dict[str, str]]:
    """Return `(test_proto_groups, test_exact_norms, test_ngrams, ngram_to_test_id)` for `task`."""
    root = Path(bioprobench_dir) if bioprobench_dir else BIOPROBENCH_DIR
    t = task.upper()
    path = root / f"{t}_test.json"
    if not path.exists():
        return frozenset(), frozenset(), frozenset(), {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    groups: set[str] = set()
    exacts: set[str] = set()
    grams: set[str] = set()
    gram_owner: dict[str, str] = {}
    for r in rows:
        rid = str(r.get("id", ""))
        groups.add(protocol_group_key(t, r))
        txt = row_full_text(t, r)
        norm = normalize_text(txt)
        if norm:
            exacts.add(norm)
        for g in word_ngrams(txt, n=n):
            grams.add(g)
            gram_owner.setdefault(g, rid)
    return frozenset(groups), frozenset(exacts), frozenset(grams), gram_owner


def is_row_contaminated(
    task: str,
    row: dict[str, Any],
    *,
    bioprobench_dir: str | None = None,
    labbench_dir: str | None = None,
    n: int = 13,
) -> bool:
    """True when `row` shares a protocol group, exact normalized text, or 13-gram with test or LAB-Bench."""
    t = task.upper()
    te_groups, te_exacts, te_grams, _ = load_bioprobench_test_index(t, bioprobench_dir, n=n)
    if protocol_group_key(t, row) in te_groups:
        return True
    lb_exacts, lb_grams, _ = load_labbench_index(labbench_dir, n=n)
    txt = row_full_text(t, row)
    norm = normalize_text(txt)
    if norm and (norm in te_exacts or norm in lb_exacts):
        return True
    r_grams = word_ngrams(txt, n=n)
    if r_grams and ((r_grams & te_grams) or (r_grams & lb_grams)):
        return True
    return False


def partition_train_val(
    task: str,
    raw_train_rows: list[dict[str, Any]],
    split: str,
    *,
    decontaminate: bool = True,
    val_mod: int = 5,
    bioprobench_dir: str | None = None,
    labbench_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Split `raw_train_rows` into protocol-disjoint `train` and `val` sets with optional decontamination."""
    t = task.upper()
    te_groups, _, _, _ = load_bioprobench_test_index(t, bioprobench_dir)
    out: list[dict[str, Any]] = []
    for row in raw_train_rows:
        gkey = protocol_group_key(t, row)
        # Never allow a protocol present in test into either train or val
        if gkey in te_groups:
            continue
        if decontaminate and is_row_contaminated(
            t, row, bioprobench_dir=bioprobench_dir, labbench_dir=labbench_dir
        ):
            continue
        bucket = int(hashlib.sha256(gkey.encode("utf-8")).hexdigest()[:8], 16) % val_mod
        row_split = "val" if bucket == 0 else "train"
        if row_split == split:
            out.append(row)
    return out
