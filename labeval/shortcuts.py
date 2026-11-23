"""Deterministic no-model shortcut baselines for LAB-Bench families (#15).

Usage:
    python -m labeval.shortcuts

Computes offline, API-free baselines per LAB-Bench family:
  - chance_with_refuse: uniform random guess over all displayed options (including 'Insufficient information')
  - chance_excl_refuse: uniform random guess excluding the 'Insufficient information' option
  - longest_chars     : pick the non-refusal option with the most characters (exact ties split 1/k)
  - longest_words     : pick the non-refusal option with the most whitespace words (exact ties split 1/k)
  - shortest_chars    : pick the non-refusal option with the fewest characters (exact ties split 1/k)
  - most_common_pos   : always pick the modal gold letter in deterministic seed=0 choice order
  - always_refuse     : always select 'Insufficient information' (0.0 accuracy, 0.0 coverage)
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
from typing import Any

from common.run_meta import build_run_meta
from labeval.data import ALL_FAMILIES, DATA, _family, load

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "protocol_reasoning"


def score_item_length_shortcut(
    ideal: str,
    distractors: list[str],
    *,
    mode: str = "longest_chars",
) -> float:
    """Score a single multiple-choice item under a length heuristic with exact `1/k` tie splitting.

    `'Insufficient information to answer the question'` is excluded from candidate comparison,
    matching standard content-length shortcut analysis.
    """
    candidates = [str(ideal), *[str(d) for d in distractors]]
    if mode == "longest_chars":
        lengths = [len(c) for c in candidates]
        target = max(lengths)
    elif mode == "shortest_chars":
        lengths = [len(c) for c in candidates]
        target = min(lengths)
    elif mode == "longest_words":
        lengths = [len(c.split()) for c in candidates]
        target = max(lengths)
    elif mode == "shortest_words":
        lengths = [len(c.split()) for c in candidates]
        target = min(lengths)
    else:
        raise ValueError(f"Unknown shortcut mode: {mode!r}")

    tied_indices = [idx for idx, val in enumerate(lengths) if val == target]
    if 0 in tied_indices:
        return 1.0 / len(tied_indices)
    return 0.0


def compute_labbench_shortcuts(*, data_dir: Path = DATA, seed: int = 0) -> dict[str, Any]:
    raw_by_fam: dict[str, list[dict[str, Any]]] = {f: [] for f in ALL_FAMILIES}
    files_used: list[Path] = []
    for p in sorted(data_dir.glob("*-public.jsonl")):
        files_used.append(p)
        fam = _family(p)
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            dis = obj["distractors"]
            if isinstance(dis, str):
                dis = ast.literal_eval(dis) if dis.startswith("[") else [dis]
            raw_by_fam.setdefault(fam, []).append({
                "id": obj["id"],
                "ideal": str(obj["ideal"]),
                "distractors": [str(x) for x in dis],
            })

    seeded_items = load(seed=seed)
    seeded_by_fam: dict[str, list[Any]] = {f: [] for f in ALL_FAMILIES}
    for it in seeded_items:
        seeded_by_fam.setdefault(it.family, []).append(it)

    families_out: dict[str, dict[str, Any]] = {}
    for fam in ALL_FAMILIES:
        rows = raw_by_fam.get(fam, [])
        if not rows:
            continue
        s_items = seeded_by_fam.get(fam, [])
        n = len(rows)
        ans_counts = Counter(it.answer for it in s_items)
        modal_letter, modal_hits = ans_counts.most_common(1)[0] if ans_counts else ("A", 0)

        families_out[fam] = {
            "n": n,
            "chance_with_refuse": round(sum(1.0 / (len(r["distractors"]) + 2) for r in rows) / n, 4),
            "chance_excl_refuse": round(sum(1.0 / (len(r["distractors"]) + 1) for r in rows) / n, 4),
            "longest_chars": round(
                sum(score_item_length_shortcut(r["ideal"], r["distractors"], mode="longest_chars") for r in rows) / n,
                4,
            ),
            "longest_words": round(
                sum(score_item_length_shortcut(r["ideal"], r["distractors"], mode="longest_words") for r in rows) / n,
                4,
            ),
            "shortest_chars": round(
                sum(score_item_length_shortcut(r["ideal"], r["distractors"], mode="shortest_chars") for r in rows) / n,
                4,
            ),
            "most_common_position": round(modal_hits / max(len(s_items), 1), 4),
            "modal_letter_seed0": modal_letter,
            "always_insufficient_info": 0.0,
        }

    macro_all = {
        k: round(sum(v[k] for v in families_out.values()) / max(len(families_out), 1), 4)
        for k in (
            "chance_with_refuse",
            "chance_excl_refuse",
            "longest_chars",
            "longest_words",
            "shortest_chars",
            "most_common_position",
            "always_insufficient_info",
        )
    }

    return {
        "meta": build_run_meta(
            data_files=files_used,
            sampling={"seed": seed},
            extra={"tie_rule": "exact ties split evenly (1/k), 'Insufficient information' excluded from length rules"},
        ),
        "families": families_out,
        "macro_all8": macro_all,
    }


def render_shortcuts_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LAB-Bench No-Model Shortcut Baselines",
        "",
        "Counting rule: `'Insufficient information to answer the question'` is excluded from option-length comparisons; exact length ties split credit evenly (`1/k`). Position baseline uses `seed=0` shuffled choice order.",
        "",
        "| family | n | chance (with refuse) | chance (excl. refuse) | longest (chars) | longest (words) | shortest (chars) | modal pos (seed 0) | always refuse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fam, d in report["families"].items():
        lines.append(
            f"| {fam} | {d['n']} | {d['chance_with_refuse']:.3f} | {d['chance_excl_refuse']:.3f} | "
            f"{d['longest_chars']:.3f} | {d['longest_words']:.3f} | {d['shortest_chars']:.3f} | "
            f"{d['most_common_position']:.3f} ({d['modal_letter_seed0']}) | {d['always_insufficient_info']:.3f} |"
        )
    m = report["macro_all8"]
    lines.append(
        f"| **macro (all 8)** | **{sum(d['n'] for d in report['families'].values())}** | "
        f"**{m['chance_with_refuse']:.3f}** | **{m['chance_excl_refuse']:.3f}** | "
        f"**{m['longest_chars']:.3f}** | **{m['longest_words']:.3f}** | "
        f"**{m['shortest_chars']:.3f}** | **{m['most_common_position']:.3f}** | "
        f"**{m['always_insufficient_info']:.3f}** |"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Compute deterministic no-model shortcut baselines on LAB-Bench.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = compute_labbench_shortcuts(seed=args.seed)
    md = render_shortcuts_markdown(report)
    (out_dir / "labbench_shortcuts.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "labbench_shortcuts.md").write_text(md, encoding="utf-8")
    print(md, end="")
    return report


if __name__ == "__main__":
    main()
