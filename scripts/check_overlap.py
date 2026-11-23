"""Check 13-gram and exact-match overlap between training files, BioProBench test, and LAB-Bench.

Usage:
    python scripts/check_overlap.py [--output results/protocol_reasoning/bioprobench_overlap.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bioenvs.overlap import (  # noqa: E402
    BIOPROBENCH_DIR,
    LABBENCH_DIR,
    load_bioprobench_test_index,
    load_labbench_index,
    normalize_text,
    partition_train_val,
    protocol_group_key,
    row_full_text,
    word_ngrams,
)
from common.run_meta import build_run_meta  # noqa: E402


def check_bioprobench_overlap(
    *,
    bioprobench_dir: Path = BIOPROBENCH_DIR,
    labbench_dir: Path = LABBENCH_DIR,
    ngram_size: int = 13,
    extra_train_files: list[str] | None = None,
) -> dict[str, Any]:
    lb_exacts, lb_grams, lb_owner = load_labbench_index(str(labbench_dir), n=ngram_size)
    tasks_report: dict[str, Any] = {}
    data_files: list[Path] = []

    for task in ("PQA", "ORD", "ERR", "GEN"):
        tr_path = bioprobench_dir / f"{task}_train.json"
        te_path = bioprobench_dir / f"{task}_test.json"
        if not tr_path.exists() or not te_path.exists():
            continue
        data_files.extend([tr_path, te_path])
        tr_rows = json.loads(tr_path.read_text(encoding="utf-8"))
        te_rows = json.loads(te_path.read_text(encoding="utf-8"))
        te_groups, te_exacts, te_grams, te_owner = load_bioprobench_test_index(
            task, str(bioprobench_dir), n=ngram_size
        )

        vs_test_matches: list[dict[str, str]] = []
        vs_labbench_matches: list[dict[str, str]] = []
        for r in tr_rows:
            rid = str(r.get("id", ""))
            txt = row_full_text(task, r)
            norm = normalize_text(txt)
            grams = word_ngrams(txt, n=ngram_size)
            gkey = protocol_group_key(task, r)

            if gkey in te_groups or (norm and norm in te_exacts) or (grams & te_grams):
                hit_g = next(iter(grams & te_grams), None)
                vs_test_matches.append({
                    "train_id": rid,
                    "test_id": te_owner.get(hit_g, "protocol_or_exact") if hit_g else "protocol_or_exact",
                    "match_type": "13gram" if hit_g else ("exact" if norm in te_exacts else "protocol_group"),
                })
            if (norm and norm in lb_exacts) or (grams & lb_grams):
                hit_lb = next(iter(grams & lb_grams), None)
                vs_labbench_matches.append({
                    "train_id": rid,
                    "labbench_id": lb_owner.get(hit_lb, "exact") if hit_lb else "exact",
                    "match_type": "13gram" if hit_lb else "exact",
                })

        test_vs_lb: list[dict[str, str]] = []
        for r in te_rows:
            rid = str(r.get("id", ""))
            txt = row_full_text(task, r)
            norm = normalize_text(txt)
            grams = word_ngrams(txt, n=ngram_size)
            if (norm and norm in lb_exacts) or (grams & lb_grams):
                hit_lb = next(iter(grams & lb_grams), None)
                test_vs_lb.append({
                    "test_id": rid,
                    "labbench_id": lb_owner.get(hit_lb, "exact") if hit_lb else "exact",
                })

        clean_train = partition_train_val(
            task, tr_rows, "train", decontaminate=True, bioprobench_dir=str(bioprobench_dir), labbench_dir=str(labbench_dir)
        )
        clean_val = partition_train_val(
            task, tr_rows, "val", decontaminate=True, bioprobench_dir=str(bioprobench_dir), labbench_dir=str(labbench_dir)
        )
        train_protos = {protocol_group_key(task, r) for r in clean_train}
        val_protos = {protocol_group_key(task, r) for r in clean_val}

        tasks_report[task] = {
            "n_raw_train": len(tr_rows),
            "n_test": len(te_rows),
            "n_clean_train": len(clean_train),
            "n_clean_val": len(clean_val),
            "protocol_overlap_train_val": len(train_protos & val_protos),
            "protocol_overlap_train_test": len(train_protos & set(te_groups)),
            "protocol_overlap_val_test": len(val_protos & set(te_groups)),
            "raw_train_vs_test_count": len(vs_test_matches),
            "raw_train_vs_test_matches": vs_test_matches,
            "raw_train_vs_labbench_count": len(vs_labbench_matches),
            "raw_train_vs_labbench_matches": vs_labbench_matches,
            "test_vs_labbench_count": len(test_vs_lb),
            "test_vs_labbench_matches": test_vs_lb,
        }

    extra_reports: dict[str, Any] = {}
    all_te_exacts: set[str] = set()
    all_te_grams: set[str] = set()
    for task in ("PQA", "ORD", "ERR", "GEN"):
        _, tex, tgr, _ = load_bioprobench_test_index(task, str(bioprobench_dir), n=ngram_size)
        all_te_exacts |= set(tex)
        all_te_grams |= set(tgr)

    for fp in extra_train_files or []:
        p = Path(fp)
        if not p.exists():
            continue
        data_files.append(p)
        lines = p.read_text(encoding="utf-8").splitlines()
        hits_te: list[str] = []
        hits_lb: list[str] = []
        n_rows = 0
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            n_rows += 1
            try:
                obj = json.loads(line)
                rid = str(obj.get("id") or (obj.get("meta") or {}).get("id") or f"row-{idx}")
                txt = json.dumps(obj, ensure_ascii=False)
            except json.JSONDecodeError:
                rid = f"line-{idx}"
                txt = line
            norm = normalize_text(txt)
            grams = word_ngrams(txt, n=ngram_size)
            if (norm and norm in all_te_exacts) or (grams & all_te_grams):
                hits_te.append(rid)
            if (norm and norm in lb_exacts) or (grams & lb_grams):
                hits_lb.append(rid)
        extra_reports[str(p)] = {
            "n_rows": n_rows,
            "vs_bioprobench_test_count": len(hits_te),
            "vs_bioprobench_test_ids": hits_te[:50],
            "vs_labbench_count": len(hits_lb),
            "vs_labbench_ids": hits_lb[:50],
        }

    return {
        "meta": build_run_meta(data_files=data_files, extra={"ngram_size": ngram_size}),
        "ngram_size": ngram_size,
        "bioprobench": tasks_report,
        "extra_files": extra_reports,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Check 13-gram and exact overlap against BioProBench test and LAB-Bench.")
    ap.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / "protocol_reasoning" / "bioprobench_overlap.json"),
        help="Output JSON report path.",
    )
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--train-files", nargs="*", default=[], help="Optional extra JSONL/text training files to check.")
    args = ap.parse_args(argv)

    report = check_bioprobench_overlap(ngram_size=args.ngram, extra_train_files=args.train_files)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for task, stats in report["bioprobench"].items():
        print(
            f"{task:4s}: raw_train={stats['n_raw_train']} -> clean_train={stats['n_clean_train']}, "
            f"clean_val={stats['n_clean_val']} | raw_vs_test={stats['raw_train_vs_test_count']} "
            f"raw_vs_labbench={stats['raw_train_vs_labbench_count']}"
        )
    return report


if __name__ == "__main__":
    main()
