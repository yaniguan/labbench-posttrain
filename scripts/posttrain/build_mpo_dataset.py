"""Build unified MPO (Paired DPO + Unpaired BCO + SFT) datasets from `data/v0`.

Combines:
1. Paired trajectories from `data/v0/tasks/pairs_{split}.jsonl` (`success_vs_failure`, `corrupted_actions`, etc.)
2. Unpaired experimental outcome trajectories from `data/v0/{split}.parquet` & `data/v0/sample.jsonl`
   where `outcome.success` is True (`label=1`) or False (`label=0`), unlocking thousands of real lab
   procedures and fault logs that have no exact same-prompt twin for DPO.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_mpo_records(
    *,
    data_dir: str | Path = REPO_ROOT / "data" / "v0",
    split: str = "train",
    pair_kinds: tuple[str, ...] = ("success_vs_failure", "corrupted_actions", "wrong_provenance"),
    max_paired: int | None = None,
    max_unpaired: int | None = None,
) -> dict[str, Any]:
    root = Path(data_dir)
    paired_rows: list[dict[str, Any]] = []
    unpaired_rows: list[dict[str, Any]] = []

    pairs_file = root / "tasks" / f"pairs_{split}.jsonl"
    if pairs_file.exists():
        with open(pairs_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if pair_kinds and row.get("kind") not in pair_kinds:
                    continue
                paired_rows.append({
                    "record_type": "paired",
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                    "kind": row.get("kind", "paired"),
                    "split": split,
                })
                if max_paired is not None and len(paired_rows) >= max_paired:
                    break

    # Mine unpaired experimental outcomes from parquet (if present) or sample.jsonl + sft_{split}.jsonl
    parquet_file = root / f"{split}.parquet"
    if parquet_file.exists():
        import pandas as pd

        df = pd.read_parquet(parquet_file)
        for rec in df.to_dict(orient="records"):
            if max_unpaired is not None and len(unpaired_rows) >= max_unpaired:
                break
            outcome = rec.get("outcome")
            if isinstance(outcome, str):
                try:
                    outcome = json.loads(outcome)
                except Exception:  # noqa: BLE001
                    outcome = None
            if not isinstance(outcome, dict) or outcome.get("success") is None:
                continue
            raw_proto = rec.get("protocol_text")
            proto = raw_proto.strip() if isinstance(raw_proto, str) else ""
            if len(proto) < 8:
                continue
            entity = rec.get("entity")
            conds = rec.get("conditions")
            prompt = [
                {"role": "system", "content": "You are a careful lab-automation assistant."},
                {
                    "role": "user",
                    "content": f"Draft a laboratory execution protocol or log entry for:\nentity: {entity}\nconditions: {conds}",
                },
            ]
            unpaired_rows.append({
                "record_type": "unpaired",
                "prompt": prompt,
                "completion": proto,
                "label": 1 if bool(outcome["success"]) else 0,
                "kind": f"unpaired_{rec.get('modality', 'lab')}",
                "split": split,
                "id": rec.get("id"),
            })
    else:
        sample_file = root / "sample.jsonl"
        if sample_file.exists():
            with open(sample_file, encoding="utf-8") as f:
                for line in f:
                    if max_unpaired is not None and len(unpaired_rows) >= max_unpaired:
                        break
                    rec = json.loads(line)
                    if rec.get("split") != split:
                        continue
                    outcome = rec.get("outcome") or {}
                    if outcome.get("success") is None:
                        continue
                    proto = (rec.get("protocol_text") or "").strip()
                    if len(proto) < 8:
                        continue
                    unpaired_rows.append({
                        "record_type": "unpaired",
                        "prompt": [
                            {"role": "user", "content": f"Execute lab step for {rec.get('entity')}: {rec.get('conditions')}"}
                        ],
                        "completion": proto,
                        "label": 1 if bool(outcome["success"]) else 0,
                        "kind": f"unpaired_{rec.get('modality', 'lab')}",
                        "split": split,
                        "id": rec.get("id"),
                    })

    all_rows = paired_rows + unpaired_rows
    n_pos = sum(1 for r in unpaired_rows if r["label"] == 1)
    n_neg = sum(1 for r in unpaired_rows if r["label"] == 0)
    summary = {
        "split": split,
        "n_total": len(all_rows),
        "n_paired": len(paired_rows),
        "n_paired_success_vs_failure": sum(1 for r in paired_rows if r["kind"] == "success_vs_failure"),
        "n_unpaired": len(unpaired_rows),
        "n_unpaired_positive": n_pos,
        "n_unpaired_negative": n_neg,
    }
    return {"records": all_rows, "summary": summary}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Build MPO hybrid dataset from data/v0.")
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data" / "v0"))
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    parser.add_argument("--max-paired", type=int, default=1000)
    parser.add_argument("--max-unpaired", type=int, default=2000)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    built = build_mpo_records(
        data_dir=args.data_dir,
        split=args.split,
        max_paired=args.max_paired,
        max_unpaired=args.max_unpaired,
    )
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for r in built["records"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(built["summary"], indent=2))
    return built


if __name__ == "__main__":
    main()
