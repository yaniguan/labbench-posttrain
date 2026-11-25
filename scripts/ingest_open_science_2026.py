"""Ingest & SplitGuard-audit 2024-2026 Open Science SFT/RLVR datasets and eval-only benchmarks.

Downloads on-demand ONLY into gitignored `data/open_science_2026/` (never modifying tracked
`data/v0`, `data/calisol`, `data/bioprobench`, or `data/labbench`).

Examples:
    python scripts/ingest_open_science_2026.py --list
    python scripts/ingest_open_science_2026.py --datasets aqsoldb,massspecgym --limit 200
    python scripts/ingest_open_science_2026.py --datasets chembench,supergpqa --allow-eval-download --limit 50
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from labpt.io import write_jsonl  # noqa: E402
from labpt.schema import Record  # noqa: E402
from labrl.core.types import SplitLeakageError  # noqa: E402
from labrl.data.open_datasets_2026 import (  # noqa: E402
    EvalOnlyBenchmarkItem,
    OpenScienceDatasetSpec,
    assert_trainable_dataset,
    get_open_dataset_spec,
    list_open_datasets_2026,
    normalize_eval_only_item,
    normalize_open_science_record,
    partition_open_science_records,
)

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "open_science_2026"
PROTECTED_DATA_SUBDIRS = frozenset({"v0", "calisol", "bioprobench", "labbench"})
HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def validate_output_directory(out_dir: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    """Ensure downloads/artifacts inside the repo write ONLY to gitignored `data/` paths."""
    resolved = Path(out_dir).resolve()
    repo_res = repo_root.resolve()
    data_root = (repo_res / "data").resolve()

    try:
        rel_repo = resolved.relative_to(repo_res)
    except ValueError:
        # Outside repo root (e.g., pytest `tmp_path`) is allowed for isolated testing.
        return resolved

    if not rel_repo.parts or rel_repo.parts[0] != "data" or resolved == data_root:
        raise ValueError(
            f"Refusing to write ingested dataset files to `{resolved}`: inside `{repo_res}`, "
            "output must reside under gitignored `data/open_science_2026/` (or `data/raw/`, `data/interim/`)."
        )
    if len(rel_repo.parts) >= 2 and rel_repo.parts[1] in PROTECTED_DATA_SUBDIRS:
        raise ValueError(
            f"Refusing to overwrite checked-in dataset directory `data/{rel_repo.parts[1]}` (`{resolved}`)."
        )
    return resolved


def load_local_rows(input_file: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read local `.jsonl`, `.json`, `.csv`, or `.tsv` fixtures for offline ingestion."""
    suffix = input_file.suffix.lower()
    rows: list[dict[str, Any]] = []
    if suffix == ".jsonl":
        for line in input_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
        return rows
    if suffix == ".json":
        payload = json.loads(input_file.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("rows", [])
        for obj in items:
            row = obj["row"] if isinstance(obj, Mapping) and "row" in obj and isinstance(obj["row"], Mapping) else obj
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
        return rows
    delim = "\t" if suffix == ".tsv" else ","
    with input_file.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            rows.append(dict(r))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _fetch_hf_datasets_server_rows(
    spec: OpenScienceDatasetSpec,
    *,
    limit: int,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Fetch paginated rows via the Hugging Face Datasets Server HTTP API."""
    collected: list[dict[str, Any]] = []
    offset = 0
    while len(collected) < limit:
        page_len = min(100, limit - len(collected))
        resp = requests.get(
            HF_ROWS_ENDPOINT,
            params={
                "dataset": spec.hf_repo_id,
                "config": spec.hf_default_config,
                "split": spec.hf_default_split,
                "offset": offset,
                "length": page_len,
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            break
        batch = resp.json().get("rows") or []
        if not batch:
            break
        for entry in batch:
            row = entry.get("row") if isinstance(entry, Mapping) else None
            if isinstance(row, Mapping):
                collected.append(dict(row))
        if len(batch) < page_len:
            break
        offset += len(batch)
    return collected[:limit]


def _stream_hf_delimited_rows(url: str, *, delimiter: str, limit: int, timeout: int = 60) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        lines_iter = (line.decode("utf-8", errors="replace") for line in resp.iter_lines() if line)
        reader = csv.DictReader(lines_iter, delimiter=delimiter)
        for r in reader:
            rows.append(dict(r))
            if len(rows) >= limit:
                break
    return rows


def _stream_hf_jsonl_rows(url: str, *, limit: int, timeout: int = 60) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            rows.append(json.loads(raw_line.decode("utf-8", errors="replace")))
            if len(rows) >= limit:
                break
    return rows


def fetch_live_dataset_rows(spec: OpenScienceDatasetSpec, *, limit: int = 200) -> list[dict[str, Any]]:
    """Fetch live rows from Hugging Face Hub with dataset-aware streaming fallbacks."""
    # For SuperGPQA, request extra rows because scientific discipline filtering retains ~25-35% of rows.
    fetch_cap = limit * 5 if spec.dataset_id == "supergpqa" else limit
    rows = _fetch_hf_datasets_server_rows(spec, limit=fetch_cap)
    if rows:
        return rows

    if spec.dataset_id == "massspecgym":
        return _stream_hf_delimited_rows(
            "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/MassSpecGym1.5.tsv",
            delimiter="\t",
            limit=limit,
        )
    if spec.dataset_id == "aqsoldb":
        return _stream_hf_delimited_rows(
            "https://huggingface.co/datasets/maomlab/AqSolDB/resolve/main/AqSolDB/train.csv",
            delimiter=",",
            limit=limit,
        )
    if spec.dataset_id == "supergpqa":
        return _stream_hf_jsonl_rows(
            "https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/main/SuperGPQA-all.jsonl",
            limit=fetch_cap,
        )
    if spec.dataset_id == "bixbench":
        return _stream_hf_jsonl_rows(
            "https://huggingface.co/datasets/futurehouse/BixBench/resolve/main/BixBench.jsonl",
            limit=limit,
        )
    if spec.dataset_id == "ether0_benchmark":
        import pandas as pd

        resp = requests.get(
            "https://huggingface.co/datasets/futurehouse/ether0-benchmark/resolve/main/data/test-00000-of-00001.parquet",
            timeout=60,
        )
        resp.raise_for_status()
        df = pd.read_parquet(io.BytesIO(resp.content)).head(limit)
        return df.to_dict(orient="records")

    raise RuntimeError(
        f"Dataset `{spec.dataset_id}` (`{spec.hf_repo_id}`) uses archive packaging on Hugging Face; "
        "download its archive to `data/open_science_2026/raw/` or pass `--input-file <path>`."
    )


def ingest_trainable_dataset(
    dataset_id: str,
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Normalize, deduplicate, `SplitGuard`-partition, and write `train`/`val`/`test` JSONL + manifest."""
    spec = assert_trainable_dataset(dataset_id, stage="ingest_trainable_dataset")
    safe_out = validate_output_directory(out_dir)
    ds_out = safe_out / spec.dataset_id
    ds_out.mkdir(parents=True, exist_ok=True)

    normalized: list[Record] = []
    seen_dedup: set[str] = set()
    dropped_invalid = 0

    for row in raw_rows:
        try:
            rec = normalize_open_science_record(spec.dataset_id, row)
        except SplitLeakageError:
            # Never swallow canary or eval_only leakage violations!
            raise
        except (ValueError, KeyError, TypeError):
            dropped_invalid += 1
            continue
        dk = rec.dedup_key()
        if dk in seen_dedup:
            continue
        seen_dedup.add(dk)
        normalized.append(rec)

    partitioned = partition_open_science_records(spec.dataset_id, normalized)

    for sp in ("train", "val", "test"):
        sp_rows = [r for r in partitioned.records if r.split == sp]
        write_jsonl(sp_rows, ds_out / f"{sp}.jsonl")

    manifest = {
        "dataset_id": spec.dataset_id,
        "hf_repo_id": spec.hf_repo_id,
        "url": spec.url,
        "upstream_url": spec.upstream_url,
        "spdx_license": spec.spdx_license,
        "commercial_use_ok": spec.commercial_use_ok,
        "share_alike": spec.share_alike,
        "eval_only": False,
        "trainable": True,
        "group_key_strategy": spec.group_key_strategy,
        "scaffold_split_guard_signature": partitioned.scaffold_guard.signature,
        "doi_split_guard_signature": partitioned.doi_guard.signature,
        "n_records": len(partitioned.records),
        "n_invalid_dropped": dropped_invalid,
        "counts_by_split": partitioned.counts_by_split,
        "cross_split_scaffold_overlap": 0,
        "cross_split_doi_overlap": 0,
    }
    (ds_out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def ingest_eval_only_dataset(
    dataset_id: str,
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    allow_eval_download: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Export an `eval_only=True` benchmark ONLY into isolated `eval_only/` with mandatory canary stamps."""
    spec = get_open_dataset_spec(dataset_id)
    if not spec.eval_only:
        raise ValueError(f"`{spec.dataset_id}` is trainable (`eval_only=False`); use `ingest_trainable_dataset`.")
    if not allow_eval_download:
        raise SplitLeakageError(
            f"Refusing to ingest `eval_only=True` benchmark `{spec.dataset_id}` (`{spec.hf_repo_id}`) "
            "without explicit `--allow-eval-download` isolation flag."
        )
    safe_out = validate_output_directory(out_dir)
    eval_dir = safe_out / "eval_only"
    eval_dir.mkdir(parents=True, exist_ok=True)

    items: list[EvalOnlyBenchmarkItem] = []
    for row in raw_rows:
        item = normalize_eval_only_item(spec.dataset_id, row)
        if item is None:
            continue
        items.append(item)
        if limit is not None and len(items) >= limit:
            break

    jsonl_path = eval_dir / f"{spec.dataset_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=True) + "\n")

    manifest = {
        "dataset_id": spec.dataset_id,
        "hf_repo_id": spec.hf_repo_id,
        "url": spec.url,
        "spdx_license": spec.spdx_license,
        "eval_only": True,
        "trainable": False,
        "canary_protected": True,
        "n_eval_items": len(items),
        "output_path": str(jsonl_path),
    }
    (eval_dir / f"{spec.dataset_id}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _resolve_requested_datasets(spec_arg: str) -> list[OpenScienceDatasetSpec]:
    cleaned = (spec_arg or "trainable").strip().lower()
    if cleaned == "trainable":
        return list_open_datasets_2026(trainable_only=True)
    if cleaned == "eval":
        return list_open_datasets_2026(eval_only=True)
    if cleaned == "all":
        return list_open_datasets_2026()
    ids = [tok.strip() for tok in spec_arg.split(",") if tok.strip()]
    return [get_open_dataset_spec(ds_id) for ds_id in ids]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print JSON registry of 2024-2026 open science datasets and exit.",
    )
    ap.add_argument(
        "--datasets",
        default="trainable",
        help="Comma-separated dataset IDs, or 'trainable' (default), 'eval', or 'all'.",
    )
    ap.add_argument("--limit", type=int, default=200, help="Max rows per dataset (default: 200).")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Gitignored output root under `data/` (default: data/open_science_2026).",
    )
    ap.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional local JSONL/JSON/CSV/TSV fixture for offline ingestion (single dataset).",
    )
    ap.add_argument(
        "--mode",
        choices=("auto", "train", "eval"),
        default="auto",
        help="Ingestion mode; `--mode train` strictly rejects `eval_only=True` benchmarks.",
    )
    ap.add_argument(
        "--allow-eval-download",
        action="store_true",
        help="Explicitly authorize writing `eval_only=True` benchmarks into isolated `eval_only/`.",
    )
    return ap


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        payload = {"datasets": [s.to_dict() for s in list_open_datasets_2026()]}
        print(json.dumps(payload, indent=2))
        return payload

    specs = _resolve_requested_datasets(args.datasets)
    if args.input_file is not None and len(specs) != 1:
        raise ValueError("`--input-file` requires specifying exactly one dataset in `--datasets`.")

    results: dict[str, Any] = {}
    for spec in specs:
        if args.mode == "train":
            assert_trainable_dataset(spec, stage="cli_mode_train")
        if spec.eval_only and not args.allow_eval_download:
            raise SplitLeakageError(
                f"Dataset `{spec.dataset_id}` (`{spec.hf_repo_id}`) is `eval_only=True` and blocked from "
                "default trainable ingestion; pass `--allow-eval-download` to export isolated eval items."
            )

        raw_rows = (
            load_local_rows(args.input_file, limit=args.limit * 5 if spec.dataset_id == "supergpqa" else args.limit)
            if args.input_file is not None
            else fetch_live_dataset_rows(spec, limit=args.limit)
        )

        if spec.eval_only:
            results[spec.dataset_id] = ingest_eval_only_dataset(
                spec.dataset_id,
                raw_rows,
                out_dir=args.out_dir,
                allow_eval_download=args.allow_eval_download,
                limit=args.limit,
            )
        else:
            results[spec.dataset_id] = ingest_trainable_dataset(
                spec.dataset_id,
                raw_rows[: args.limit],
                out_dir=args.out_dir,
            )

    summary = {"out_dir": str(validate_output_directory(args.out_dir)), "ingested": results}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
