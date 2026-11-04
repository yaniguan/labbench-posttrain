from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import Record

RAW = Path("data/raw")
INTERIM = Path("data/interim")


def write_jsonl(records: Iterable[Record], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[Record]:
    with open(path) as f:
        return [Record.model_validate_json(l) for l in f if l.strip()]


def to_row(r: Record) -> dict:
    d = r.model_dump()
    for k in ("entity", "conditions", "protocol_steps", "outcome", "annotations"):
        d[k] = json.dumps(d[k], default=str)
    return d


def from_row(d: dict) -> Record:
    d = dict(d)
    for k in ("entity", "conditions", "protocol_steps", "outcome", "annotations"):
        if isinstance(d.get(k), str):
            d[k] = json.loads(d[k])
    d = {k: (None if (isinstance(v, float) and v != v) else v) for k, v in d.items()}
    return Record.model_validate(d)
