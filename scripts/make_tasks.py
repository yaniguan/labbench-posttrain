"""Build task examples + preference pairs from data/v0/records.parquet -> data/v0/tasks/*.jsonl"""
import collections
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from labpt.io import from_row
from labpt.tasks import TASKS, build_examples, make_pairs

MAX_PER_TASK_SPLIT = {"train": 4000, "val": 400, "test": 400}  # keep files small for git

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=Path, default=Path('data/v0'))
args = parser.parse_args()
OUT = args.data_dir / 'tasks'
OUT.mkdir(parents=True, exist_ok=True)
# Blind answers never enter routine task generation or preference pairing.
recs = [from_row(r) for r in pd.read_parquet(args.data_dir / 'records.parquet').to_dict('records')
        if r['split'] != 'prospective_blind']
by_id = {r.id: r for r in recs}
random.seed(0)
exs = build_examples(recs, list(TASKS))
random.shuffle(exs)
kept, cnt = [], collections.Counter()
for e in exs:
    k = (e["task"], e["split"])
    if cnt[k] < MAX_PER_TASK_SPLIT[e["split"]]:
        cnt[k] += 1
        kept.append(e)
pairs = make_pairs(kept, by_id)
for split in ("train", "val", "test"):
    with open(OUT / f"sft_{split}.jsonl", "w") as f:
        for e in kept:
            if e["split"] == split:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(OUT / f"pairs_{split}.jsonl", "w") as f:
        for p in pairs:
            if p["split"] == split:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
stats = {"examples": {f"{t}|{s}": n for (t, s), n in sorted(cnt.items())},
         "pairs": dict(collections.Counter(f"{p['kind']}|{p['split']}" for p in pairs))}
json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
