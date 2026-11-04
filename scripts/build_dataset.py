"""Merge interim/*.jsonl -> validate -> dedup -> split -> data/v0 (parquet + jsonl sample + datacard)."""
import collections
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from labpt.io import to_row
from labpt.schema import Record, json_schema
from labpt.splits import assign_splits, split_of

# default physical scale per modality (a source can override when ingesting)
SCALE = {"robot_episode": "lab-process", "reaction": "molecular", "bioassay": "molecular",
         "molecular_property": "molecular", "run_log": "lab-process", "lab_note": "lab-process"}
CELL_SOURCES = {"pubchem_bioassay"}  # cell-based HTS screens

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--interim', type=Path, default=Path('data/interim'))
parser.add_argument('--out', type=Path, default=Path('data/v0'))
parser.add_argument('--split-mode', '--split_mode', choices=['group', 'time'], default='group')
parser.add_argument('--val-year', type=int)
parser.add_argument('--test-year', type=int)
parser.add_argument('--prospective-source', action='append', default=[])
parser.add_argument('--prospective-id-file', type=Path, help='one record id per line; reserves entire groups')
parser.add_argument('--prospective-fraction', type=float, default=0.0)
args = parser.parse_args()
INTERIM, OUT = args.interim, args.out
OUT.mkdir(parents=True, exist_ok=True)

records, bad = [], 0
for f in sorted(INTERIM.glob("*.jsonl")):
    if f.name == "llm_annotations.jsonl":
        continue
    for line in open(f):
        try:
            r = Record.model_validate_json(line)
        except Exception as e:  # validation failure -> count and drop
            bad += 1
            continue
        if r.physical_scale == "molecular" and r.modality in SCALE:
            r.physical_scale = "cell" if r.source in CELL_SOURCES else SCALE[r.modality]
        records.append(r)
print("loaded", len(records), "invalid", bad)

# merge optional LLM annotations (codex CLI), flagged auto=True
from labpt.schema import Annotation
llm_file = INTERIM / "llm_annotations.jsonl"
if llm_file.exists():
    llm = {}
    for line in open(llm_file):
        x = json.loads(line)
        llm[x["id"]] = x
    for r in records:
        x = llm.get(r.id)
        if x:
            for key in ("safety_level", "failure_reason"):
                if x.get(key) is not None:
                    r.annotations.append(Annotation(key=key, value=x[key], annotator=x["annotator"], auto=True))
            if x.get("step_types"):
                r.annotations.append(Annotation(key="note", value={"llm_step_types": x["step_types"]}, annotator=x["annotator"], auto=True))

# dedup on content key (same source+entity+property+value+conditions)
seen, uniq = set(), []
for r in records:
    k = r.dedup_key()
    if k in seen:
        continue
    seen.add(k)
    uniq.append(r)
print("after dedup", len(uniq))


policy = assign_splits(uniq, args.split_mode, args.val_year, args.test_year,
                       prospective_sources=args.prospective_source,
                       prospective_ids=args.prospective_id_file.read_text().splitlines() if args.prospective_id_file else [],
                       prospective_fraction=args.prospective_fraction)

rows = [to_row(r) for r in uniq]
df = pd.DataFrame(rows)
df.to_parquet(OUT / "records.parquet", index=False)
for s in ("train", "val", "test", "prospective_blind"):
    df[df.split == s].to_parquet(OUT / f"{s}.parquet", index=False)

# small browsable sample: 20 per source
with open(OUT / "sample.jsonl", "w") as f:
    for src, g in df[df.split != "prospective_blind"].groupby("source"):
        for _, row in g.head(20).iterrows():
            r = next(x for x in uniq if x.id == row["id"])
            f.write(r.model_dump_json() + "\n")

json.dump(json_schema(), open(OUT / "schema.json", "w"), indent=1)


def counts(col):
    return {str(k): int(v) for k, v in df[col].value_counts().items()}


card = {
    "version": "v0",
    "split_policy": policy,
    "n_records": len(df),
    "n_invalid_dropped": bad,
    "n_duplicates_dropped": len(records) - len(uniq),
    "by_source": counts("source"),
    "by_provenance": counts("provenance"),
    "by_license": counts("license"),
    "by_modality": counts("modality"),
    "by_physical_scale": counts("physical_scale"),
    "by_split": counts("split"),
    "source_x_provenance": {f"{a}|{b}": int(n) for (a, b), n in df.groupby(["source", "provenance"]).size().items()},
    "with_protocol_steps": int((df.protocol_steps != "[]").sum()),
    "annotators": dict(collections.Counter(a["annotator"] for r in uniq for a in [x.model_dump() for x in r.annotations])),
}
# group leakage check
g = df.groupby("group_key").split.nunique()
card["groups_in_multiple_splits"] = int((g > 1).sum())
json.dump(card, open(OUT / "datacard.json", "w"), indent=1)
print(json.dumps(card, indent=1))
