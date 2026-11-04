"""Ingest Open Reaction Database (ORD) reactions (CC-BY-SA-4.0).

Downloads a few dataset parquet files from github.com/open-reaction-database/ord-data,
decodes each Reaction protobuf, and keeps: inputs (SMILES + roles), conditions,
procedure text -> structured steps, product yield (measurement).
"""
import random
import sys
from pathlib import Path

import pandas as pd
import requests
from ord_schema.proto import reaction_pb2

sys.path.insert(0, "src")
from labpt.annotate import annotate
from labpt.io import write_jsonl
from labpt.protocol_parser import parse_procedure
from labpt.schema import Outcome, Record, make_id

MAX_DATASETS = 25
MAX_PER_DATASET = 300
RAW = Path("data/raw/ord")
RAW.mkdir(parents=True, exist_ok=True)
random.seed(0)

tree = requests.get("https://api.github.com/repos/open-reaction-database/ord-data/git/trees/main?recursive=1", timeout=60).json()
paths = sorted(t["path"] for t in tree["tree"] if t["path"].endswith(".parquet"))
random.shuffle(paths)

ROLE = reaction_pb2.ReactionRole.ReactionRoleType


def smiles_of(comp):
    for ident in comp.identifiers:
        if ident.type == ident.SMILES:
            return ident.value
    for ident in comp.identifiers:
        if ident.type == ident.NAME:
            return ident.value
    return None


out = []
used = 0
for p in paths:
    if used >= MAX_DATASETS:
        break
    local = RAW / Path(p).name
    if not local.exists():
        resp = requests.get(f"https://media.githubusercontent.com/media/open-reaction-database/ord-data/main/{p}", timeout=120)
        if resp.status_code != 200 or len(resp.content) > 60_000_000:
            continue
        local.write_bytes(resp.content)
    df = pd.read_parquet(local)
    ds_id = Path(p).stem
    kept = 0
    rows = df.sample(frac=1, random_state=0)
    for rid, blob in zip(rows["reaction_id"], rows["reaction"]):
        if kept >= MAX_PER_DATASET:
            break
        rx = reaction_pb2.Reaction.FromString(blob)
        reactants, reagents, solvents = [], [], []
        for key, inp in rx.inputs.items():
            for c in inp.components:
                s = smiles_of(c)
                if not s:
                    continue
                if c.reaction_role == ROLE.SOLVENT:
                    solvents.append(s)
                elif c.reaction_role in (ROLE.REAGENT, ROLE.CATALYST):
                    reagents.append(s)
                else:
                    reactants.append(s)
        products, yld = [], None
        analyses = []
        for o in rx.outcomes:
            for an_key, an in o.analyses.items():
                analyses.append(reaction_pb2.Analysis.AnalysisType.Name(an.type))
            for pr in o.products:
                s = smiles_of(pr)
                if s:
                    products.append(s)
                for m in pr.measurements:
                    if m.type == m.YIELD and m.HasField("percentage") and yld is None:
                        yld = m.percentage.value
        if yld is None or not products:
            continue
        cond = {}
        if rx.conditions.temperature.HasField("setpoint"):
            sp = rx.conditions.temperature.setpoint
            cond["temperature_C"] = round(sp.value if sp.units == sp.CELSIUS else sp.value - 273.15, 2) if sp.units in (sp.CELSIUS, sp.KELVIN) else None
        if rx.conditions.HasField("reflux"):
            cond["reflux"] = rx.conditions.reflux
        for o in rx.outcomes[:1]:
            if o.HasField("reaction_time"):
                cond["reaction_time"] = f"{o.reaction_time.value:g} {reaction_pb2.Time.TimeUnit.Name(o.reaction_time.units).lower()}"
        cond["solvents"] = solvents
        rtype = [i.value for i in rx.identifiers if i.type == i.REACTION_TYPE or "type" in (i.details or "").lower()]
        proc = rx.notes.procedure_details or None
        yld = max(0.0, min(float(yld), 100.0))
        r = Record(
            id=make_id("ord", rid),
            source="ord",
            source_id=rid,
            source_url=f"https://open-reaction-database.org/id/{rid}",
            license="CC-BY-SA-4.0",
            modality="reaction",
            provenance="measurement",
            method=", ".join(sorted(set(analyses))) or "reported isolated/assay yield",
            property="yield",
            value=round(yld, 2),
            units="%",
            entity={"reactants": reactants, "reagents": reagents, "products": products,
                    "reaction_type": rtype[0] if rtype else None, "ord_dataset": ds_id},
            conditions=cond,
            protocol_text=proc,
            protocol_steps=parse_procedure(proc) if proc else [],
            outcome=Outcome(success=yld >= 20, metric="yield", value=round(yld, 2), units="%"),
            group_key=f"ord|{ds_id}",
        )
        out.append(annotate(r))
        kept += 1
    if kept:
        used += 1
        print(ds_id, kept)

print("ord records:", write_jsonl(out, Path("data/interim/ord.jsonl")))
