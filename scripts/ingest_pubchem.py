"""Ingest PubChem BioAssay results (measurement) + PubChem computed properties.

PubChem data is public domain per NCBI policy (depositor terms may apply).
- concise assay tables: active/inactive + activity value -> measurement (cell or biochemical)
- XLogP3 (a fitted predictive model) -> ml_prediction
- MolecularWeight, TPSA (deterministic from structure) -> compute
"""
import csv
import io
import sys
import time
from pathlib import Path

import requests
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, "src")
from labpt.annotate import annotate
from labpt.io import write_jsonl
from labpt.schema import Outcome, Record, make_id

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
LIC = "PubChem-public-domain (NCBI policy; depositor terms may apply)"
# dose-response confirmatory assays with potency values (small, well known)
AIDS = [1000, 588342, 720516, 1053197, 652065, 504466]
MAX_ROWS = 1500


def get(url):
    for _ in range(4):
        r = requests.get(url, timeout=120)
        if r.ok:
            return r
        time.sleep(3)
    return None


def scaffold(smi):
    m = Chem.MolFromSmiles(smi or "")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


out, cids = [], set()
for aid in AIDS:
    meta = get(f"{PUG}/assay/aid/{aid}/summary/JSON")
    if not meta:
        continue
    s = meta.json()["AssaySummaries"]["AssaySummary"][0]
    name = s.get("Name")
    # stream only the first MAX_ROWS lines (some assays have 300k+ rows)
    lines = []
    try:
        with requests.get(f"{PUG}/assay/aid/{aid}/concise/CSV", stream=True, timeout=120) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                lines.append(line)
                if len(lines) > MAX_ROWS:
                    break
    except requests.RequestException as e:
        print("skip", aid, e)
        continue
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    n = 0
    for row in rows:
        cid = row.get("CID")
        if not cid:
            continue
        outcome = row.get("Activity Outcome") or row.get("PUBCHEM_ACTIVITY_OUTCOME")
        val = row.get("Activity Value [uM]") or row.get("PUBCHEM_ACTIVITY_SCORE")
        try:
            val = float(val) if val not in (None, "") else None
        except ValueError:
            val = None
        cids.add(int(cid))
        out.append(Record(
            id=make_id("pubchem", aid, row.get("SID"), cid),
            source="pubchem_bioassay", source_id=f"AID{aid}/SID{row.get('SID')}",
            source_url=f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
            license=LIC, modality="bioassay", provenance="measurement",
            method=s.get("SourceName"), property=row.get("Activity Name") or "activity",
            value=val, units="uM" if "Activity Value [uM]" in row and val is not None else None,
            entity={"cid": int(cid), "aid": aid, "assay_name": name},
            conditions={"protocol_hint": (s.get("Protocol") or [""])[0][:300] if s.get("Protocol") else None},
            outcome=Outcome(success={"Active": True, "Inactive": False}.get(outcome), metric="activity_outcome"),
            group_key=f"cid|{cid}",
        ))
        n += 1
    print("AID", aid, n)

# properties for the compounds seen
cid_list = sorted(cids)[:3000]
props = {}
for i in range(0, len(cid_list), 200):
    r = get(f"{PUG}/compound/cid/{','.join(map(str, cid_list[i:i + 200]))}/property/SMILES,MolecularWeight,XLogP,TPSA/JSON")
    if r:
        for p in r.json()["PropertyTable"]["Properties"]:
            props[p["CID"]] = p
for rec in out:
    p = props.get(rec.entity["cid"])
    if p:
        rec.entity["smiles"] = p.get("SMILES")
        rec.group_key = f"scaffold|{scaffold(p.get('SMILES'))}"
for cid, p in props.items():
    for key, prov, method, units in [
        ("XLogP", "ml_prediction", "XLogP3 (fitted predictive model, PubChem)", None),
        ("MolecularWeight", "compute", "computed from structure (PubChem)", "g/mol"),
        ("TPSA", "compute", "Cactvs TPSA (PubChem)", "A^2"),
    ]:
        if p.get(key) is None:
            continue
        out.append(Record(
            id=make_id("pubchem", cid, key), source="pubchem_compound", source_id=f"CID{cid}#{key}",
            source_url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}", license=LIC,
            modality="molecular_property", provenance=prov, method=method, property=key,
            value=float(p[key]), units=units, entity={"cid": cid, "smiles": p.get("SMILES")},
            group_key=f"scaffold|{scaffold(p.get('SMILES'))}",
        ))
out = [annotate(r) for r in out]
print("pubchem records:", write_jsonl(out, Path("data/interim/pubchem.jsonl")))
