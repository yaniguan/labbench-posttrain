"""Ingest PubChem BioAssay results (measurement) + PubChem computed properties.

PubChem data is public domain per NCBI policy (depositor terms may apply).
- concise assay tables: active/inactive + activity value -> measurement (cell or biochemical)
- XLogP3 (a fitted predictive model) -> ml_prediction
- MolecularWeight, TPSA (deterministic from structure) -> compute
"""
from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path
from typing import Iterable

import requests

sys.path.insert(0, "src")
from labpt.annotate import annotate
from labpt.io import write_jsonl
from labpt.schema import Outcome, Record, make_id
from labpt.splits import canonical_scaffold_group_key

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
LIC = "PubChem-public-domain (NCBI policy; depositor terms may apply)"
# dose-response confirmatory assays with potency values (small, well known)
AIDS = [1000, 588342, 720516, 1053197, 652065, 504466]
MAX_ROWS = 1500
# Throttled below PubChem PUG-REST 5 requests/second limit (#46)
MIN_REQUEST_INTERVAL_S = 0.25


def get(url: str, *, min_interval_s: float = MIN_REQUEST_INTERVAL_S):
    for attempt in range(4):
        time.sleep(min_interval_s)
        r = requests.get(url, timeout=120)
        if r.ok:
            return r
        time.sleep(2.0 * (attempt + 1))
    return None


def post_form(url: str, data: dict[str, str], *, min_interval_s: float = MIN_REQUEST_INTERVAL_S):
    for attempt in range(4):
        time.sleep(min_interval_s)
        r = requests.post(url, data=data, timeout=120)
        if r.ok:
            return r
        time.sleep(2.0 * (attempt + 1))
    return None


def extract_smiles(prop_row: dict) -> str | None:
    for key in ("SMILES", "CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES"):
        val = prop_row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def fetch_cid_properties(
    cids: Iterable[int],
    *,
    batch_size: int = 200,
    min_interval_s: float = MIN_REQUEST_INTERVAL_S,
    properties: str = "SMILES,ConnectivitySMILES,MolecularWeight,XLogP,TPSA",
) -> dict[int, dict]:
    """Batch-fetch PubChem compound properties for all CIDs without truncation (< 5 req/s, #46)."""
    cid_list = sorted({int(c) for c in cids})
    props: dict[int, dict] = {}
    url = f"{PUG}/compound/cid/property/{properties}/JSON"
    for i in range(0, len(cid_list), batch_size):
        batch = cid_list[i : i + batch_size]
        r = post_form(url, {"cid": ",".join(map(str, batch))}, min_interval_s=min_interval_s)
        if r is None:
            r = get(
                f"{PUG}/compound/cid/{','.join(map(str, batch))}/property/{properties}/JSON",
                min_interval_s=min_interval_s,
            )
        if r is not None:
            for p in r.json().get("PropertyTable", {}).get("Properties", []):
                smi = extract_smiles(p)
                if smi and not p.get("SMILES"):
                    p["SMILES"] = smi
                props[int(p["CID"])] = p
    return props


def scaffold(smi: str | None) -> str:
    return canonical_scaffold_group_key(smi)


def main() -> None:
    out: list[Record] = []
    cids: set[int] = set()
    for aid in AIDS:
        meta = get(f"{PUG}/assay/aid/{aid}/summary/JSON")
        if not meta:
            continue
        s = meta.json()["AssaySummaries"]["AssaySummary"][0]
        name = s.get("Name")
        # stream only the first MAX_ROWS lines (some assays have 300k+ rows)
        lines = []
        try:
            time.sleep(MIN_REQUEST_INTERVAL_S)
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
            out.append(
                Record(
                    id=make_id("pubchem", aid, row.get("SID"), cid),
                    source="pubchem_bioassay",
                    source_id=f"AID{aid}/SID{row.get('SID')}",
                    source_url=f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
                    license=LIC,
                    modality="bioassay",
                    provenance="measurement",
                    method=s.get("SourceName"),
                    property=row.get("Activity Name") or "activity",
                    value=val,
                    units="uM" if "Activity Value [uM]" in row and val is not None else None,
                    entity={"cid": int(cid), "aid": aid, "assay_name": name},
                    conditions={"protocol_hint": (s.get("Protocol") or [""])[0][:300] if s.get("Protocol") else None},
                    outcome=Outcome(success={"Active": True, "Inactive": False}.get(outcome), metric="activity_outcome"),
                    group_key=f"cid|{cid}",
                )
            )
            n += 1
        print("AID", aid, n)

    # Fetch properties for ALL compounds seen (throttled batches of 200, < 5 req/s, #46)
    props = fetch_cid_properties(cids, batch_size=200, min_interval_s=MIN_REQUEST_INTERVAL_S)
    for rec in out:
        p = props.get(rec.entity["cid"])
        smi = extract_smiles(p) if p else None
        if smi:
            rec.entity["smiles"] = smi
            rec.group_key = canonical_scaffold_group_key(smi)
    for cid, p in props.items():
        smi = extract_smiles(p)
        for key, prov, method, units in [
            ("XLogP", "ml_prediction", "XLogP3 (fitted predictive model, PubChem)", None),
            ("MolecularWeight", "compute", "computed from structure (PubChem)", "g/mol"),
            ("TPSA", "compute", "Cactvs TPSA (PubChem)", "A^2"),
        ]:
            if p.get(key) is None:
                continue
            out.append(
                Record(
                    id=make_id("pubchem", cid, key),
                    source="pubchem_compound",
                    source_id=f"CID{cid}#{key}",
                    source_url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                    license=LIC,
                    modality="molecular_property",
                    provenance=prov,
                    method=method,
                    property=key,
                    value=float(p[key]),
                    units=units,
                    entity={"cid": cid, "smiles": smi},
                    group_key=canonical_scaffold_group_key(smi),
                )
            )
    out = [annotate(r) for r in out]
    print("pubchem records:", write_jsonl(out, Path("data/interim/pubchem.jsonl")))


if __name__ == "__main__":
    main()
