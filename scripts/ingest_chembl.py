"""Ingest ChEMBL (CC-BY-SA-3.0) bioactivity measurements + predicted/computed molecule properties.

- activities with a pChEMBL value  -> provenance=measurement
- molecule_properties.alogp / np_likeness_score (fitted statistical models) -> provenance=ml_prediction
- molecule_properties.full_mwt / psa (deterministic from structure) -> provenance=compute
"""
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

API = "https://www.ebi.ac.uk/chembl/api/data"
LIC = "CC-BY-SA-3.0"
# a few well-studied targets across families
TARGETS = ["CHEMBL203", "CHEMBL279", "CHEMBL240", "CHEMBL1862", "CHEMBL4005", "CHEMBL2971", "CHEMBL244", "CHEMBL217"]
PER_TARGET = 600


def get(url, **params):
    for _ in range(4):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(3)
    raise RuntimeError(url)


def scaffold(smi):
    m = Chem.MolFromSmiles(smi or "")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


out, mols = [], {}
for t in TARGETS:
    got, offset = 0, 0
    while got < PER_TARGET:
        d = get(f"{API}/activity.json", target_chembl_id=t, pchembl_value__isnull="false",
                limit=200, offset=offset, format="json")
        acts = d["activities"]
        if not acts:
            break
        for a in acts:
            smi = a.get("canonical_smiles")
            if not smi or a.get("standard_value") is None:
                continue
            mols[a["molecule_chembl_id"]] = smi
            pchembl = float(a["pchembl_value"])
            r = Record(
                id=make_id("chembl", a["activity_id"]),
                source="chembl",
                source_id=str(a["activity_id"]),
                source_url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{a['molecule_chembl_id']}/",
                license=LIC,
                modality="bioassay",
                provenance="measurement",
                method=f"{a.get('standard_type')} ({a.get('bao_label') or a.get('assay_type')})",
                instrument=None,
                property=a.get("standard_type"),
                value=float(a["standard_value"]),
                value_relation=a.get("standard_relation"),
                units=a.get("standard_units"),
                entity={"smiles": smi, "molecule_chembl_id": a["molecule_chembl_id"],
                        "target_chembl_id": t, "target": a.get("target_pref_name"),
                        "assay_chembl_id": a.get("assay_chembl_id"), "assay_description": a.get("assay_description")},
                conditions={"assay_type": a.get("assay_type"), "organism": a.get("target_organism"),
                            "pchembl_value": pchembl, "document_year": a.get("document_year")},
                outcome=Outcome(success=pchembl >= 6.0, metric="pchembl>=6 (active)", value=pchembl),
                group_key=f"scaffold|{scaffold(smi)}",
            )
            out.append(annotate(r))
            got += 1
        offset += 200
    print(t, got)

# molecule properties for a subset of molecules
ids = list(mols)[:1500]
for i in range(0, len(ids), 50):
    d = get(f"{API}/molecule.json", molecule_chembl_id__in=",".join(ids[i:i + 50]), limit=50, format="json")
    for m in d["molecules"]:
        mp = m.get("molecule_properties") or {}
        smi = mols.get(m["molecule_chembl_id"])
        for prop, prov, method, units in [
            ("alogp", "ml_prediction", "ALogP fitted atom-contribution model (ChEMBL pipeline)", None),
            ("np_likeness_score", "ml_prediction", "NP-likeness Bayesian model score (ChEMBL pipeline)", None),
            ("full_mwt", "compute", "molecular weight from structure (ChEMBL pipeline)", "Da"),
            ("psa", "compute", "topological polar surface area (ChEMBL pipeline)", "A^2"),
        ]:
            if mp.get(prop) is None:
                continue
            out.append(annotate(Record(
                id=make_id("chembl", m["molecule_chembl_id"], prop),
                source="chembl", source_id=f"{m['molecule_chembl_id']}#{prop}",
                source_url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{m['molecule_chembl_id']}/",
                license=LIC, modality="molecular_property", provenance=prov, method=method,
                property=prop, value=float(mp[prop]), units=units,
                entity={"smiles": smi, "molecule_chembl_id": m["molecule_chembl_id"]},
                group_key=f"scaffold|{scaffold(smi)}",
            )))

print("chembl records:", write_jsonl(out, Path("data/interim/chembl.jsonl")))
