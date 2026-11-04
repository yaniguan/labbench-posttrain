"""Ingest a sample of QM9 (DFT B3LYP/6-31G(2df,p)) computed properties. License CC0-1.0
(Ramakrishnan et al. 2014, figshare). Mirror: DeepChem S3 qm9.csv."""
import sys
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, "src")
from labpt.annotate import annotate
from labpt.io import write_jsonl
from labpt.schema import Record, make_id

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
raw = Path("data/raw/qm9.csv")
raw.parent.mkdir(parents=True, exist_ok=True)
if not raw.exists():
    raw.write_bytes(requests.get(URL, timeout=300).content)
df = pd.read_csv(raw).sample(n=1000, random_state=0)
PROPS = {"gap": "Ha", "homo": "Ha", "lumo": "Ha", "mu": "D", "u0": "Ha"}
out = []
for _, row in df.iterrows():
    smi = row["smiles"]
    m = Chem.MolFromSmiles(smi)
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""
    for p, u in PROPS.items():
        out.append(annotate(Record(
            id=make_id("qm9", row["mol_id"], p), source="qm9", source_id=f"{row['mol_id']}#{p}",
            source_url="https://doi.org/10.6084/m9.figshare.c.978904.v5", license="CC0-1.0",
            modality="molecular_property", provenance="compute", physical_scale="atomic",
            method="DFT B3LYP/6-31G(2df,p)", property=p, value=float(row[p]), units=u,
            entity={"smiles": smi}, group_key=f"scaffold|{scaf}",
        )))
print("qm9 records:", write_jsonl(out, Path("data/interim/qm9.jsonl")))
