"""Load non-text scientific data from data/v0/records.parquet as arrays (molecules, spectra, images) + paired text."""
import json

import numpy as np
import pandas as pd

from encoders import hash_text, morgan, norm_curve


def load(path="data/v0/records.parquet"):
    return pd.read_parquet(path)


def molecules(df, prop_list=("gap", "homo", "lumo", "mu", "u0"), source="qm9"):
    """QM9 molecules with multi-target computed labels (one row per molecule)."""
    d = df[df.source == source].copy()
    d["smiles"] = d.entity.map(lambda e: json.loads(e).get("smiles"))
    piv = d.pivot_table(index=["smiles", "split"], columns="property", values="value").reset_index().dropna()
    X = np.stack([morgan(s) for s in piv.smiles])
    Y = piv[list(prop_list)].to_numpy(np.float32)
    return X, Y, piv.split.to_numpy(), piv.smiles.tolist()


def spectra(df):
    """All curves (XRD, HPLC, IR, growth curves) with a text description for alignment."""
    X, T, S, K = [], [], [], []
    for _, r in df[df.modality.isin(["spectrum", "table"])].iterrows():
        c, e, o = json.loads(r.conditions), json.loads(r.entity), json.loads(r.outcome)
        y = c.get("y") or c.get("od600")
        if not y or len(y) < 8:
            continue
        X.append(norm_curve(y))
        T.append(f"{r.method}. {r.property}. {o.get('metric') or ''} {json.dumps(e)[:200]}")
        S.append(r.split)
        K.append(r.source)
    return np.stack(X), T, np.array(S), K


def molecule_text_pairs(df, max_n=6000):
    """Molecule <-> text from ChEMBL/PubChem (assay descriptions, targets, property statements)."""
    X, T, S = [], [], []
    for _, r in df[df.source.isin(["chembl", "pubchem_bioassay"])].head(max_n).iterrows():
        e = json.loads(r.entity)
        if not e.get("smiles"):
            continue
        X.append(morgan(e["smiles"]))
        T.append(f"{e.get('target') or ''} {e.get('assay_description') or e.get('assay_name') or ''} {r.property}")
        S.append(r.split)
    return np.stack(X), T, np.array(S)


def images(df):
    d = df[df.modality == "image"]
    X = np.stack([np.array(json.loads(c)["pixels_32x32_ch0"], np.float32) for c in d.conditions])
    X = (X - X.mean()) / (X.std() + 1e-6)
    return X, d.value.to_numpy(np.float32), d.split.to_numpy()


def texts_to_array(texts):
    return np.stack([hash_text(t) for t in texts])
