import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, "src")
from labpt.io import from_row
from labpt.schema import Action, Record

V0 = Path("data/v0")


def test_sample_records_validate():
    for line in open(V0 / "sample.jsonl"):
        Record.model_validate_json(line)


def test_every_record_has_provenance_and_license():
    df = pd.read_parquet(V0 / "records.parquet")
    assert len(df) > 1000
    assert set(df.provenance) <= {"measurement", "compute", "ml_prediction", "human_annotation"}
    assert df.license.notna().all() and (df.license.str.len() > 0).all()
    assert df.split.isin(["train", "val", "test"]).all()


def test_no_group_leakage_across_splits():
    df = pd.read_parquet(V0 / "records.parquet", columns=["group_key", "split"])
    assert (df.groupby("group_key").split.nunique() <= 1).all()


def test_ids_unique():
    df = pd.read_parquet(V0 / "records.parquet", columns=["id"])
    assert df.id.is_unique


def test_datacard_matches_data():
    card = json.load(open(V0 / "datacard.json"))
    df = pd.read_parquet(V0 / "records.parquet", columns=["source", "provenance"])
    assert card["n_records"] == len(df)
    assert card["by_provenance"] == {k: int(v) for k, v in df.provenance.value_counts().items()}


def test_roundtrip_row():
    df = pd.read_parquet(V0 / "records.parquet").head(50)
    for row in df.to_dict("records"):
        from_row(row)


def test_unknown_action_rejected():
    with pytest.raises(Exception):
        Action(action="teleport")
