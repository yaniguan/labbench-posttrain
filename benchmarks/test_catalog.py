import csv
from pathlib import Path

CATALOG = Path(__file__).parent / "catalog.csv"

MODALITIES = {"text/protocol", "robot action/video", "image", "spectra", "3D structure/trajectory",
              "tabular assay", "time-series instrument log", "graph", "multimodal"}
SOURCES = {"experimental measurement", "first-principles compute", "ML prediction", "human annotation", "mixed"}
TRACKS = {"scientist", "operator", "both"}
MESS = {"raw", "semi-clean", "clean"}


def load_rows():
    with open(CATALOG) as f:
        return list(csv.DictReader(f))


def test_has_many_rows():
    assert len(load_rows()) >= 150


def test_names_are_unique():
    names = [r["name"] for r in load_rows()]
    assert len(names) == len(set(names))


def test_values_are_allowed():
    for r in load_rows():
        assert r["modality"] in MODALITIES, r["name"]
        assert r["source"] in SOURCES, r["name"]
        assert r["track_fit"] in TRACKS, r["name"]
        assert r["messiness"] in MESS, r["name"]


def test_links_look_like_urls():
    for r in load_rows():
        assert r["link"].startswith("http"), r["name"]
