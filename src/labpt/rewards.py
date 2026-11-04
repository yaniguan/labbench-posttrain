"""Verifiable rewards for GRPO and eval. All are cheap (tier-1/2) checks."""
from __future__ import annotations

import json
import re

from .schema import ACTION_VOCAB

UNITS_OK = {
    "yield": {"%"}, "gap": {"Ha", "eV"}, "homo": {"Ha", "eV"}, "lumo": {"Ha", "eV"}, "u0": {"Ha", "eV"},
    "mu": {"D"}, "IC50": {"nM", "uM"}, "Ki": {"nM", "uM"}, "EC50": {"nM", "uM"}, "Kd": {"nM", "uM"},
    "full_mwt": {"Da", "g/mol"}, "MolecularWeight": {"g/mol", "Da"}, "TPSA": {"A^2"}, "psa": {"A^2"},
}
MAX_TEMP_C = 250  # robot / hotplate limit
MIN_TEMP_C = -100


def parse_json(text: str):
    text = text.strip()
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def schema_valid_actions(text: str) -> float:
    """1.0 if a non-empty list of known actions with dict params and physical limits respected."""
    obj = parse_json(text)
    if not isinstance(obj, list) or not obj:  # empty protocol gets 0: blocks reward hacking
        return 0.0
    for a in obj:
        if not isinstance(a, dict) or a.get("action") not in ACTION_VOCAB:
            return 0.0
        p = a.get("params", {})
        if not isinstance(p, dict):
            return 0.0
        t = p.get("temperature_C")
        if isinstance(t, (int, float)) and not (MIN_TEMP_C <= t <= MAX_TEMP_C):
            return 0.0
    return 1.0


def action_overlap(text: str, reference: str) -> float:
    """F1 over action verbs (order-insensitive) vs reference. Key-step coverage."""
    obj, ref = parse_json(text), parse_json(reference)
    if not isinstance(obj, list) or not isinstance(ref, list) or not ref:
        return 0.0
    a = [x.get("action") for x in obj if isinstance(x, dict)]
    b = [x.get("action") for x in ref]
    common = sum(min(a.count(v), b.count(v)) for v in set(b))
    if common == 0:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def provenance_correct(text: str, gold: str) -> float:
    obj = parse_json(text)
    return 1.0 if isinstance(obj, dict) and obj.get("provenance") == gold else 0.0


def unit_consistent(text: str, prop: str | None, gold_units: str | None) -> float:
    obj = parse_json(text)
    if not isinstance(obj, dict):
        return 0.0
    u = obj.get("units")
    if prop in UNITS_OK:
        return 1.0 if u in UNITS_OK[prop] else 0.0
    return 1.0 if u == gold_units else 0.0


def total_reward(text: str, task: str, answer: str, meta: dict) -> float:
    if task == "protocol2actions":
        return schema_valid_actions(text) + action_overlap(text, answer)
    if task == "provenance":
        return provenance_correct(text, meta["provenance"]) + 0.5 * unit_consistent(text, meta.get("property"), meta.get("units"))
    gold, obj = parse_json(answer), parse_json(text)
    if not isinstance(obj, dict):
        return 0.0
    return sum(1.0 for k, v in gold.items() if obj.get(k) == v) / max(len(gold), 1)
