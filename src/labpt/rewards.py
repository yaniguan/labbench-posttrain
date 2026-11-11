"""Verifiable rewards for GRPO and eval. All are cheap (tier-1/2) checks."""
from __future__ import annotations

from itertools import combinations
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


def _param_match_score(pred_params: dict, ref_params: dict, rel_tol: float = 0.05) -> float:
    if not ref_params:
        return 1.0 if not pred_params else 0.8
    hits = 0.0
    for k, rv in ref_params.items():
        if k not in pred_params:
            continue
        pv = pred_params[k]
        if isinstance(rv, (int, float)) and not isinstance(rv, bool) and isinstance(pv, (int, float)) and not isinstance(pv, bool):
            denom = max(abs(float(rv)), 1.0)
            err = abs(float(pv) - float(rv)) / denom
            hits += 1.0 if err <= rel_tol else max(0.0, 1.0 - err)
        elif isinstance(rv, list) and isinstance(pv, list) and len(rv) == len(pv) and rv:
            if all(isinstance(x, (int, float)) for x in rv + pv):
                diffs = [abs(float(a) - float(b)) / max(abs(float(b)), 1.0) for a, b in zip(pv, rv)]
                mean_err = sum(diffs) / len(diffs)
                hits += 1.0 if mean_err <= rel_tol else max(0.0, 1.0 - mean_err)
            else:
                hits += float(pv == rv)
        else:
            hits += float(str(pv).strip().lower() == str(rv).strip().lower())
    return hits / len(ref_params)


def _kendall_tau_indices(gold_idx: list[int], pred_idx: list[int]) -> float:
    if len(gold_idx) <= 1:
        return 1.0
    rg = {x: i for i, x in enumerate(gold_idx)}
    rp = {x: i for i, x in enumerate(pred_idx)}
    pairs = list(combinations(gold_idx, 2))
    if not pairs:
        return 1.0
    conc = sum((rg[a] - rg[b]) * (rp[a] - rp[b]) > 0 for a, b in pairs)
    return (2.0 * conc - len(pairs)) / len(pairs)


def ordered_action_reward(text: str, reference: str, *, rel_tol: float = 0.05) -> dict[str, float]:
    """Order-sensitive protocol2actions verifier with multi-channel GDPO components.

    Replaces bag-of-verbs `action_overlap` when `reward_version='v2'` is used:
    - `schema_valid`: 1.0 if all actions obey `ACTION_VOCAB` and physical temperature bounds
    - `action_f1`: one-to-one step alignment F1 (penalizes omissions AND duplicate padding)
    - `order_tau`: rescaled Kendall tau in [0, 1] over matched steps (`aspirate` before `dispense`, etc.)
    - `param_precision`: parameter fidelity (`volume_uL`, `temperature_C`, `well`, `labware`, `duration_s`)
    """
    valid = schema_valid_actions(text)
    obj, ref = parse_json(text), parse_json(reference)
    zero = {
        "schema_valid": valid,
        "action_f1": 0.0,
        "order_tau": 0.0,
        "param_precision": 0.0,
        "total": 0.0,
        "reward/schema": valid,
        "reward/action_f1": 0.0,
        "reward/order": 0.0,
        "reward/params": 0.0,
    }
    if valid == 0.0 or not isinstance(obj, list) or not isinstance(ref, list) or not ref:
        return zero
    pred_steps = [x for x in obj if isinstance(x, dict)]
    ref_steps = [x for x in ref if isinstance(x, dict)]
    if not pred_steps or not ref_steps:
        return zero

    # Greedy / Hungarian one-to-one alignment favouring same verb + parameter similarity + positional proximity
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n_r, n_p = len(ref_steps), len(pred_steps)
    score_mat = np.zeros((n_r, n_p), dtype=np.float64)
    param_mat = np.zeros((n_r, n_p), dtype=np.float64)
    for ri, r_step in enumerate(ref_steps):
        for pi, p_step in enumerate(pred_steps):
            if r_step.get("action") == p_step.get("action"):
                ps = _param_match_score(p_step.get("params", {}) or {}, r_step.get("params", {}) or {}, rel_tol=rel_tol)
                pos_bonus = 1e-3 * (1.0 - abs(ri / max(n_r, 1) - pi / max(n_p, 1)))
                param_mat[ri, pi] = ps
                score_mat[ri, pi] = 0.5 + 0.5 * ps + pos_bonus

    r_ind, p_ind = linear_sum_assignment(-score_mat)
    matched = [(int(r), int(p)) for r, p in zip(r_ind, p_ind) if score_mat[r, p] >= 0.5]
    if not matched:
        return zero

    tp = float(len(matched))
    prec, rec = tp / n_p, tp / n_r
    action_f1 = 2.0 * prec * rec / (prec + rec)

    if len(matched) > 1:
        by_ref = [r for r, _ in sorted(matched, key=lambda x: x[0])]
        by_pred = [r for r, _ in sorted(matched, key=lambda x: x[1])]
        order_tau = (_kendall_tau_indices(by_ref, by_pred) + 1.0) / 2.0
    elif n_r == 1:
        order_tau = 1.0
    else:
        order_tau = 0.0

    param_precision = sum(float(param_mat[r, p]) for r, p in matched) / n_r
    total = valid * (0.4 * action_f1 + 0.3 * order_tau + 0.3 * param_precision)
    return {
        "schema_valid": valid,
        "action_f1": action_f1,
        "order_tau": order_tau,
        "param_precision": param_precision,
        "total": total,
        "reward/schema": valid,
        "reward/action_f1": action_f1,
        "reward/order": order_tau,
        "reward/params": param_precision,
    }


def counterfactual_consistency_reward(score_orig: float, score_counterfactual: float, *, threshold: float = 0.7) -> float:
    """1.0 when a policy succeeds on BOTH the original and perturbed counterfactual prompt;
    geometric mean when both are positive; 0.0 if it collapses on the counterfactual.
    """
    if score_orig <= 0.0 or score_counterfactual <= 0.0:
        return 0.0
    if score_orig >= threshold and score_counterfactual >= threshold:
        return 1.0
    return float((score_orig * score_counterfactual) ** 0.5)


def total_reward(text: str, task: str, answer: str, meta: dict, *, reward_version: str = "v1") -> float:
    if task == "protocol2actions":
        if reward_version == "v2":
            return 2.0 * ordered_action_reward(text, answer)["total"]
        return schema_valid_actions(text) + action_overlap(text, answer)
    if task == "provenance":
        return provenance_correct(text, meta["provenance"]) + 0.5 * unit_consistent(text, meta.get("property"), meta.get("units"))
    gold, obj = parse_json(answer), parse_json(text)
    if not isinstance(obj, dict):
        return 0.0
    return sum(1.0 for k, v in gold.items() if obj.get(k) == v) / max(len(gold), 1)
