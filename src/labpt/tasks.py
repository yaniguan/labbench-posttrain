"""Turn unified records into chat-style training examples.

Each example: {"task", "split", "prompt" (list of chat messages), "answer" (str), "meta"}.
Tasks:
  protocol2actions  free-text procedure / robot instruction -> JSON list of actions
  provenance        describe a number -> measurement | compute | ml_prediction
  outcome           reaction/assay context -> success true/false
  safety_failure    procedure -> {"safety_level", "failure_reason"}
  property_qa       molecule + property -> value with units (scientist track)
"""
from __future__ import annotations

import json
import random

from .schema import Record

SYSTEM = "You are a careful lab-automation assistant. Answer with JSON only."


def _msg(user: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _ann(r: Record, key: str):
    for a in r.annotations:
        if a.key == key:
            return a.value
    return None


def protocol2actions(r: Record):
    if not r.protocol_text or not r.protocol_steps:
        return None
    kind = "robot instruction" if r.modality == "robot_episode" else "lab procedure"
    user = (f"Convert this {kind} into a JSON list of structured actions. "
            f"Each item: {{\"action\": <verb>, \"params\": {{...}}}}.\n\n{r.protocol_text[:1500]}")
    ans = json.dumps([a.model_dump() for a in r.protocol_steps], ensure_ascii=False)
    return {"prompt": _msg(user), "answer": ans}


def describe_number(r: Record) -> str:
    # NOTE: never print the source name; the model must infer from method/context
    parts = [f"property: {r.property}", f"value: {r.value}"]
    if r.units:
        parts.append(f"units: {r.units}")
    if r.method:
        parts.append(f"method: {r.method}")
    if r.instrument:
        parts.append(f"instrument: {r.instrument}")
    smi = r.entity.get("smiles")
    if smi:
        parts.append(f"molecule: {smi}")
    return "\n".join(parts)


def provenance(r: Record):
    if r.value is None:
        return None
    user = ("Where does this number come from? Answer {\"provenance\": one of "
            "measurement|compute|ml_prediction|human_annotation, \"units\": <units or null>}.\n\n" + describe_number(r))
    return {"prompt": _msg(user), "answer": json.dumps({"provenance": r.provenance, "units": r.units})}


def outcome(r: Record):
    if r.outcome.success is None:
        return None
    if r.modality == "reaction":
        ctx = (f"reactants: {r.entity.get('reactants')}\nreagents: {r.entity.get('reagents')}\n"
               f"product: {r.entity.get('products')}\nconditions: {json.dumps(r.conditions)}")
        q = "Will this reaction give yield >= 20%?"
    elif r.modality == "bioassay":
        ctx = (f"molecule: {r.entity.get('smiles')}\nassay: {r.entity.get('assay_description') or r.entity.get('assay_name')}\n"
               f"target: {r.entity.get('target')}")
        q = "Will this compound be active in this assay?"
    else:
        return None
    return {"prompt": _msg(f"{q} Answer {{\"success\": true|false}}.\n\n{ctx}"),
            "answer": json.dumps({"success": bool(r.outcome.success)})}


def safety_failure(r: Record):
    lvl = _ann(r, "safety_level")
    if lvl is None or not r.protocol_text:
        return None
    fr = _ann(r, "failure_reason")
    ans = {"safety_level": lvl, "failure_reason": fr}
    user = ("Annotate this protocol. Answer {\"safety_level\": low|medium|high, "
            "\"failure_reason\": null or a short tag}.\n\n" + r.protocol_text[:1500]
            + (f"\nreported yield: {r.outcome.value}%" if r.outcome.value is not None else ""))
    return {"prompt": _msg(user), "answer": json.dumps(ans)}


def property_qa(r: Record):
    if r.modality != "molecular_property" or not r.entity.get("smiles") or r.value is None:
        return None
    user = (f"Molecule {r.entity['smiles']}. Give {r.property} and say where the value comes from. "
            "Answer {\"value\": number, \"units\": str|null, \"provenance\": str}.")
    return {"prompt": _msg(user), "answer": json.dumps({"value": round(r.value, 4), "units": r.units, "provenance": r.provenance})}


TASKS = {"protocol2actions": protocol2actions, "provenance": provenance, "outcome": outcome,
         "safety_failure": safety_failure, "property_qa": property_qa}


def build_examples(records: list[Record], tasks: list[str]):
    out = []
    for r in records:
        if r.split == "prospective_blind":
            continue
        for t in tasks:
            ex = TASKS[t](r)
            if ex:
                ex.update(task=t, split=r.split, meta={"id": r.id, "source": r.source, "provenance": r.provenance,
                                                      "units": r.units, "property": r.property})
                out.append(ex)
    return out


# ---------------- preference pairs (DPO / ORPO / reward model) -----------------
WRONG_PROV = {"measurement": "ml_prediction", "compute": "measurement", "ml_prediction": "measurement",
              "human_annotation": "compute"}


def corrupt_actions(answer: str, rng: random.Random) -> str:
    steps = json.loads(answer)
    if len(steps) > 2 and rng.random() < 0.5:
        steps.pop(rng.randrange(len(steps)))  # drop a key step
    else:
        rng.shuffle(steps)
        if len(steps) > 1 and json.dumps(steps) == answer:
            steps = steps[::-1]
        if len(steps) <= 1:
            steps = []  # empty protocol: executable but useless
    return json.dumps(steps, ensure_ascii=False)


def make_pairs(examples: list[dict], records_by_id: dict, seed: int = 0):
    rng = random.Random(seed)
    pairs = []
    for ex in examples:
        if ex["split"] == "prospective_blind":
            continue
        if ex["task"] == "provenance":
            a = json.loads(ex["answer"])
            bad = dict(a, provenance=WRONG_PROV[a["provenance"]])
            pairs.append({"prompt": ex["prompt"], "chosen": ex["answer"], "rejected": json.dumps(bad),
                          "kind": "wrong_provenance", "split": ex["split"]})
        elif ex["task"] == "protocol2actions":
            rej = corrupt_actions(ex["answer"], rng)
            if rej != ex["answer"]:
                pairs.append({"prompt": ex["prompt"], "chosen": ex["answer"], "rejected": rej,
                              "kind": "corrupted_actions", "split": ex["split"]})
    # successful vs failed real protocols within the same ORD dataset
    by_ds: dict[str, list[Record]] = {}
    for r in records_by_id.values():
        if r.split != "prospective_blind" and r.modality == "reaction" and r.protocol_text and r.outcome.value is not None:
            by_ds.setdefault((r.split, r.entity.get("ord_dataset")), []).append(r)
    for (split, ds), rs in by_ds.items():
        good = [r for r in rs if r.outcome.value >= 60]
        bad = [r for r in rs if r.outcome.value < 20]
        rng.shuffle(good)
        rng.shuffle(bad)
        for g, b in zip(good[:100], bad[:100]):
            prod = g.entity.get("products", [""])[0]
            prompt = _msg("Write a lab procedure likely to succeed (high yield) for this transformation. "
                          f"Target product: {prod}. Reaction family dataset: {ds}.")
            pairs.append({"prompt": prompt, "chosen": g.protocol_text[:1500], "rejected": b.protocol_text[:1500],
                          "kind": "success_vs_failure", "split": g.split,
                          "chosen_id": g.id, "rejected_id": b.id})
    # A-Lab: same element system, a run that reacted vs one that did not (real execution outcomes)
    by_tgt: dict[str, list[Record]] = {}
    for r in records_by_id.values():
        if r.split != "prospective_blind" and r.source == "alab_precursor_genome" and r.outcome.success is not None:
            by_tgt.setdefault((r.split, "-".join(sorted(r.entity.get("elements") or []))), []).append(r)
    for (split, tgt), rs in by_tgt.items():
        good = [r for r in rs if r.outcome.success]
        bad = [r for r in rs if not r.outcome.success]
        for g, b in zip(good, bad):
            prompt = _msg(f"Plan an automated solid-state synthesis run (dosing, heating, XRD) in the {tgt} element system that reacts (does not stay unreacted).")
            pairs.append({"prompt": prompt, "chosen": json.dumps([a.model_dump() for a in g.protocol_steps]),
                          "rejected": json.dumps([a.model_dump() for a in b.protocol_steps]),
                          "kind": "success_vs_failure", "split": g.split,
                          "chosen_id": g.id, "rejected_id": b.id})
    return pairs
