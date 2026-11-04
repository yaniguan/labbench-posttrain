"""Rule-based annotations (step type, failure reason, safety level)."""
from __future__ import annotations

import re

from .schema import Annotation, Record

RULE = "rule:v0"

HIGH_HAZARD = re.compile(
    r"\bNaH\b|LiAlH4|\bLAH\b|n-?BuLi|t-?BuLi|sec-?BuLi|butyllithium|pyrophoric|azide|NaN3|"
    r"\bHF\b|hydrofluoric|diazomethane|\bH2\b(?! ?O)|hydrogen gas|Raney|phosgene|triphosgene|"
    r"oleum|fuming|perchlor|\bCrO3\b|osmium|cyanide|\bKCN\b|\bNaCN\b|Grignard|MgBr|autoclave|"
    r"sealed tube|high pressure|thionyl chloride|SOCl2|oxalyl chloride|POCl3|Br2\b",
    re.I,
)
MED_HAZARD = re.compile(
    r"DCM|dichloromethane|chloroform|\bTFA\b|trifluoroacetic|\bHCl\b|H2SO4|sulfuric|nitric|"
    r"\bDMF\b|\bTHF\b|ether|hexane|toluene|reflux|palladium|\bPd\b|BBr3|\bNaOH\b|\bKOH\b|LiOH|"
    r"acetonitrile|methanol|ethanol",
    re.I,
)


def safety_level(text: str, max_temp: float | None = None) -> str:
    if HIGH_HAZARD.search(text or "") or (max_temp is not None and max_temp >= 150):
        return "high"
    if MED_HAZARD.search(text or "") or (max_temp is not None and max_temp >= 60):
        return "medium"
    return "low"


STEP_TYPE = {
    "add": "charging", "dissolve": "charging", "transfer": "handling", "stir": "reaction",
    "heat": "reaction", "reflux": "reaction", "cool": "reaction", "wait": "reaction",
    "degas": "setup", "quench": "workup", "extract": "workup", "wash": "workup", "dry": "workup",
    "filter": "workup", "concentrate": "workup", "purify": "purification", "crystallize": "purification",
    "centrifuge": "workup", "measure": "analysis",
    "move": "motion", "pick": "manipulation", "place": "manipulation", "grasp": "manipulation",
    "release": "manipulation", "open": "manipulation", "close": "manipulation", "pour": "liquid_handling",
    "push": "manipulation", "wipe": "manipulation", "rotate": "manipulation",
}


def annotate(r: Record) -> Record:
    anns: list[Annotation] = []
    if r.protocol_steps:
        anns.append(Annotation(key="step_type", value=[STEP_TYPE.get(a.action, "other") for a in r.protocol_steps], annotator=RULE))
    # safety
    if r.modality == "reaction":
        temps = [a.params.get("temperature_C") for a in r.protocol_steps if a.params.get("temperature_C") is not None]
        t = r.conditions.get("temperature_C")
        if t is not None:
            temps.append(t)
        blob = " ".join([r.protocol_text or "", " ".join(r.entity.get("reactants", [])), " ".join(r.entity.get("reagents", []))])
        anns.append(Annotation(key="safety_level", value=safety_level(blob, max(temps) if temps else None), annotator=RULE))
    elif r.modality == "robot_episode":
        instr = (r.entity.get("instruction") or "").lower()
        lvl = "medium" if re.search(r"knife|pour|liquid|water|cup|bottle|glass|hot|stove|oven", instr) else "low"
        anns.append(Annotation(key="safety_level", value=lvl, annotator=RULE))
    # failure reason
    fr = None
    if r.modality == "reaction" and r.outcome.value is not None:
        if r.outcome.value < 5:
            fr = "no_or_trace_product"
        elif r.outcome.value < 20:
            fr = "low_yield"
    elif r.modality == "bioassay" and r.outcome.success is False:
        fr = "inactive_in_assay"
    elif r.modality == "robot_episode" and r.outcome.success is False:
        fr = "task_not_completed"
    if fr:
        anns.append(Annotation(key="failure_reason", value=fr, annotator=RULE))
    r.annotations = [a for a in r.annotations if a.annotator != RULE] + anns
    return r
