"""Rule-based conversion of free-text wet-lab procedures into structured actions.

Deterministic and transparent (annotator='rule:v0'). Not perfect: it is a
weak-label source for SFT targets and is flagged auto=True.
"""
from __future__ import annotations

import re

from .schema import Action

_TEMP = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°|º|deg(?:rees)?\s*)\s*C\b")
_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?|min|mins|minutes?|days?|s|sec|seconds?)\b", re.I)
_AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|g|kg|mL|ml|µL|uL|μL|L|mmol|mol|µmol|equiv|eq)\b")

# ordered: first match wins for a clause
_RULES: list[tuple[str, re.Pattern]] = [
    ("quench", re.compile(r"\bquench", re.I)),
    ("reflux", re.compile(r"\breflux", re.I)),
    ("degas", re.compile(r"\bdegas|\bsparg|purged with (?:N2|nitrogen|argon)", re.I)),
    ("extract", re.compile(r"\bextract", re.I)),
    ("wash", re.compile(r"\bwash", re.I)),
    ("dry", re.compile(r"\bdried\b|\bdry\b|\bdrying\b|over (?:Na2SO4|MgSO4)", re.I)),
    ("filter", re.compile(r"\bfilter|\bfiltrat", re.I)),
    ("concentrate", re.compile(r"\bconcentrat|\bevaporat|in vacuo|under reduced pressure", re.I)),
    ("purify", re.compile(r"\bpurif|chromatograph|\bHPLC\b|\bcolumn\b|\bISCO\b", re.I)),
    ("crystallize", re.compile(r"\brecrystalli|\bcrystalli|\btriturat", re.I)),
    ("centrifuge", re.compile(r"\bcentrifug", re.I)),
    ("cool", re.compile(r"\bcool|ice bath|\b0\s*°C", re.I)),
    ("heat", re.compile(r"\bheat|\bwarm|oil bath|microwave", re.I)),
    ("stir", re.compile(r"\bstir", re.I)),
    ("dissolve", re.compile(r"\bdissolv", re.I)),
    ("add", re.compile(r"\badd|\bcharged|\btreated with|\bwas added|\bdropwise", re.I)),
    ("transfer", re.compile(r"\btransfer|\bpoured", re.I)),
    ("measure", re.compile(r"\bLCMS|\bTLC\b|\bNMR\b|\bMS shows|\bmonitor", re.I)),
    ("wait", re.compile(r"\bovernight|\bo/n\b|\bleft to stand|\ballowed to", re.I)),
]

_SPLIT = re.compile(r"(?<=[.;])\s+|\s*,\s*(?:then|and then)\s+|\s+then\s+")


def _params(clause: str) -> dict:
    p: dict = {}
    if m := _TEMP.search(clause):
        p["temperature_C"] = float(m.group(1))
    if m := _TIME.search(clause):
        p["duration"] = f"{m.group(1)} {m.group(2)}"
    amts = _AMOUNT.findall(clause)
    if amts:
        p["amounts"] = [f"{a} {u}" for a, u in amts[:4]]
    if re.search(r"overnight|o/n", clause, re.I) and "duration" not in p:
        p["duration"] = "overnight"
    return p


def parse_procedure(text: str, max_steps: int = 24) -> list[Action]:
    if not text:
        return []
    steps: list[Action] = []
    for clause in _SPLIT.split(text):
        clause = clause.strip()
        if len(clause) < 4:
            continue
        hits = []
        for name, pat in _RULES:
            m = pat.search(clause)
            if m:
                hits.append((m.start(), name))
        names = {n for _, n in hits}
        if "reflux" in names:  # 'heated at reflux' is one step
            hits = [h for h in hits if h[1] != "heat"]
        for _, name in sorted(hits):
            a = Action(action=name, params=_params(clause))
            if not (steps and steps[-1].action == a.action):
                steps.append(a)
        if len(steps) >= max_steps:
            break
    return steps


# ---------------- robot episodes -----------------
_ROBOT_VERBS = [
    ("pour", r"\bpour"), ("open", r"\bopen"), ("close", r"\bclose|\bshut"),
    ("wipe", r"\bwipe|\bclean"), ("push", r"\bpush|\bslide"), ("rotate", r"\bturn|\brotate|\btwist|\bflip"),
    ("pick", r"\bpick|\bgrab|\blift|\btake|\bremove"), ("place", r"\bput|\bplace|\bdrop|\bstack|\binsert|\bmove"),
]


def robot_plan_from_instruction(instr: str) -> list[Action]:
    """Map an instruction to a coarse primitive plan (rule:v0)."""
    s = instr.lower()
    plan: list[Action] = []
    for name, pat in _ROBOT_VERBS:
        if re.search(pat, s):
            if name == "pick":
                plan += [Action(action="move", params={"to": "object"}), Action(action="grasp"), Action(action="pick")]
            elif name == "place":
                if not any(a.action == "pick" for a in plan):
                    plan += [Action(action="move", params={"to": "object"}), Action(action="grasp"), Action(action="pick")]
                plan += [Action(action="move", params={"to": "target"}), Action(action="place"), Action(action="release")]
            elif name == "pour":
                plan += [Action(action="move", params={"to": "container"}), Action(action="grasp"),
                         Action(action="move", params={"to": "receptacle"}), Action(action="pour"), Action(action="place"), Action(action="release")]
            else:
                plan += [Action(action="move", params={"to": "object"}), Action(action=name)]
    if not plan:
        plan = [Action(action="move", params={"to": "object"})]
    return plan
