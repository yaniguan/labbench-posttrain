"""Anti-hack answer extraction, dimensional unit normalization, and chemical graph invariants."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from typing import Any

from posttrain.hacking_checks import META_RE
from posttrain.reward_baselines import no_repetition_reward

_ANSWER_TAG_RE = re.compile(r"\[ANSWER_START\](.*?)\[ANSWER_END\]", re.S | re.I)
_XML_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S | re.I)
_STRIP_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


@dataclass(frozen=True)
class ExtractedAnswer:
    """Verified answer extraction result with explicit anti-hack audit flags."""

    raw_completion: str
    answer_text: str
    parsed: bool
    parse_mode: str
    multi_tag_violation: bool = False
    metagaming_violation: bool = False
    repetition_violation: bool = False
    repetition_score: float = 1.0

    @property
    def is_clean(self) -> bool:
        return (
            self.parsed
            and not self.multi_tag_violation
            and not self.metagaming_violation
            and not self.repetition_violation
        )


def strip_think_blocks(text: str) -> str:
    """Remove `<think>...</think>` reasoning spans before extracting structured output."""
    if not text:
        return ""
    cleaned = _STRIP_THINK_RE.sub("", text)
    lower = cleaned.lower()
    if "</think>" in lower:
        cleaned = cleaned[lower.index("</think>") + len("</think>") :]
    elif "<think>" in lower:
        # Unclosed <think> means truncated mid-thought before emitting an answer
        cleaned = cleaned[: lower.index("<think>")]
    return cleaned.strip()


def extract_verified_answer(
    completion: str,
    *,
    allow_raw_fallback: bool = False,
    forbid_multi_tag: bool = True,
    forbid_metagaming: bool = True,
    min_repetition_score: float = 0.25,
) -> ExtractedAnswer:
    """Extract a single verified answer span while blocking multi-tag hedging and repetition loops."""
    if not completion or not completion.strip():
        return ExtractedAnswer("", "", False, "empty")

    rep_score = float(no_repetition_reward(completion))
    rep_bad = rep_score < min_repetition_score
    post_think = strip_think_blocks(completion)
    # Audit metagaming on the visible answer region (and blatant grader-talk overall)
    meta_bad = bool(forbid_metagaming and META_RE.search(completion))

    for mode, pattern in (
        ("answer_tag", _ANSWER_TAG_RE),
        ("xml_answer", _XML_ANS_RE),
        ("boxed", _BOXED_RE),
    ):
        matches = pattern.findall(post_think) or pattern.findall(completion)
        if matches:
            multi_bad = forbid_multi_tag and len(matches) > 1
            span = matches[0].strip()
            return ExtractedAnswer(
                raw_completion=completion,
                answer_text=span,
                parsed=bool(span) and not multi_bad,
                parse_mode=mode,
                multi_tag_violation=multi_bad,
                metagaming_violation=meta_bad,
                repetition_violation=rep_bad,
                repetition_score=rep_score,
            )

    fenced = _FENCED_JSON_RE.findall(post_think)
    if fenced:
        multi_bad = forbid_multi_tag and len(fenced) > 1
        return ExtractedAnswer(
            raw_completion=completion,
            answer_text=fenced[0].strip(),
            parsed=not multi_bad,
            parse_mode="fenced_json",
            multi_tag_violation=multi_bad,
            metagaming_violation=meta_bad,
            repetition_violation=rep_bad,
            repetition_score=rep_score,
        )

    if allow_raw_fallback and post_think:
        return ExtractedAnswer(
            raw_completion=completion,
            answer_text=post_think,
            parsed=True,
            parse_mode="raw_post_think",
            multi_tag_violation=False,
            metagaming_violation=meta_bad,
            repetition_violation=rep_bad,
            repetition_score=rep_score,
        )

    return ExtractedAnswer(
        raw_completion=completion,
        answer_text="",
        parsed=False,
        parse_mode="missing_tag",
        metagaming_violation=meta_bad,
        repetition_violation=rep_bad,
        repetition_score=rep_score,
    )


def parse_strict_json(text: str) -> Any | None:
    """Parse JSON strictly without allowing trailingprose or concatenated objects."""
    s = (text or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Dimensional Scientific Unit Normalizer
# ---------------------------------------------------------------------------

_UNIT_ALIASES: dict[str, str] = {
    "hartree": "ha",
    "ha": "ha",
    "ev": "ev",
    "kcal/mol": "kcal/mol",
    "kj/mol": "kj/mol",
    "m": "m",
    "mol/l": "m",
    "mm": "mm",
    "mmol/l": "mm",
    "um": "um",
    "µm": "um",
    "μm": "um",
    "nm": "nm",
    "pm": "pm",
    "k": "k",
    "kelvin": "k",
    "c": "c",
    "°c": "c",
    "degc": "c",
    "ms/cm": "ms/cm",
    "s/m": "s/m",
    "s/cm": "s/cm",
    "us/cm": "us/cm",
    "µs/cm": "us/cm",
    "da": "da",
    "g/mol": "da",
    "kda": "kda",
    "l": "l",
    "ml": "ml",
    "ul": "ul",
    "µl": "ul",
    "μl": "ul",
    "s": "s",
    "sec": "s",
    "min": "min",
    "h": "h",
    "hr": "h",
    "%": "%",
    "fraction": "fraction",
}

# Multiplicative factors into canonical base units per dimension
_LINEAR_DIMENSIONS: dict[str, dict[str, float]] = {
    "energy": {
        "ev": 1.0,
        "ha": 27.211386245988,
        "kcal/mol": 0.0433641153087705,
        "kj/mol": 0.0103642696562622,
    },
    "concentration": {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
    },
    "conductivity": {
        "ms/cm": 1.0,
        "s/m": 10.0,       # 1 S/m = 10 mS/cm
        "s/cm": 1000.0,    # 1 S/cm = 1000 mS/cm
        "us/cm": 1e-3,
    },
    "mass_mw": {
        "da": 1.0,
        "kda": 1000.0,
    },
    "volume": {
        "ul": 1.0,
        "ml": 1000.0,
        "l": 1e6,
    },
    "time": {
        "s": 1.0,
        "min": 60.0,
        "h": 3600.0,
    },
    "ratio": {
        "fraction": 1.0,
        "%": 0.01,
    },
}


def normalize_unit_symbol(unit: str | None) -> str:
    if not unit:
        return ""
    raw = unit.strip().lower().replace(" ", "")
    return _UNIT_ALIASES.get(raw, raw)


def units_compatible(u1: str | None, u2: str | None) -> bool:
    a, b = normalize_unit_symbol(u1), normalize_unit_symbol(u2)
    if a == b:
        return True
    if {a, b} == {"k", "c"}:
        return True
    for table in _LINEAR_DIMENSIONS.values():
        if a in table and b in table:
            return True
    return False


def convert_scientific_unit(value: float, from_unit: str | None, to_unit: str | None) -> float:
    """Convert `value` between compatible physical units or raise `ValueError` on dimensional mismatch."""
    if not math.isfinite(value):
        raise ValueError("Non-finite physical value")
    u_from = normalize_unit_symbol(from_unit)
    u_to = normalize_unit_symbol(to_unit)
    if not u_to or u_from == u_to:
        return float(value)
    if not u_from:
        raise ValueError(f"Missing predicted unit when target expects {to_unit!r}")

    if {u_from, u_to} == {"c", "k"}:
        return float(value) + 273.15 if u_from == "c" else float(value) - 273.15

    for table in _LINEAR_DIMENSIONS.values():
        if u_from in table and u_to in table:
            base_val = float(value) * table[u_from]
            return base_val / table[u_to]

    raise ValueError(f"Incompatible scientific dimensions: cannot convert {from_unit!r} -> {to_unit!r}")


# ---------------------------------------------------------------------------
# Pure-Python SMILES & Stoichiometry Verifier (with optional RDKit upgrade)
# ---------------------------------------------------------------------------

_VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Xe", "Cs", "Ba", "Pt", "Au", "Hg", "Pb", "Bi",
}
_AROMATIC_ORGANIC = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S"}
_RING_MULTI_RE = re.compile(r"%(\d\d)")
_BRACKET_ATOM_RE = re.compile(r"\[([^\[\]]+)\]")


def parse_smiles_invariants(smiles: str) -> dict[str, Any]:
    """Validate chemical SMILES syntax (branches, ring closures, element table) without requiring RDKit."""
    s = (smiles or "").strip()
    if not s or any(ch.isspace() for ch in s):
        return {"valid": False, "formula": {}, "canonical": "", "reason": "empty_or_whitespace"}

    # Optional exact RDKit path when installed
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import rdMolDescriptors  # type: ignore

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return {"valid": False, "formula": {}, "canonical": "", "reason": "rdkit_invalid"}
        canon = Chem.MolToSmiles(mol, canonical=True)
        f_str = rdMolDescriptors.CalcMolFormula(mol)
        counts = {el: int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f_str)}
        return {"valid": True, "formula": counts, "canonical": canon, "reason": "ok_rdkit"}
    except ImportError:
        pass

    # Pure-Python structural & stoichiometry parser
    depth = 0
    in_bracket = False
    for ch in s:
        if ch == "[":
            if in_bracket:
                return {"valid": False, "formula": {}, "canonical": "", "reason": "nested_bracket"}
            in_bracket = True
        elif ch == "]":
            if not in_bracket:
                return {"valid": False, "formula": {}, "canonical": "", "reason": "unmatched_bracket"}
            in_bracket = False
        elif not in_bracket:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return {"valid": False, "formula": {}, "canonical": "", "reason": "unmatched_paren"}
    if depth != 0 or in_bracket:
        return {"valid": False, "formula": {}, "canonical": "", "reason": "unclosed_branch_or_bracket"}

    # Strip bracketed atoms after validating element symbols inside brackets
    atom_counts: Counter[str] = Counter()
    for b_body in _BRACKET_ATOM_RE.findall(s):
        # e.g. [13CH3+], [Li+], [PF6-], [nH]
        m = re.match(r"^\d*([A-Z][a-z]?|[bcnops])", b_body)
        if not m:
            return {"valid": False, "formula": {}, "canonical": "", "reason": f"bad_bracket_atom:{b_body}"}
        sym = m.group(1)
        el = _AROMATIC_ORGANIC.get(sym, sym)
        if el not in _VALID_ELEMENTS:
            return {"valid": False, "formula": {}, "canonical": "", "reason": f"unknown_element:{el}"}
        atom_counts[el] += 1

    outside = _BRACKET_ATOM_RE.sub("", s)
    # Check ring closures (`%10`..`%99` and single digits `1`..`9`)
    ring_tokens: Counter[str] = Counter()
    for r2 in _RING_MULTI_RE.findall(outside):
        ring_tokens[f"%{r2}"] += 1
    outside_no_multi = _RING_MULTI_RE.sub("", outside)
    for d in re.findall(r"\d", outside_no_multi):
        ring_tokens[d] += 1
    for rk, cnt in ring_tokens.items():
        if cnt % 2 != 0:
            return {"valid": False, "formula": {}, "canonical": "", "reason": f"unclosed_ring:{rk}"}

    # Scan remaining unbracketed tokens for valid organic subset & bond symbols
    i = 0
    allowed_bonds = set("-=#$:/\\.")
    while i < len(outside_no_multi):
        ch = outside_no_multi[i]
        if ch in "()" or ch.isdigit() or ch in allowed_bonds:
            i += 1
            continue
        if i + 1 < len(outside_no_multi) and outside_no_multi[i : i + 2] in ("Cl", "Br", "Si", "Li", "Na"):
            atom_counts[outside_no_multi[i : i + 2]] += 1
            i += 2
            continue
        if ch in _AROMATIC_ORGANIC:
            atom_counts[_AROMATIC_ORGANIC[ch]] += 1
            i += 1
            continue
        if ch in ("B", "C", "N", "O", "P", "S", "F", "I", "H"):
            atom_counts[ch] += 1
            i += 1
            continue
        return {"valid": False, "formula": {}, "canonical": "", "reason": f"invalid_token:{ch}"}

    if sum(atom_counts.values()) == 0:
        return {"valid": False, "formula": {}, "canonical": "", "reason": "no_atoms"}
    formula = {k: atom_counts[k] for k in sorted(atom_counts) if k != "H"}
    return {"valid": True, "formula": formula, "canonical": s, "reason": "ok_python"}


def smiles_similarity(pred_smiles: str, gold_smiles: str) -> float:
    """Compute Morgan Tanimoto (if RDKit available) or multiset SMILES character 3-gram Tanimoto in [0, 1]."""
    try:
        from rdkit import Chem, DataStructs  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore

        m1, m2 = Chem.MolFromSmiles(pred_smiles), Chem.MolFromSmiles(gold_smiles)
        if m1 is not None and m2 is not None:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
            return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except ImportError:
        pass

    def _ngrams(txt: str, n: int = 3) -> Counter[str]:
        padded = f"^{txt.strip()}$"
        return Counter(padded[i : i + n] for i in range(max(1, len(padded) - n + 1)))

    c1, c2 = _ngrams(pred_smiles), _ngrams(gold_smiles)
    inter = sum((c1 & c2).values())
    union = sum((c1 | c2).values())
    return float(inter / union) if union > 0 else 0.0
