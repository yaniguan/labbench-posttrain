"""Verifiable RLVR environments built on the CALiSol-23 experimental ionic conductivity atlas.

Two complementary single-turn RLVR tasks grounded directly on experimental observables (`k`, in mS/cm):
  1. `calisol_design` (`CALiSolDesign`):
     Inverse electrolyte formulation optimization under target operating temperature `T` (K),
     concentration unit (`mol/kg` or `mol/L`), and solvent basis (`w`, `v`, or `mol`).
     Verifiable multi-channel GDPO reward:
       - `reward/conductivity`: conservative Nadaraya-Watson experimental atlas conductivity normalized by
         the regime's maximum measured conductivity `k_max(T)` and gated by atlas support.
       - `reward/support`: `exp(-d_min / tau)` proximity to measured CALiSol-23 formulations (defeats
         out-of-distribution extrapolation / hallucinated recipes).
       - `reward/validity`: strict JSON schema + stoichiometry (`sum(solvents) == 1.0`, valid bounds).
  2. `calisol_predict` (`CALiSolPredict`):
     Forward experimental observable prediction + epistemic uncertainty calibration `(k_mS_cm, sigma_mS_cm)`
     evaluated on zero-DOI-leakage splits (`train` / `val` / `test` disjoint by publication `doi`).
     Verifiable multi-channel GDPO reward:
       - `reward/accuracy`: `exp(-|log1p(k_hat) - log1p(k_obs)| / 0.35)` across 3 orders of magnitude.
       - `reward/calibration`: bounded Gaussian proper log-score incentivizing honest `sigma* = |k_hat - k_obs|`.
       - `reward/validity`: physical non-negativity and well-formed JSON output.
"""
from __future__ import annotations

import csv
import functools
import json
import math
import re
from pathlib import Path
from typing import Any

from bioenvs.base import BioEnv, Reward, Sample, extract_answer, register

CALISOL_DIR = Path(__file__).resolve().parents[1] / "data" / "calisol"

# Stratified publication-level (`doi`) holdout splits:
# - Zero DOI overlap across train / val / test (prevents temperature-sweep leakage).
# - `train` retains all 13 unique lithium salts (~72.1% of clean measurements).
VAL_DOIS = frozenset({
    "10.1021/je0498863",
    "10.1149/1.1403730",
    "10.1149/2.0271802jes",
    "10.5012/bkcs.2011.32.10.3682",
    "10.1149/1.1872737",
})

TEST_DOIS = frozenset({
    "10.1149/1.1914757",
    "10.1149/1.1513556",
    "10.1016/S0378-7753(03)00153-8",
    "10.1016/S0013-4686(99)00431-4",
    "10.1016/S0378-7753(01)00697-8",
    "10.1149/1.1466857",
    "10.1007/s10008-002-0300-9",
})

_TWO_COL_SOLVENT_ABBREV = {
    "ethylene carbonate": "EC",
    "propylene carbonate": "PC",
    "dimethyl carbonate": "DMC",
    "ethyl methyl carbonate": "EMC",
}

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.S)


def normalize_c_units(raw: str) -> str:
    s = (raw or "").strip().lower().replace(" ", "")
    if s in ("mol/l", "m", "molar"):
        return "mol/L"
    if s in ("mol/kg", "molal"):
        return "mol/kg"
    return raw.strip()


def normalize_ratio_type(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in ("w", "wt", "weight", "mass"):
        return "w"
    if s in ("v", "vol", "volume"):
        return "v"
    if s in ("mol", "molar", "mole"):
        return "mol"
    return s


def parse_json_answer(text: str) -> dict[str, Any] | None:
    ans = extract_answer(text)
    if not ans:
        return None
    cleaned = _JSON_FENCE_RE.sub("", ans.strip()).strip()
    try:
        obj = json.loads(cleaned)
    except Exception:
        # Fallback: locate outermost {...} inside the answer block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(cleaned[start : end + 1])
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


@functools.lru_cache(maxsize=2)
def load_smiles_catalog(smiles_path: str | None = None) -> dict[str, dict[str, str]]:
    """Parse `calisol_smiles.csv` (`;` delimited, mixed 2/3 columns, `\\xa0` whitespace)."""
    path = Path(smiles_path) if smiles_path else (CALISOL_DIR / "calisol_smiles.csv")
    salts: dict[str, str] = {}
    solvents: dict[str, str] = {}
    if not path.exists():
        return {"salts": salts, "solvents": solvents}

    for line_idx, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        clean_line = raw_line.replace("\xa0", " ").strip()
        if not clean_line:
            continue
        parts = [p.strip() for p in clean_line.split(";") if p.strip()]
        if len(parts) < 2:
            continue
        if line_idx < 13 and len(parts) >= 3:
            # Salt lines: `Abbrev ; Full Name ; SMILES`
            salts[parts[0]] = parts[2]
        elif len(parts) == 2:
            # Lines 14-17: `Full Name ; SMILES`
            abbrev = _TWO_COL_SOLVENT_ABBREV.get(parts[0].lower(), parts[0])
            solvents[abbrev] = parts[1]
        else:
            # Solvent lines: `Full Name ; Abbrev ; SMILES`
            solvents[parts[1]] = parts[2]
    return {"salts": salts, "solvents": solvents}


def _assign_doi_split(doi: str) -> str:
    if doi in VAL_DOIS:
        return "val"
    if doi in TEST_DOIS:
        return "test"
    return "train"


@functools.lru_cache(maxsize=2)
def load_clean_calisol_records(csv_path: str | None = None) -> tuple[dict[str, Any], ...]:
    """Load, sanitize, and split `calisol23.csv` into clean physical records.

    Cleaning rules:
      - Strip whitespace on `salt` (`' LiBPFPB'` -> `'LiBPFPB'`) and `doi`.
      - Canonicalize `c_units` (`mol/kg`, `mol/L`) and `solvent_ratio_type` (`w`, `v`, `mol`).
      - Require strictly finite positive `k > 0`, `c > 0`, `T > 0` (rejects 523 NaNs + negative digitization noise).
      - Require active solvent fractions to be non-negative with sum in `[0.94, 1.06]`, then normalize to `1.0`.
    """
    path = Path(csv_path) if csv_path else (CALISOL_DIR / "calisol23.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `python scripts/get_calisol.py` to fetch CALiSol-23."
        )
    catalog = load_smiles_catalog()
    salt_smiles = catalog["salts"]
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or len(reader.fieldnames) < 8:
            raise ValueError(f"Malformed CALiSol CSV header in {path}")
        solvent_cols = list(reader.fieldnames[7:])
        for raw_idx, row in enumerate(reader):
            try:
                k = float(row["k"])
                t_kelvin = float(row["T"])
                conc = float(row["c"])
            except (TypeError, ValueError, KeyError):
                continue
            if not (
                math.isfinite(k)
                and k >= 1e-4
                and math.isfinite(conc)
                and conc >= 1e-3
                and math.isfinite(t_kelvin)
                and t_kelvin > 0.0
            ):
                continue

            raw_solvents: dict[str, float] = {}
            neg_flag = False
            for scol in solvent_cols:
                val = float(row.get(scol) or 0.0)
                if not math.isfinite(val) or val < -1e-4:
                    neg_flag = True
                    break
                if val > 1e-5:
                    raw_solvents[scol.strip()] = val
            if neg_flag or not raw_solvents:
                continue

            total_frac = sum(raw_solvents.values())
            if not (0.94 <= total_frac <= 1.06):
                continue
            norm_solvents = {s: round(v / total_frac, 6) for s, v in sorted(raw_solvents.items())}

            doi = (row.get("doi") or "").strip()
            salt = (row.get("salt") or "").strip()
            c_units = normalize_c_units(row.get("c units") or "")
            ratio_type = normalize_ratio_type(row.get("solvent ratio type") or "")
            split = _assign_doi_split(doi)
            comp_sig = f"{salt}|{round(conc, 3)}|{c_units}|{ratio_type}|" + ",".join(
                f"{s}:{w:.3f}" for s, w in norm_solvents.items()
            )
            records.append({
                "id": f"calisol-{raw_idx:05d}",
                "row_idx": raw_idx,
                "doi": doi,
                "split": split,
                "salt": salt,
                "salt_smiles": salt_smiles.get(salt, ""),
                "k_mS_cm": round(k, 6),
                "T_K": round(t_kelvin, 3),
                "c": round(conc, 5),
                "c_units": c_units,
                "solvent_ratio_type": ratio_type,
                "solvents": norm_solvents,
                "composition_key": comp_sig,
            })
    return tuple(records)


def formulation_distance(
    proposed_salt: str,
    proposed_c: float,
    proposed_solvents: dict[str, float],
    target_t_k: float,
    ref_record: dict[str, Any],
) -> float:
    """Physically scaled chemical distance between a proposed electrolyte and an isothermal atlas record."""
    salt_pen = 0.0 if proposed_salt == ref_record["salt"] else 1.25
    ref_solv: dict[str, float] = ref_record["solvents"]
    all_solv = set(proposed_solvents) | set(ref_solv)
    tv_solv = 0.5 * sum(abs(proposed_solvents.get(s, 0.0) - ref_solv.get(s, 0.0)) for s in all_solv)
    log_c_diff = abs(math.log(max(proposed_c, 1e-3)) - math.log(max(float(ref_record["c"]), 1e-3))) / 0.85
    # Only penalize thermal offset beyond the +-5 K isothermal regime window
    excess_dt = max(0.0, abs(float(target_t_k) - float(ref_record["T_K"])) - 5.0)
    temp_diff = excess_dt / 25.0
    return salt_pen + tv_solv + log_c_diff + temp_diff


def evaluate_atlas_conductivity(
    proposed_salt: str,
    proposed_c: float,
    proposed_solvents: dict[str, float],
    target_t_k: float,
    atlas_records: list[dict[str, Any]],
    *,
    top_k: int = 5,
    kernel_bandwidth: float = 0.05,
    support_scale: float = 0.28,
) -> tuple[float, float, float]:
    """Return `(estimated_k_mS_cm, support_score, d_min)` from real experimental measurements."""
    if not atlas_records:
        return 0.0, 0.0, 10.0
    scored = [
        (formulation_distance(proposed_salt, proposed_c, proposed_solvents, target_t_k, rec), float(rec["k_mS_cm"]))
        for rec in atlas_records
    ]
    # Tie-break identical formulations (`d` within 1e-6) in favor of peak measured conductivity
    scored.sort(key=lambda x: (round(x[0], 6), -x[1]))
    d_min = scored[0][0]
    support = math.exp(-d_min / support_scale)

    nearest = scored[: max(1, min(top_k, len(scored)))]
    weights = [math.exp(-(d - d_min) / kernel_bandwidth) for d, _ in nearest]
    w_sum = sum(weights) or 1.0
    k_est = sum(w * k_val for w, (_, k_val) in zip(weights, nearest)) / w_sum
    return k_est, support, d_min


@functools.lru_cache(maxsize=6)
def _build_design_tasks(split: str) -> tuple[dict[str, Any], ...]:
    """Construct verifiable electrolyte formulation optimization tasks strictly within `split`'s DOIs."""
    all_records = load_clean_calisol_records()
    split_records = [r for r in all_records if r["split"] == split]
    catalog = load_smiles_catalog()
    all_salts = sorted({r["salt"] for r in all_records})
    all_solvents = sorted({s for r in all_records for s in r["solvents"]})

    # Candidate 5K isothermal centers observed in `split`
    centers = sorted({(int(round(r["T_K"] / 5.0) * 5), r["c_units"], r["solvent_ratio_type"]) for r in split_records})

    tasks: list[dict[str, Any]] = []
    for t_band, c_units, ratio_type in centers:
        # Isothermal +-5 K experimental neighborhood strictly inside `split` DOIs
        neighborhood = [
            {
                "salt": r["salt"],
                "c": r["c"],
                "T_K": r["T_K"],
                "k_mS_cm": r["k_mS_cm"],
                "solvents": r["solvents"],
                "doi": r["doi"],
                "composition_key": r["composition_key"],
            }
            for r in split_records
            if r["c_units"] == c_units
            and r["solvent_ratio_type"] == ratio_type
            and abs(r["T_K"] - float(t_band)) <= 5.0
        ]
        by_comp: dict[str, dict[str, Any]] = {}
        for r in neighborhood:
            prev = by_comp.get(r["composition_key"])
            if prev is None or r["k_mS_cm"] > prev["k_mS_cm"]:
                by_comp[r["composition_key"]] = r
        unique_forms = sorted(by_comp.values(), key=lambda x: x["k_mS_cm"], reverse=True)
        if len(unique_forms) < 3:
            continue

        best = unique_forms[0]
        worst = unique_forms[-1]
        median_rec = unique_forms[len(unique_forms) // 2]
        k_ref = float(best["k_mS_cm"])
        k_min = float(worst["k_mS_cm"])
        if k_ref < 0.5 or (k_ref / max(k_min, 1e-3)) < 1.25:
            continue

        active_salts = sorted({r["salt"] for r in neighborhood})
        active_solvents = sorted({s for r in neighborhood for s in r["solvents"]})
        # Include deterministic distractor salts & solvents so random selection cannot trivially guess the manifold
        extra_salts = [s for s in all_salts if s not in active_salts][: max(2, 6 - len(active_salts))]
        extra_solvents = [s for s in all_solvents if s not in active_solvents][: max(4, 12 - len(active_solvents))]
        cand_salts = sorted(set(active_salts + extra_salts))
        cand_solvents = sorted(set(active_solvents + extra_solvents))

        atlas_clean = [{k: v for k, v in r.items() if k != "composition_key"} for r in unique_forms]
        tasks.append({
            "id": f"calisol-design-{split}-{t_band}K-{c_units.replace('/', '')}-{ratio_type}",
            "type": "electrolyte_design",
            "split": split,
            "target_T_K": float(t_band),
            "c_units": c_units,
            "solvent_ratio_type": ratio_type,
            "candidate_salts": cand_salts,
            "candidate_solvents": cand_solvents,
            "salt_smiles": {s: catalog["salts"].get(s, "") for s in cand_salts},
            "solvent_smiles": {s: catalog["solvents"].get(s, "") for s in cand_solvents},
            "k_max": round(k_ref, 6),
            "k_min": round(k_min, 6),
            "k_median": round(float(median_rec["k_mS_cm"]), 6),
            "dois": sorted({r["doi"] for r in neighborhood}),
            "oracle_formulation": {
                "salt": best["salt"],
                "c": best["c"],
                "c_units": c_units,
                "solvent_ratio_type": ratio_type,
                "solvents": best["solvents"],
            },
            "worst_formulation": {
                "salt": worst["salt"],
                "c": worst["c"],
                "c_units": c_units,
                "solvent_ratio_type": ratio_type,
                "solvents": worst["solvents"],
            },
            "atlas_records": atlas_clean,
        })
    return tuple(tasks)


@register
class CALiSolDesign(BioEnv):
    """Inverse liquid electrolyte design maximizing experimental ionic conductivity `k` (mS/cm)."""

    name = "calisol_design"
    task = "CALISOL_DESIGN"

    def load_rows(self, split: str) -> list[dict]:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of ('train', 'val', 'test'), got {split!r}")
        return [dict(r) for r in _build_design_tasks(split)]

    def make_sample(self, row: dict) -> Sample:
        t_k = float(row["target_T_K"])
        t_c = t_k - 273.15
        ratio_desc = {"w": "mass fraction (w)", "v": "volume fraction (v)", "mol": "mole fraction (mol)"}.get(
            row["solvent_ratio_type"], row["solvent_ratio_type"]
        )
        salt_str = ", ".join(row["candidate_salts"])
        solv_str = ", ".join(row["candidate_solvents"])
        prompt = (
            "You are designing a high-conductivity lithium-ion liquid electrolyte grounded on experimental measurements.\n"
            f"Target operating temperature: {t_k:.2f} K ({t_c:.2f} C)\n"
            f"Concentration unit (`c_units`): {row['c_units']} (valid range: 0.05 to 4.5)\n"
            f"Solvent composition basis (`solvent_ratio_type`): {row['solvent_ratio_type']} ({ratio_desc}; fractions must sum to 1.0)\n"
            f"Allowed lithium salts: {salt_str}\n"
            f"Allowed organic solvents: {solv_str}\n\n"
            "Reason briefly in <think>...</think> about dielectric dissociation, viscosity, and VTF temperature behavior, "
            "then output ONLY a JSON object inside [ANSWER_START]...[ANSWER_END] with keys:\n"
            '{"salt": "<salt>", "c": <float>, "c_units": "'
            + row["c_units"]
            + '", "solvent_ratio_type": "'
            + row["solvent_ratio_type"]
            + '", "solvents": {"<SOLV_1>": <float>, ...}}'
        )
        return Sample(id=row["id"], prompt=prompt, info=dict(row))

    def score(self, sample: Sample, completion: str) -> Reward:
        zero = Reward(
            0.0,
            False,
            {
                "reward/conductivity": 0.0,
                "reward/support": 0.0,
                "reward/validity": 0.0,
                "estimated_k_mS_cm": 0.0,
            },
        )
        obj = parse_json_answer(completion)
        if obj is None:
            return zero

        salt = str(obj.get("salt") or "").strip()
        try:
            conc = float(obj.get("c"))
        except (TypeError, ValueError):
            return zero
        c_units = normalize_c_units(str(obj.get("c_units") or sample.info["c_units"]))
        ratio_type = normalize_ratio_type(str(obj.get("solvent_ratio_type") or sample.info["solvent_ratio_type"]))
        raw_solv = obj.get("solvents")
        if not isinstance(raw_solv, dict) or not raw_solv:
            return zero

        # Parsed valid JSON structure, now check physical & chemical stoichiometry
        allowed_salts = set(sample.info["candidate_salts"])
        allowed_solvents = set(sample.info["candidate_solvents"])
        if (
            salt not in allowed_salts
            or not math.isfinite(conc)
            or not (0.02 <= conc <= 4.5)
            or c_units != sample.info["c_units"]
            or ratio_type != sample.info["solvent_ratio_type"]
        ):
            return Reward(
                0.0,
                True,
                {
                    "reward/conductivity": 0.0,
                    "reward/support": 0.0,
                    "reward/validity": 0.0,
                    "estimated_k_mS_cm": 0.0,
                },
            )

        clean_solv: dict[str, float] = {}
        for k_s, v_s in raw_solv.items():
            s_name = str(k_s).strip()
            try:
                w = float(v_s)
            except (TypeError, ValueError):
                return zero
            if not math.isfinite(w) or w < -1e-6 or s_name not in allowed_solvents:
                return Reward(
                    0.0,
                    True,
                    {
                        "reward/conductivity": 0.0,
                        "reward/support": 0.0,
                        "reward/validity": 0.0,
                        "estimated_k_mS_cm": 0.0,
                    },
                )
            if w > 1e-6:
                clean_solv[s_name] = w

        tot_w = sum(clean_solv.values())
        if not clean_solv or not (0.97 <= tot_w <= 1.03):
            return Reward(
                0.0,
                True,
                {
                    "reward/conductivity": 0.0,
                    "reward/support": 0.0,
                    "reward/validity": 0.0,
                    "estimated_k_mS_cm": 0.0,
                },
            )
        norm_solv = {s: w / tot_w for s, w in clean_solv.items()}

        k_est, support, _ = evaluate_atlas_conductivity(
            salt,
            conc,
            norm_solv,
            float(sample.info["target_T_K"]),
            sample.info["atlas_records"],
        )
        k_ref = max(float(sample.info["k_max"]), 1e-3)
        raw_k_norm = min(1.0, max(0.0, k_est / k_ref))
        if self.reward_version == "v1":
            # Legacy v1 omits conservative support gating (susceptible to OOD surrogate extrapolation)
            cond_ch = raw_k_norm
            scalar = raw_k_norm
        else:
            cond_ch = raw_k_norm * support
            scalar = 0.80 * cond_ch + 0.20 * support

        return Reward(
            float(min(1.0, max(0.0, scalar))),
            True,
            {
                "reward/conductivity": float(cond_ch),
                "reward/support": float(support),
                "reward/validity": 1.0,
                "estimated_k_mS_cm": float(k_est),
            },
        )

    def corrupt(self, info: dict, other: dict) -> dict:
        k_hi = float(info["k_max"])
        k_lo = float(info["k_min"])
        flipped = [
            {**rec, "k_mS_cm": max(0.05, k_hi + k_lo - float(rec["k_mS_cm"]))}
            for rec in info["atlas_records"]
        ]
        return {**info, "atlas_records": flipped}

    def make_counterfactual(self, sample: Sample, other_row: dict) -> Sample:
        """Pair with the most thermally separated operating condition to test VTF adaptation."""
        t0 = float(sample.info["target_T_K"])
        best_row = max(self.rows, key=lambda r: abs(float(r["target_T_K"]) - t0))
        cf_s = self.make_sample(best_row)
        return Sample(f"{sample.id}:cf", cf_s.prompt, cf_s.info)


@register
class CALiSolPredict(BioEnv):
    """Forward experimental ionic conductivity (`k` in mS/cm) + calibrated uncertainty (`sigma`) RLVR."""

    name = "calisol_predict"
    task = "CALISOL_PREDICT"

    def load_rows(self, split: str) -> list[dict]:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of ('train', 'val', 'test'), got {split!r}")
        rows = [dict(r) for r in load_clean_calisol_records() if r["split"] == split]
        return rows

    def make_sample(self, row: dict) -> Sample:
        t_k = float(row["T_K"])
        t_c = t_k - 273.15
        solv_fmt = ", ".join(f"{s}: {w:.3f}" for s, w in row["solvents"].items())
        smiles_note = f" (SMILES: {row['salt_smiles']})" if row.get("salt_smiles") else ""
        prompt = (
            "Predict the experimental ionic conductivity `k` (in mS/cm) and epistemic 1-sigma uncertainty "
            "`sigma` (in mS/cm) for the following liquid lithium-ion electrolyte:\n"
            f"  Lithium salt: {row['salt']}{smiles_note}\n"
            f"  Concentration: {row['c']:.4f} {row['c_units']}\n"
            f"  Solvent blend ({row['solvent_ratio_type']} basis): {{{solv_fmt}}}\n"
            f"  Temperature: {t_k:.2f} K ({t_c:.2f} C)\n\n"
            "Reason in <think>...</think> about ion dissociation, solvent permittivity/viscosity, and temperature, "
            "then output ONLY JSON inside [ANSWER_START]...[ANSWER_END]:\n"
            '{"k_mS_cm": <float>, "sigma_mS_cm": <float>}'
        )
        info = {
            "id": row["id"],
            "type": "conductivity_prediction",
            "doi": row["doi"],
            "split": row["split"],
            "salt": row["salt"],
            "c": row["c"],
            "c_units": row["c_units"],
            "solvent_ratio_type": row["solvent_ratio_type"],
            "solvents": row["solvents"],
            "T_K": t_k,
            "k_mS_cm": float(row["k_mS_cm"]),
            "composition_key": row["composition_key"],
        }
        return Sample(id=row["id"], prompt=prompt, info=info)

    def score(self, sample: Sample, completion: str) -> Reward:
        zero = Reward(0.0, False, {"reward/accuracy": 0.0, "reward/calibration": 0.0, "reward/validity": 0.0})
        obj = parse_json_answer(completion)
        if obj is None:
            return zero
        raw_k = obj.get("k_mS_cm", obj.get("k"))
        raw_sig = obj.get("sigma_mS_cm", obj.get("sigma", 0.5 if self.reward_version == "v1" else None))
        try:
            k_hat = float(raw_k)
            sig_hat = float(raw_sig)
        except (TypeError, ValueError):
            return zero

        if not (math.isfinite(k_hat) and math.isfinite(sig_hat)) or k_hat < 0.0 or not (0.005 <= sig_hat <= 60.0):
            return Reward(0.0, True, {"reward/accuracy": 0.0, "reward/calibration": 0.0, "reward/validity": 0.0})

        k_true = float(sample.info["k_mS_cm"])
        log_err = abs(math.log1p(k_hat) - math.log1p(k_true))
        r_acc = math.exp(-log_err / 0.35)

        # Bounded Gaussian proper log-score calibration:
        # For any prediction error e = |k_hat - k_true|, NLL excess is minimized at honest sigma* = e,
        # and hits 0 (r_cal = 1.0) when e -> 0 and sigma <= sigma_scale.
        sigma_scale = max(0.25, 0.15 * k_true)
        z = (k_hat - k_true) / sig_hat
        nll_excess = 0.5 * (z * z) + math.log(sig_hat / sigma_scale) - 0.5
        r_cal = math.exp(-max(0.0, nll_excess) / 2.0)

        scalar = r_acc if self.reward_version == "v1" else (0.70 * r_acc + 0.30 * r_cal)
        return Reward(
            float(min(1.0, max(0.0, scalar))),
            True,
            {
                "reward/accuracy": float(r_acc),
                "reward/calibration": float(r_cal),
                "reward/validity": 1.0,
            },
        )

    def corrupt(self, info: dict, other: dict) -> dict:
        k_true = float(info["k_mS_cm"])
        k_wrong = k_true * 6.0 + 8.0 if k_true < 5.0 else max(0.08, k_true * 0.06)
        return {**info, "k_mS_cm": round(k_wrong, 4)}

    def make_counterfactual(self, sample: Sample, other_row: dict) -> Sample:
        """Select the same chemical composition measured at the furthest temperature in the sweep if available."""
        comp_key = sample.info.get("composition_key")
        t0 = float(sample.info["T_K"])
        same_comp = [r for r in self.rows if r.get("composition_key") == comp_key and abs(float(r["T_K"]) - t0) >= 15.0]
        chosen = max(same_comp, key=lambda r: abs(float(r["T_K"]) - t0)) if same_comp else other_row
        cf_s = self.make_sample(chosen)
        return Sample(f"{sample.id}:cf", cf_s.prompt, cf_s.info)
