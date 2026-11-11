"""Verified Reward Templates (`labrl.rewards.templates`) for Scientific & Robotic RLVR.

Covers the 8 canonical verifiable reward archetypes across `docs/benchmark_map.md` (205 benchmarks):
  1. `CalibratedMCQTemplate` (`calibrated_mcq`) — LAB-Bench, ProtocolQA, BioProBench PQA, GPQA, ChemBench
  2. `ScalarObservableTemplate` (`scalar_observable`) — QM9, ChEMBL IC50/Ki, ESOL, CALiSol Predict, Battery Cycle-Life
  3. `InverseDesignSupportTemplate` (`inverse_design`) — CALiSol Design, Clio, Buchwald-Hartwig / Suzuki HTE
  4. `PermutationOrderTemplate` (`step_ordering`) — BioProBench ORD, BioProt, X-WLP
  5. `FaultDiagnosisRepairTemplate` (`fault_repair`) — BioProBench ERR, CloningScenarios, Dark Reactions
  6. `StructuredActionPlanTemplate` (`action_plan`) — `protocol2actions`, Open X-Embodiment, DROID, LabSafe-1700
  7. `MolecularValiditySimilarityTemplate` (`molecular_design`) — USPTO-50K/MIT, OCSR Image-to-SMILES, MassSpecGym
  8. `SpectralPeakSetTemplate` (`spectral_peaks`) — NMRShiftDB2, QM9-NMR, MassBank, GNPS, Raman/RRUFF, XRD
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from itertools import combinations
import json
import math
import re
from typing import Any

from labpt.rewards import _kendall_tau_indices, _param_match_score
from labpt.schema import ACTION_VOCAB
from labrl.core.registry import register_reward_template
from labrl.core.types import RewardPacket
from labrl.envs import VerifiableEnv
from labrl.rewards.extractors import (
    ExtractedAnswer,
    convert_scientific_unit,
    extract_verified_answer,
    normalize_unit_symbol,
    parse_smiles_invariants,
    parse_strict_json,
    smiles_similarity,
)

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


class TemplateVerifiableEnv(VerifiableEnv):
    """Adapter turning any `VerifiedRewardTemplate` into a dual-engine (`TorchLocalEngine` + `NeMoRLEngine`) `VerifiableEnv`."""

    def __init__(
        self,
        template: "VerifiedRewardTemplate",
        *,
        name: str | None = None,
        multi_reward: bool = True,
    ) -> None:
        self.template = template
        self.name = name or f"template_{template.template_id}"
        self.multi_reward = bool(multi_reward)

    def score_observable(self, info: dict[str, Any], completion: str) -> RewardPacket:
        return self.template.score(info, completion)


class VerifiedRewardTemplate(ABC):
    """Base protocol for anti-hack, multi-channel GDPO verifiable scientific reward templates."""

    template_id: str = "base_template"
    description: str = ""
    benchmark_families: tuple[str, ...] = ()
    channels: tuple[str, ...] = ("reward/accuracy", "reward/validity")
    anti_hack_guarantees: tuple[str, ...] = (
        "Blocks multi-tag answer hedging ([ANSWER_START]x2)",
        "Blocks grader/benchmark metagaming mentions",
        "Blocks degenerate n-gram repetition loops",
    )
    expected_info_keys: tuple[str, ...] = ()
    output_format_hint: str = "[ANSWER_START]...your structured answer...[ANSWER_END]"

    def __init__(
        self,
        *,
        allow_raw_fallback: bool = False,
        forbid_multi_tag: bool = True,
        forbid_metagaming: bool = True,
        min_repetition_score: float = 0.25,
    ) -> None:
        self.allow_raw_fallback = allow_raw_fallback
        self.forbid_multi_tag = forbid_multi_tag
        self.forbid_metagaming = forbid_metagaming
        self.min_repetition_score = float(min_repetition_score)

    def zero_packet(self, *, parsed: bool = False, **telemetry: float) -> RewardPacket:
        return RewardPacket(
            scalar=0.0,
            parsed=parsed,
            channels={ch: 0.0 for ch in self.channels},
            telemetry=telemetry,
        )

    def score(self, info: dict[str, Any], completion: str) -> RewardPacket:
        extracted = extract_verified_answer(
            completion,
            allow_raw_fallback=self.allow_raw_fallback,
            forbid_multi_tag=self.forbid_multi_tag,
            forbid_metagaming=self.forbid_metagaming,
            min_repetition_score=self.min_repetition_score,
        )
        if not extracted.is_clean:
            return self.zero_packet(
                parsed=False,
                anti_hack_blocked=float(
                    extracted.multi_tag_violation
                    or extracted.metagaming_violation
                    or extracted.repetition_violation
                ),
                multi_tag_violation=float(extracted.multi_tag_violation),
                metagaming_violation=float(extracted.metagaming_violation),
                repetition_violation=float(extracted.repetition_violation),
            )
        return self._score_extracted(info, extracted)

    @abstractmethod
    def _score_extracted(self, info: dict[str, Any], extracted: ExtractedAnswer) -> RewardPacket:
        raise NotImplementedError

    def format_prompt_contract(self) -> str:
        """Generate an aligned prompt suffix so instruction format and verifier never drift apart."""
        return (
            "Reason concisely inside <think>...</think>, then output EXACTLY ONE final answer span formatted as:\n"
            f"{self.output_format_hint}"
        )

    def as_env(self, *, name: str | None = None, multi_reward: bool = True) -> TemplateVerifiableEnv:
        """Convert this template into a plug-and-play `VerifiableEnv` for `PostTrainPipeline`."""
        return TemplateVerifiableEnv(self, name=name, multi_reward=multi_reward)

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "template_id": cls.template_id,
            "class_name": cls.__name__,
            "description": cls.description,
            "benchmark_families": list(cls.benchmark_families),
            "channels": list(cls.channels),
            "expected_info_keys": list(cls.expected_info_keys),
            "output_format_hint": cls.output_format_hint,
            "anti_hack_guarantees": list(cls.anti_hack_guarantees),
        }


# ---------------------------------------------------------------------------
# 1. Calibrated Multiple-Choice & Abstention Template
# ---------------------------------------------------------------------------


@register_reward_template("calibrated_mcq")
class CalibratedMCQTemplate(VerifiedRewardTemplate):
    """Strictly proper Brier-calibrated MCQ verifier with abstention & zero-confidence anti-hack gate."""

    description = "Calibrated multiple-choice & abstention verifier (strictly proper Brier rule; blocks zero-conf wrong-guess farming)."
    benchmark_families = ("LAB-Bench", "ProtocolQA", "BioProBench-PQA", "GPQA", "ChemBench", "ScienceQA", "PubMedQA")
    channels = ("reward/accuracy", "reward/calibration", "reward/validity")
    expected_info_keys = ("answer", "choices")
    output_format_hint = "[ANSWER_START]<choice> & <confidence_0_to_100>[ANSWER_END] (or [ANSWER_START]ABSTAIN[ANSWER_END])"
    anti_hack_guarantees = VerifiedRewardTemplate.anti_hack_guarantees + (
        "Wrong answer with 0% confidence scores 0.0 (eliminates zero-confidence partial-credit farming)",
        "Concatenating multiple choices ('A | B') fails strict choice membership",
        "Explicit ABSTAIN earns controlled abstain_reward instead of encouraging blind guessing",
    )

    def __init__(
        self,
        *,
        abstain_reward: float = 0.10,
        default_confidence: float = 0.80,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.abstain_reward = float(abstain_reward)
        self.default_confidence = float(default_confidence)

    def _score_extracted(self, info: dict[str, Any], extracted: ExtractedAnswer) -> RewardPacket:
        raw = extracted.answer_text.strip()
        gold = str(info["answer"]).strip()
        choices = [str(c).strip() for c in info.get("choices", [])] if info.get("choices") else []

        if raw.upper() in ("ABSTAIN", "IDK", "INSUFFICIENT_EVIDENCE"):
            return RewardPacket(
                scalar=self.abstain_reward,
                parsed=True,
                channels={"reward/accuracy": 0.0, "reward/calibration": self.abstain_reward, "reward/validity": 1.0},
                telemetry={"abstained": 1.0},
            )

        obj = parse_strict_json(raw)
        if isinstance(obj, dict) and "answer" in obj:
            pred_choice = str(obj["answer"]).strip()
            raw_conf = float(obj.get("confidence", self.default_confidence))
            conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
        elif "&" in raw:
            parts = raw.rsplit("&", 1)
            pred_choice = parts[0].strip()
            m_num = _NUM_RE.search(parts[1])
            if not m_num:
                return self.zero_packet(parsed=False)
            c_val = float(m_num.group(0))
            if not (0.0 <= c_val <= 100.0):
                return self.zero_packet(parsed=False)
            conf = c_val / 100.0
        else:
            pred_choice = raw
            conf = self.default_confidence

        conf = max(0.0, min(1.0, conf))
        if choices:
            norm_map = {c.lower(): c for c in choices}
            if pred_choice.lower() not in norm_map:
                return self.zero_packet(parsed=False)
            pred_choice = norm_map[pred_choice.lower()]

        correct = float(pred_choice.lower() == gold.lower())
        # Strictly proper Brier score: 1 - (y - p)^2 in [0, 1]
        brier_score = 1.0 - (correct - conf) ** 2
        # Anti-hack gate: a wrong guess NEVER receives positive scalar reward even if conf=0;
        # calibration shapes the reward among correct predictions and tracks overconfidence on errors.
        cal_channel = brier_score if correct > 0.5 else 0.0
        scalar = 0.75 * correct + 0.25 * cal_channel
        return RewardPacket(
            scalar=scalar,
            parsed=True,
            channels={
                "reward/accuracy": correct,
                "reward/calibration": cal_channel,
                "reward/validity": 1.0,
            },
            telemetry={"confidence": conf, "brier_raw": brier_score},
        )


# ---------------------------------------------------------------------------
# 2. Scalar Physical Observable & Calibrated Uncertainty Template
# ---------------------------------------------------------------------------


@register_reward_template("scalar_observable")
class ScalarObservableTemplate(VerifiedRewardTemplate):
    """Unit-aware continuous physical observable + proper Gaussian uncertainty verifier."""

    description = "Continuous physical measurement verifier with dimensional unit conversion, log/linear tolerance, and proper Gaussian uncertainty scoring."
    benchmark_families = (
        "QM9", "ChEMBL-IC50/Ki", "ESOL/AqSolDB", "CALiSol-Predict",
        "USPTO-Yields", "Severson/NASA-Battery", "Matbench",
    )
    channels = ("reward/accuracy", "reward/calibration", "reward/unit_validity")
    expected_info_keys = ("target", "units")
    output_format_hint = '[ANSWER_START]{"value": <float>, "sigma": <positive_float>, "units": "<unit>"}[ANSWER_END]'
    anti_hack_guarantees = VerifiedRewardTemplate.anti_hack_guarantees + (
        "Dimensional mismatch (e.g., eV vs nM) immediately triggers unit_validity=0 and scalar=0",
        "Physical domain violation (e.g., T <= 0 K, yield > 100%, IC50 <= 0) scores 0.0",
        "Proper Gaussian log-score uniquely peaks at sigma == |error| (blocks sigma->inf and sigma->0 gaming)",
    )

    def __init__(
        self,
        *,
        scale: str = "linear",
        tolerance: float = 1.0,
        value_key: str = "value",
        sigma_key: str = "sigma",
        min_value: float | None = None,
        max_value: float | None = None,
        require_sigma: bool = False,
        **kw: Any,
    ) -> None:
        if scale not in ("linear", "relative", "log10"):
            raise ValueError(f"Unsupported scale {scale!r}; choose 'linear', 'relative', or 'log10'")
        super().__init__(**kw)
        self.scale = scale
        self.tolerance = max(float(tolerance), 1e-8)
        self.value_key = value_key
        self.sigma_key = sigma_key
        self.min_value = min_value
        self.max_value = max_value
        self.require_sigma = bool(require_sigma)

    def _score_extracted(self, info: dict[str, Any], extracted: ExtractedAnswer) -> RewardPacket:
        gold_val = float(info.get("target", info.get(self.value_key)))
        gold_units = info.get("units")
        raw = extracted.answer_text.strip()

        pred_val: float | None = None
        pred_sigma: float | None = None
        pred_units: str | None = gold_units

        obj = parse_strict_json(raw)
        if isinstance(obj, dict):
            v_raw = obj.get(self.value_key, obj.get("value"))
            if not isinstance(v_raw, (int, float)) or isinstance(v_raw, bool):
                return self.zero_packet(parsed=False)
            pred_val = float(v_raw)
            s_raw = obj.get(self.sigma_key, obj.get("sigma"))
            if s_raw is not None:
                if not isinstance(s_raw, (int, float)) or isinstance(s_raw, bool):
                    return self.zero_packet(parsed=False)
                pred_sigma = float(s_raw)
            if "units" in obj and obj["units"] is not None:
                pred_units = str(obj["units"])
        else:
            # Support `<val> +/- <sigma> <units>` or `<val> <units>`
            pm_match = re.match(
                r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
                r"(?:\s*(?:\+/-|±)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))?"
                r"\s*([A-Za-z%/°µμ0-9^+-]*)\s*$",
                raw,
            )
            if not pm_match:
                return self.zero_packet(parsed=False)
            pred_val = float(pm_match.group(1))
            if pm_match.group(2):
                pred_sigma = float(pm_match.group(2))
            if pm_match.group(3):
                pred_units = pm_match.group(3)

        if pred_val is None or not math.isfinite(pred_val):
            return self.zero_packet(parsed=False)
        if self.require_sigma and pred_sigma is None:
            return self.zero_packet(parsed=False)

        # Convert predicted value (and sigma if linear) into gold_units
        try:
            norm_val = convert_scientific_unit(pred_val, pred_units, gold_units)
            if pred_sigma is not None:
                if not math.isfinite(pred_sigma) or pred_sigma <= 0.0:
                    return self.zero_packet(parsed=False)
                # Offset units (C -> K) keep interval sigma invariant
                if {normalize_unit_symbol(pred_units), normalize_unit_symbol(gold_units)} == {"c", "k"}:
                    norm_sigma = pred_sigma
                else:
                    norm_sigma = convert_scientific_unit(pred_sigma, pred_units, gold_units)
            else:
                norm_sigma = None
        except ValueError:
            return self.zero_packet(parsed=True, unit_mismatch=1.0)

        if self.min_value is not None and norm_val < self.min_value:
            return self.zero_packet(parsed=True, domain_violation=1.0)
        if self.max_value is not None and norm_val > self.max_value:
            return self.zero_packet(parsed=True, domain_violation=1.0)

        if self.scale == "log10":
            if norm_val <= 0.0 or gold_val <= 0.0:
                return self.zero_packet(parsed=True, domain_violation=1.0)
            abs_err = abs(math.log10(norm_val) - math.log10(gold_val))
        elif self.scale == "relative":
            abs_err = abs(norm_val - gold_val) / max(abs(gold_val), 1e-8)
        else:
            abs_err = abs(norm_val - gold_val)

        acc = math.exp(-abs_err / self.tolerance)

        if norm_sigma is not None:
            # Proper Gaussian log-likelihood rescaled to [0, 1], uniquely maximized at norm_sigma == |norm_val - gold_val|
            raw_err = abs(norm_val - gold_val)
            sigma_floor = max(1e-4 * max(abs(gold_val), 1.0), 1e-6)
            eff_err = max(raw_err, sigma_floor)
            s_eff = max(norm_sigma, sigma_floor)
            # Optimal NLL at s_eff = eff_err is 0.5 + log(eff_err); excess NLL >= 0:
            excess_nll = 0.5 * ((eff_err / s_eff) ** 2 - 1.0) + math.log(s_eff / eff_err)
            cal = math.exp(-max(0.0, excess_nll))
            scalar = 0.70 * acc + 0.20 * cal + 0.10
        else:
            cal = acc
            scalar = 0.90 * acc + 0.10

        return RewardPacket(
            scalar=scalar,
            parsed=True,
            channels={
                "reward/accuracy": acc,
                "reward/calibration": cal,
                "reward/unit_validity": 1.0,
            },
            telemetry={"normalized_value": norm_val, "abs_scaled_error": abs_err},
        )


# ---------------------------------------------------------------------------
# 3. Inverse Formulation / Condition Design with Conservative Support Gate
# ---------------------------------------------------------------------------


@register_reward_template("inverse_design")
class InverseDesignSupportTemplate(VerifiedRewardTemplate):
    """Experimental-atlas inverse design verifier with simplex stoichiometry & OOD support gating."""

    description = "Inverse formulation/condition design verifier using Nadaraya-Watson kernel estimation + conservative support gate to block OOD reward hacking."
    benchmark_families = ("CALiSol-Design", "Clio-Electrolyte", "Buchwald-Hartwig-HTE", "Suzuki-Miyaura-HTE", "Matbench-Discovery")
    channels = ("reward/objective", "reward/support", "reward/validity")
    expected_info_keys = ("atlas", "objective_key")
    output_format_hint = '[ANSWER_START]{"components": {"A": 0.4, "B": 0.6}, "params": {"c": 1.0}}[ANSWER_END]'
    anti_hack_guarantees = VerifiedRewardTemplate.anti_hack_guarantees + (
        "Simplex conservation check (sum of fractions == 1.0 +- simplex_tol, all >= 0)",
        "Conservative kernel support gate (g_supp = 1 - exp(-W_eff / support_tau)) zeros out-of-hull hallucinations",
        "Categorical vocabulary enforcement against allowed chemical/reagent sets",
    )

    def __init__(
        self,
        *,
        simplex_tol: float = 0.03,
        kernel_bandwidth: float = 0.12,
        support_tau: float = 0.50,
        higher_is_better: bool = True,
        param_scales: dict[str, float] | None = None,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.simplex_tol = float(simplex_tol)
        self.kernel_bandwidth = max(float(kernel_bandwidth), 1e-5)
        self.support_tau = max(float(support_tau), 1e-5)
        self.higher_is_better = bool(higher_is_better)
        self.param_scales = dict(param_scales or {})
        self.param_bounds = dict(param_bounds or {})

    def _score_extracted(self, info: dict[str, Any], extracted: ExtractedAnswer) -> RewardPacket:
        obj = parse_strict_json(extracted.answer_text)
        if not isinstance(obj, dict):
            return self.zero_packet(parsed=False)

        comps = obj.get("components", obj.get("solvents", {}))
        params = obj.get("params", {k: v for k, v in obj.items() if k not in ("components", "solvents")})
        if not isinstance(comps, dict) or not comps or not isinstance(params, dict):
            return self.zero_packet(parsed=False)

        allowed_comps = set(info.get("allowed_components", []))
        fracs: dict[str, float] = {}
        for k, v in comps.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool) or float(v) < 0.0:
                return self.zero_packet(parsed=True, invalid_simplex=1.0)
            if allowed_comps and k not in allowed_comps:
                return self.zero_packet(parsed=True, unknown_component=1.0)
            if float(v) > 0.0:
                fracs[str(k)] = float(v)

        total_frac = sum(fracs.values())
        if not fracs or abs(total_frac - 1.0) > self.simplex_tol:
            return self.zero_packet(parsed=True, invalid_simplex=1.0)
        fracs = {k: v / total_frac for k, v in fracs.items()}

        for pk, (lo, hi) in self.param_bounds.items():
            if pk in params:
                pv = params[pk]
                if not isinstance(pv, (int, float)) or not (lo <= float(pv) <= hi):
                    return self.zero_packet(parsed=True, param_out_of_bounds=1.0)

        atlas = list(info.get("atlas", []))
        obj_key = str(info.get("objective_key", "y"))
        if not atlas:
            return self.zero_packet(parsed=True)

        weights: list[float] = []
        ys: list[float] = []
        for row in atlas:
            r_comps = row.get("components", row.get("solvents", {})) or {}
            r_params = row.get("params", row)
            all_keys = set(fracs) | set(r_comps)
            # L1/2 Bray-Curtis distance on simplex in [0, 1]
            d_comp = 0.5 * sum(abs(fracs.get(k, 0.0) - float(r_comps.get(k, 0.0))) for k in all_keys)
            d_param_sq = 0.0
            for pk, pv in params.items():
                if pk not in r_params:
                    continue
                rv = r_params[pk]
                if isinstance(pv, (int, float)) and isinstance(rv, (int, float)):
                    sc = float(self.param_scales.get(pk, max(abs(float(rv)), 1.0)))
                    d_param_sq += ((float(pv) - float(rv)) / sc) ** 2
                elif str(pv).strip().lower() != str(rv).strip().lower():
                    d_param_sq += 4.0  # strong mismatch penalty on categorical catalyst/salt
            dist_sq = (d_comp ** 2 + d_param_sq) / (self.kernel_bandwidth ** 2)
            w = math.exp(-0.5 * dist_sq)
            weights.append(w)
            ys.append(float(row[obj_key]))

        w_sum = sum(weights)
        if w_sum <= 1e-12:
            return self.zero_packet(parsed=True, ood_rejection=1.0)

        y_hat = sum(w * y for w, y in zip(weights, ys)) / w_sum
        y_min, y_max = min(ys), max(ys)
        span = max(y_max - y_min, 1e-8)
        norm_obj = (y_hat - y_min) / span if self.higher_is_better else (y_max - y_hat) / span
        norm_obj = max(0.0, min(1.0, norm_obj))

        support_gate = 1.0 - math.exp(-w_sum / self.support_tau)
        gated_obj = support_gate * norm_obj
        scalar = 0.75 * gated_obj + 0.15 * support_gate + 0.10

        return RewardPacket(
            scalar=scalar,
            parsed=True,
            channels={
                "reward/objective": gated_obj,
                "reward/support": support_gate,
                "reward/validity": 1.0,
            },
            telemetry={"estimated_observable": y_hat, "kernel_weight_sum": w_sum},
        )


# ---------------------------------------------------------------------------
# 4. Bijective Step Ordering & Safety Prerequisite Template
# ---------------------------------------------------------------------------


@register_reward_template("step_ordering")
class PermutationOrderTemplate(VerifiedRewardTemplate):
    """Bijective step-permutation verifier with Kendall's tau and hard safety prerequisite gating."""

    description = "Protocol step-ordering verifier enforcing strict permutation bijection, Kendall's tau rank correlation, and hard prerequisite safety pairs."
    benchmark_families = ("BioProBench-ORD", "BioProt", "X-WLP", "CloningScenarios")
    channels = ("reward/exact", "reward/order_tau", "reward/safety_prereq")
    expected_info_keys = ("gold",)
    output_format_hint = "[ANSWER_START][0, 2, 1, 3][ANSWER_END]"
    anti_hack_guarantees = VerifiedRewardTemplate.anti_hack_guarantees + (
        "Strict bijection check sorted(pred) == sorted(gold) blocks duplicate/partial list exploits",
        "Violating any hard safety prerequisite pair (e.g., quench/cool after exotherm, aspirate before dispense) gates total reward to 0.0",
    )

    def _score_extracted(self, info: dict[str, Any], extracted: ExtractedAnswer) -> RewardPacket:
        gold = [int(x) for x in info["gold"]]
        pred = parse_strict_json(extracted.answer_text)
        if not isinstance(pred, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in pred):
            return self.zero_packet(parsed=False)
        if len(pred) != len(gold) or sorted(pred) != sorted(gold):
            return self.zero_packet(parsed=False, bijection_violation=1.0)

        pos_pred = {step_id: idx for idx, step_id in enumerate(pred)}
        pos_gold = {step_id: idx for idx, step_id in enumerate(gold)}

        # Hard prerequisite pairs: list of [must_come_before, must_come_after] step IDs
        prereqs = info.get("prerequisites", []) or []
        prereq_ok = 1.0
        for before_id, after_id in prereqs:
            if pos_pred.get(int(before_id), 0) >= pos_pred.get(int(after_id), -1):
                prereq_ok = 0.0
                break

        exact = float(pred == gold)
        if len(gold) <= 1:
            tau_rescaled = 1.0
        else:
            pairs = list(combinations(gold, 2))
            conc = sum((pos_gold[a] - pos_gold[b]) * (pos_pred[a] - pos_pred[b]) > 0 for a, b in pairs)
            tau = (2.0 * conc - len(pairs)) / len(pairs)
            tau_rescaled = 0.5 * (tau + 1.0)

        scalar = prereq_ok * (0.55 * exact + 0.45 * tau_rescaled)
        return RewardPacket(
            scalar=scalar,
            parsed=True,
            channels={
                "reward/exact": prereq_ok * exact,
                "reward/order_tau": prereq_ok * tau_rescaled,
                "reward/safety_prereq": prereq_ok,
            },
        )


# ---------------------------------------------------------------------------
# 5. Causal Fault Diagnosis & Anti-Copy Repair Template
# ---------------------------------------------------------------------------


@register_reward_template("fault_repair")
class FaultDiagnosisRepairTemplate(VerifiedRewardTemplate):
    """Two-stage protocol fault detection + quantitative repair verifier."""

    description = "Wet-lab error diagnosis and repair verifier with anti-copy guard and quantitative reagent/parameter recovery."
    benchmark_families = ("BioProBench-ERR", "ProtocolQA-Repair", "Dark-Reactions")
    channels = ("reward/detection", "reward/repair", "reward/validity")
    expected_info_keys = ("is_correct", "shown", "gold")
    output_format_hint = "[ANSWER_START]True[ANSWER_END] or [ANSWER_START]False || <corrected_step>[ANSWER_END]"
    anti_hack_guarantees = VerifiedRewardTemplate.anti_hack_guarantees + (
        "Anti-copy guard: repeating the corrupted `shown` step as the repair scores 0.0 on repair",
        "Strict boolean verdict token ('Falsehood' or hedging prose fails verdict parse)",
        "Numeric parameter F1 checks exact recovery of altered molarity/temperature/time",
    )

    @staticmethod
    def _norm_tokens(s: str) -> list[str]:
        return re.findall(r"[a-z0-9.]+", (s or "").lower())

    def _score_extracted(self, info: dict[str, Any], extracted: ExtractedAnswer) -> RewardPacket:
        is_correct_gold = bool(info["is_correct"])
        shown = str(info.get("shown", ""))
        gold_fix = str(info.get("gold", ""))
        raw = extracted.answer_text.strip()

        obj = parse_strict_json(raw)
        if isinstance(obj, dict) and "is_correct" in obj and isinstance(obj["is_correct"], bool):
            pred_ok = obj["is_correct"]
            repair_text = str(obj.get("repair", "")).strip()
        elif "||" in raw:
            left, right = raw.split("||", 1)
            verdict = left.strip().lower()
            if verdict not in ("true", "false"):
                return self.zero_packet(parsed=False)
            pred_ok = verdict == "true"
            repair_text = right.strip()
        else:
            verdict = raw.lower()
            if verdict not in ("true", "false"):
                return self.zero_packet(parsed=False)
            pred_ok = verdict == "true"
            repair_text = ""

        det = float(pred_ok == is_correct_gold)
        if is_correct_gold:
            # Clean step: full credit when correctly diagnosed True without unnecessary rewrite
            rep = 1.0 if pred_ok else 0.0
            scalar = 1.0 if pred_ok else 0.0
            return RewardPacket(
                scalar=scalar,
                parsed=True,
                channels={"reward/detection": det, "reward/repair": rep, "reward/validity": 1.0},
            )

        # Corrupted step (`is_correct_gold == False`): must detect False AND provide genuine repair
        if pred_ok or not repair_text:
            return RewardPacket(
                scalar=0.25 * det,
                parsed=True,
                channels={"reward/detection": det, "reward/repair": 0.0, "reward/validity": 1.0},
            )

        pred_toks = self._norm_tokens(repair_text)
        shown_toks = self._norm_tokens(shown)
        gold_toks = self._norm_tokens(gold_fix)
        if not pred_toks or pred_toks == shown_toks:
            return RewardPacket(
                scalar=0.20 * det,
                parsed=True,
                channels={"reward/detection": det, "reward/repair": 0.0, "reward/validity": 1.0},
                telemetry={"copied_corrupted_step": 1.0},
            )

        # Compute repair overlap on newly introduced/changed tokens + numeric parameters
        c_pred, c_gold = Counter(pred_toks), Counter(gold_toks)
        common = sum((c_pred & c_gold).values())
        tok_f1 = (2.0 * common / (len(pred_toks) + len(gold_toks))) if (pred_toks and gold_toks) else 0.0

        gold_nums = _NUM_RE.findall(gold_fix)
        pred_nums = _NUM_RE.findall(repair_text)
        if gold_nums:
            cg_n, cp_n = Counter(gold_nums), Counter(pred_nums)
            c_num = sum((cg_n & cp_n).values())
            num_f1 = 2.0 * c_num / max(len(gold_nums) + len(pred_nums), 1)
            rep_score = 0.50 * tok_f1 + 0.50 * num_f1
        else:
            rep_score = tok_f1

        scalar = 0.35 * det + 0.65 * rep_score
        return RewardPacket(
            scalar=scalar,
            parsed=True,
            channels={"reward/detection": det, "reward/repair": rep_score, "reward/validity": 1.0},
        )


# ---------------------------------------------------------------------------
# 6. Structured Robotic / Liquid-Handler Action Plan Template
# ---------------------------------------------------------------------------


@register_reward_template("action_plan")
class StructuredActionPlanTemplate(VerifiedRewardTemplate):
    """Hungarian 1-to-1 action sequence + hard actuator safety bounds verifier."""

    description = "Structured laboratory automation & robotic action plan verifier (1-to-1 Hungarian step matching + hard physical safety bounds)."
    benchmark_families = ("protocol2actions", "Open-X-Embodiment", "DROID", "LabSafe-1700", "ALOHA")
    channels = ("reward/schema", "reward/action_f1", "reward/order", "reward/params")
    expected_info_keys = ("reference_actions",)
    output_format_hint = '[ANSWER_START][{"action": "aspirate", "params": {"volume_uL": 100.0}}][ANSWER_END]'
    anti_hack_guarantees = VerifiedRewardTemplate.anti_hack_guarantees + (
        "1-to-1 Hungarian matching blocks bag-of-verbs repetition padding",
        "Hard physical parameter safety bounds (temperature_C, volume_uL, speed_rpm) multiplicatively gate unsafe plans to 0.0",
    )

    def __init__(
        self,
        *,
        rel_tol: float = 0.05,
        allowed_actions: set[str] | list[str] | None = None,
        safety_bounds: dict[str, tuple[float, float]] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.rel_tol = float(rel_tol)
        self.allowed_actions = set(
            allowed_actions
            if allowed_actions is not None
            else (set(ACTION_VOCAB) | {"incubate", "vortex", "pipette", "mix", "seal", "unseal"})
        )
        self.safety_bounds = dict(
            safety_bounds
            or {
                "temperature_C": (-100.0, 250.0),
                "volume_uL": (0.1, 50000.0),
                "speed_rpm": (0.0, 20000.0),
                "duration_s": (0.0, 604800.0),
            }
        )

    def _score_extracted(self, info: dict[str, Any], extracted: ExtractedAnswer) -> RewardPacket:
        ref = info["reference_actions"]
        ref_obj = json.loads(ref) if isinstance(ref, str) else ref
        pred_obj = parse_strict_json(extracted.answer_text)
        if not isinstance(pred_obj, list) or not pred_obj or not isinstance(ref_obj, list) or not ref_obj:
            return self.zero_packet(parsed=False)

        allowed = set(info.get("allowed_actions", [])) or self.allowed_actions

        # Hard schema & physical actuator safety gate
        for step in pred_obj:
            if not isinstance(step, dict) or step.get("action") not in allowed:
                return self.zero_packet(parsed=False)
            p = step.get("params", {})
            if not isinstance(p, dict):
                return self.zero_packet(parsed=False)
            for b_key, (lo, hi) in self.safety_bounds.items():
                if b_key in p and isinstance(p[b_key], (int, float)):
                    if not (lo <= float(p[b_key]) <= hi):
                        return self.zero_packet(parsed=True, safety_violation=1.0)

        import numpy as np
        from scipy.optimize import linear_sum_assignment

        n_r, n_p = len(ref_obj), len(pred_obj)
        score_mat = np.zeros((n_r, n_p), dtype=np.float64)
        param_mat = np.zeros((n_r, n_p), dtype=np.float64)
        for ri, r_step in enumerate(ref_obj):
            for pi, p_step in enumerate(pred_obj):
                if r_step.get("action") == p_step.get("action"):
                    ps = _param_match_score(p_step.get("params", {}) or {}, r_step.get("params", {}) or {}, rel_tol=self.rel_tol)
                    pos_bonus = 1e-3 * (1.0 - abs(ri / max(n_r, 1) - pi / max(n_p, 1)))
                    param_mat[ri, pi] = ps
                    score_mat[ri, pi] = 0.5 + 0.5 * ps + pos_bonus

        r_ind, p_ind = linear_sum_assignment(-score_mat)
        matched = [(int(r), int(p)) for r, p in zip(r_ind, p_ind) if score_mat[r, p] >= 0.5]
        if not matched:
            return RewardPacket(
                scalar=0.0,
                parsed=True,
                channels={"reward/schema": 1.0, "reward/action_f1": 0.0, "reward/order": 0.0, "reward/params": 0.0},
            )

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

        param_prec = sum(float(param_mat[r, p]) for r, p in matched) / n_r
        total = 0.40 * action_f1 + 0.30 * order_tau + 0.30 * param_prec
        return RewardPacket(
            scalar=total,
            parsed=True,
            channels={
                "reward/schema": 1.0,
                "reward/action_f1": action_f1,
                "reward/order": order_tau,
                "reward/params": param_prec,
            },
        )


# ---------------------------------------------------------------------------
# 7. Molecular Structure / Reaction SMILES & Stoichiometry Template
# ---------------------------------------------------------------------------


@register_reward_template("molecular_design")
class MolecularValiditySimilarityTemplate(VerifiedRewardTemplate):
    """Chemical SMILES syntax, heavy-atom stoichiometry, and structural Tanimoto verifier."""

    description = "Molecular generation, retrosynthesis, and OCSR SMILES verifier checking chemical syntax, heavy-atom stoichiometry, and Tanimoto similarity."
    benchmark_families = ("USPTO-50K", "USPTO-MIT", "OCSR-Benchmarks", "MassSpecGym", "ChemLLMBench")
    channels = ("reward/exact_match", "reward/similarity", "reward/chem_validity")
    expected_info_keys = ("target_smiles",)
    output_format_hint = "[ANSWER_START]<valid_SMILES_string>[ANSWER_END]"
    anti_hack_guarantees = VerifiedRewardTemplate.anti_hack_guarantees + (
        "Invalid SMILES syntax (unclosed rings, unmatched branches, bogus atom symbols) gates all channels to 0.0",
        "Heavy-atom stoichiometry check penalizes formula drift on isomer/elucidation tasks",
    )

    def _score_extracted(self, info: dict[str, Any], extracted: ExtractedAnswer) -> RewardPacket:
        raw = extracted.answer_text.strip()
        obj = parse_strict_json(raw)
        pred_smiles = str(obj["smiles"]).strip() if isinstance(obj, dict) and "smiles" in obj else raw

        inv_pred = parse_smiles_invariants(pred_smiles)
        if not inv_pred["valid"]:
            return self.zero_packet(parsed=False, invalid_smiles=1.0)

        gold_smiles = str(info["target_smiles"]).strip()
        inv_gold = parse_smiles_invariants(gold_smiles)
        exact = float(inv_pred["canonical"] == inv_gold.get("canonical", gold_smiles))
        sim = 1.0 if exact == 1.0 else smiles_similarity(pred_smiles, gold_smiles)

        formula_match = float(inv_pred["formula"] == inv_gold["formula"]) if inv_gold["valid"] else 0.0
        shaped_sim = 0.75 * sim + 0.25 * formula_match
        scalar = 0.50 * exact + 0.40 * shaped_sim + 0.10
        return RewardPacket(
            scalar=scalar,
            parsed=True,
            channels={
                "reward/exact_match": exact,
                "reward/similarity": shaped_sim,
                "reward/chem_validity": 1.0,
            },
            telemetry={"formula_match": formula_match, "raw_tanimoto": sim},
        )


# ---------------------------------------------------------------------------
# 8. Bipartite Spectral Peak-Set Matching Template (NMR / MS / IR / XRD)
# ---------------------------------------------------------------------------


@register_reward_template("spectral_peaks")
class SpectralPeakSetTemplate(VerifiedRewardTemplate):
    """1-to-1 bipartite spectral peak-list verifier blocking peak-spamming exploits."""

    description = "Spectroscopic peak-list verifier (1H/13C NMR, MS/MS m/z, IR/Raman cm^-1, XRD 2-theta) using 1-to-1 Hungarian tolerance matching."
    benchmark_families = ("NMRShiftDB2", "QM9-NMR", "MassBank", "GNPS", "QM9S-IR/Raman", "RRUFF-XRD")
    channels = ("reward/peak_f1", "reward/position_precision", "reward/validity")
    expected_info_keys = ("target_peaks",)
    output_format_hint = "[ANSWER_START][7.26, 3.81, 1.25][ANSWER_END]"
    anti_hack_guarantees = VerifiedRewardTemplate.anti_hack_guarantees + (
        "1-to-1 bipartite matching within instrument tolerance `tol` prevents dense peak-grid spamming (precision -> 0)",
        "Duplicate peak emission at the same shift/m/z matches at most one reference peak",
    )

    def __init__(self, *, tolerance: float = 0.15, max_peaks: int = 128, **kw: Any) -> None:
        super().__init__(**kw)
        self.tolerance = max(float(tolerance), 1e-6)
        self.max_peaks = int(max_peaks)

    @staticmethod
    def _extract_positions(obj: Any) -> list[float] | None:
        if isinstance(obj, dict) and "peaks" in obj:
            obj = obj["peaks"]
        if not isinstance(obj, list) or not obj:
            return None
        out: list[float] = []
        for item in obj:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                v = float(item)
            elif isinstance(item, dict) and isinstance(item.get("pos", item.get("shift", item.get("mz"))), (int, float)):
                v = float(item.get("pos", item.get("shift", item.get("mz"))))
            else:
                return None
            if not math.isfinite(v):
                return None
            out.append(v)
        return out

    def _score_extracted(self, info: dict[str, Any], extracted: ExtractedAnswer) -> RewardPacket:
        pred_peaks = self._extract_positions(parse_strict_json(extracted.answer_text))
        gold_peaks = self._extract_positions(info["target_peaks"])
        if not pred_peaks or not gold_peaks or len(pred_peaks) > self.max_peaks:
            return self.zero_packet(parsed=False)

        tol = float(info.get("tolerance", self.tolerance))
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        n_g, n_p = len(gold_peaks), len(pred_peaks)
        score_mat = np.zeros((n_g, n_p), dtype=np.float64)
        for gi, gv in enumerate(gold_peaks):
            for pi, pv in enumerate(pred_peaks):
                diff = abs(gv - pv)
                if diff <= tol:
                    score_mat[gi, pi] = 1.0 - 0.5 * (diff / tol)

        g_ind, p_ind = linear_sum_assignment(-score_mat)
        matched = [(int(g), int(p)) for g, p in zip(g_ind, p_ind) if score_mat[g, p] >= 0.5]
        if not matched:
            return RewardPacket(
                scalar=0.05,
                parsed=True,
                channels={"reward/peak_f1": 0.0, "reward/position_precision": 0.0, "reward/validity": 1.0},
            )

        tp = float(len(matched))
        prec, rec = tp / n_p, tp / n_g
        peak_f1 = 2.0 * prec * rec / (prec + rec)
        pos_prec = sum(1.0 - abs(gold_peaks[g] - pred_peaks[p]) / tol for g, p in matched) / n_g
        scalar = 0.55 * peak_f1 + 0.35 * pos_prec + 0.10
        return RewardPacket(
            scalar=scalar,
            parsed=True,
            channels={
                "reward/peak_f1": peak_f1,
                "reward/position_precision": pos_prec,
                "reward/validity": 1.0,
            },
            telemetry={"matched_peaks": tp, "predicted_peaks": float(n_p)},
        )
