"""Unit, mathematical-calibration, and adversarial anti-hack tests for `labrl.rewards`."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from labrl import (
    CalibratedMCQTemplate,
    CounterfactualPairScorer,
    EvalStage,
    FaultDiagnosisRepairTemplate,
    GRPOStage,
    GatedRewardTemplate,
    InverseDesignSupportTemplate,
    MolecularValiditySimilarityTemplate,
    PermutationOrderTemplate,
    PostTrainPipeline,
    SFTStage,
    ScalarObservableTemplate,
    ScientificCorpus,
    SpectralPeakSetTemplate,
    SplitGuard,
    StructuredActionPlanTemplate,
    make_reward_template,
    register_corpus,
)
from labrl.cli import main as cli_main
from labrl.rewards import convert_scientific_unit, parse_smiles_invariants


def test_global_anti_hack_guards() -> None:
    tmpl = CalibratedMCQTemplate()
    info = {"answer": "37 °C", "choices": ["25 °C", "37 °C", "95 °C"]}

    # 1. Clean completion succeeds
    clean = tmpl.score(info, "<think>Mammalian cells grow at 37 °C.</think>[ANSWER_START]37 °C & 95[ANSWER_END]")
    assert clean.parsed is True and clean.scalar > 0.95

    # 2. Multi-tag hedging exploit blocked
    hedged = tmpl.score(
        info,
        "[ANSWER_START]37 °C & 95[ANSWER_END]\n[ANSWER_START]25 °C & 95[ANSWER_END]",
    )
    assert hedged.parsed is False and hedged.scalar == 0.0
    assert hedged.telemetry["multi_tag_violation"] == 1.0

    # 3. Metagaming mention blocked
    meta = tmpl.score(
        info,
        "<think>Let us trick the reward function and ignore the checker.</think>[ANSWER_START]37 °C & 100[ANSWER_END]",
    )
    assert meta.parsed is False and meta.scalar == 0.0
    assert meta.telemetry["metagaming_violation"] == 1.0

    # 4. Degenerate n-gram repetition loop blocked
    spam = " ".join(["incubate cells now"] * 40) + "[ANSWER_START]37 °C & 100[ANSWER_END]"
    rep = tmpl.score(info, spam)
    assert rep.parsed is False and rep.scalar == 0.0
    assert rep.telemetry["repetition_violation"] == 1.0


def test_dimensional_scientific_unit_conversions() -> None:
    assert math.isclose(convert_scientific_unit(1.0, "Ha", "eV"), 27.211386245988, rel_tol=1e-7)
    assert math.isclose(convert_scientific_unit(0.5, "uM", "nM"), 500.0, rel_tol=1e-9)
    assert math.isclose(convert_scientific_unit(25.0, "°C", "K"), 298.15, rel_tol=1e-9)
    assert math.isclose(convert_scientific_unit(1.25, "S/m", "mS/cm"), 12.5, rel_tol=1e-9)
    assert math.isclose(convert_scientific_unit(2.5, "mL", "µL"), 2500.0, rel_tol=1e-9)
    with pytest.raises(ValueError):
        convert_scientific_unit(10.0, "eV", "nM")


def test_calibrated_mcq_and_scalar_observable_templates() -> None:
    # MCQ: anti-zero-confidence exploit + explicit abstention
    mcq = CalibratedMCQTemplate(abstain_reward=0.12)
    info = {"answer": "LiPF6", "choices": ["LiPF6", "NaCl", "KOH"]}
    wrong_zero_conf = mcq.score(info, "[ANSWER_START]NaCl & 0[ANSWER_END]")
    abstain = mcq.score(info, "[ANSWER_START]ABSTAIN[ANSWER_END]")
    all_choices_exploit = mcq.score(info, "[ANSWER_START]LiPF6 | NaCl | KOH & 80[ANSWER_END]")
    assert wrong_zero_conf.scalar == 0.0
    assert math.isclose(abstain.scalar, 0.12)
    assert abstain.scalar > wrong_zero_conf.scalar
    assert all_choices_exploit.parsed is False and all_choices_exploit.scalar == 0.0

    # ScalarObservable: unit conversion + proper Gaussian log-score peak at sigma == |err|
    obs = ScalarObservableTemplate(scale="linear", tolerance=2.0, min_value=0.0, require_sigma=True)
    obs_info = {"target": 12.0, "units": "mS/cm"}

    # Exact match predicted in S/m (1.2 S/m == 12.0 mS/cm)
    exact_unit_conv = obs.score(
        obs_info,
        '[ANSWER_START]{"value": 1.2, "sigma": 0.01, "units": "S/m"}[ANSWER_END]',
    )
    assert math.isclose(exact_unit_conv.channels["reward/accuracy"], 1.0, abs_tol=1e-6)

    # Proper scoring check at fixed prediction value=10.0 (|err| = 2.0 mS/cm):
    # Honest sigma=2.0 MUST strictly beat both overconfident sigma=0.05 and inflated sigma=50.0!
    honest = obs.score(obs_info, '[ANSWER_START]{"value": 10.0, "sigma": 2.0, "units": "mS/cm"}[ANSWER_END]')
    overconf = obs.score(obs_info, '[ANSWER_START]{"value": 10.0, "sigma": 0.05, "units": "mS/cm"}[ANSWER_END]')
    inflated = obs.score(obs_info, '[ANSWER_START]{"value": 10.0, "sigma": 50.0, "units": "mS/cm"}[ANSWER_END]')
    bad_unit = obs.score(obs_info, '[ANSWER_START]{"value": 12.0, "sigma": 0.5, "units": "eV"}[ANSWER_END]')
    assert honest.channels["reward/calibration"] > overconf.channels["reward/calibration"]
    assert honest.channels["reward/calibration"] > inflated.channels["reward/calibration"]
    assert bad_unit.scalar == 0.0 and bad_unit.telemetry["unit_mismatch"] == 1.0


def test_inverse_design_ordering_repair_action_smiles_and_spectra() -> None:
    # 1. InverseDesignSupportTemplate
    inv = InverseDesignSupportTemplate(kernel_bandwidth=0.15, support_tau=0.4, param_bounds={"c": (0.1, 4.0)})
    atlas = [
        {"components": {"EC": 0.3, "DMC": 0.7}, "params": {"c": 1.0}, "y": 11.5},
        {"components": {"EC": 0.8, "DMC": 0.2}, "params": {"c": 2.5}, "y": 3.2},
    ]
    inv_info = {"atlas": atlas, "objective_key": "y", "allowed_components": ["EC", "DMC", "EMC"]}
    opt_pkt = inv.score(inv_info, '[ANSWER_START]{"components": {"EC": 0.3, "DMC": 0.7}, "params": {"c": 1.0}}[ANSWER_END]')
    bad_simplex = inv.score(inv_info, '[ANSWER_START]{"components": {"EC": 0.9, "DMC": 0.9}, "params": {"c": 1.0}}[ANSWER_END]')
    ood_pkt = inv.score(inv_info, '[ANSWER_START]{"components": {"EMC": 1.0}, "params": {"c": 3.9}}[ANSWER_END]')
    assert opt_pkt.scalar > 0.85
    assert bad_simplex.scalar == 0.0
    assert opt_pkt.channels["reward/support"] > ood_pkt.channels["reward/support"]

    # 2. PermutationOrderTemplate with hard safety prerequisite
    ord_t = PermutationOrderTemplate()
    ord_info = {"gold": [0, 1, 2, 3], "prerequisites": [[0, 2]]}  # step 0 MUST precede step 2
    assert ord_t.score(ord_info, "[ANSWER_START][0, 1, 2, 3][ANSWER_END]").scalar == 1.0
    assert ord_t.score(ord_info, "[ANSWER_START][0, 0, 1, 2][ANSWER_END]").scalar == 0.0
    # Violating safety prerequisite [0, 2] zeros total reward even if other steps align
    unsafe_ord = ord_t.score(ord_info, "[ANSWER_START][2, 1, 0, 3][ANSWER_END]")
    assert unsafe_ord.scalar == 0.0 and unsafe_ord.channels["reward/safety_prereq"] == 0.0

    # 3. FaultDiagnosisRepairTemplate anti-copy guard
    err_t = FaultDiagnosisRepairTemplate()
    err_info = {
        "is_correct": False,
        "shown": "Incubate the reaction at 95 C for 60 min.",
        "gold": "Incubate the reaction at 37 C for 15 min.",
    }
    copied = err_t.score(err_info, "[ANSWER_START]False || Incubate the reaction at 95 C for 60 min.[ANSWER_END]")
    repaired = err_t.score(err_info, "[ANSWER_START]False || Incubate the reaction at 37 C for 15 min.[ANSWER_END]")
    assert copied.channels["reward/repair"] == 0.0
    assert repaired.scalar > 0.95

    # 4. StructuredActionPlanTemplate hard temperature safety bound
    act_t = StructuredActionPlanTemplate()
    ref_plan = [{"action": "aspirate", "params": {"volume_uL": 100.0}}, {"action": "incubate", "params": {"temperature_C": 37.0}}]
    good_plan = act_t.score({"reference_actions": ref_plan}, f"[ANSWER_START]{json.dumps(ref_plan)}[ANSWER_END]")
    unsafe_plan = act_t.score(
        {"reference_actions": ref_plan},
        '[ANSWER_START][{"action": "aspirate", "params": {"volume_uL": 100.0}}, {"action": "incubate", "params": {"temperature_C": 500.0}}][ANSWER_END]',
    )
    assert math.isclose(good_plan.scalar, 1.0)
    assert unsafe_plan.scalar == 0.0

    # 5. MolecularValiditySimilarityTemplate (SMILES syntax & similarity)
    mol_t = MolecularValiditySimilarityTemplate()
    assert parse_smiles_invariants("C1CCCCC1")["valid"] is True
    assert parse_smiles_invariants("C1CCCCC")["valid"] is False  # unclosed ring
    mol_exact = mol_t.score({"target_smiles": "CCO"}, "[ANSWER_START]CCO[ANSWER_END]")
    mol_invalid = mol_t.score({"target_smiles": "CCO"}, "[ANSWER_START]CC(O[ANSWER_END]")
    assert math.isclose(mol_exact.scalar, 1.0)
    assert mol_invalid.scalar == 0.0

    # 6. SpectralPeakSetTemplate anti-peak-spamming
    spec_t = SpectralPeakSetTemplate(tolerance=0.10)
    spec_info = {"target_peaks": [1.25, 3.80, 7.26]}
    spec_exact = spec_t.score(spec_info, "[ANSWER_START][7.26, 1.25, 3.80][ANSWER_END]")
    spam_grid = [round(0.5 * i, 2) for i in range(1, 25)]
    spec_spam = spec_t.score(spec_info, f"[ANSWER_START]{json.dumps(spam_grid)}[ANSWER_END]")
    assert math.isclose(spec_exact.scalar, 1.0)
    assert spec_exact.channels["reward/peak_f1"] > 3.0 * spec_spam.channels["reward/peak_f1"]


def test_composers_cli_and_pipeline_as_env(tmp_path: Path) -> None:
    # 1. GatedRewardTemplate + CounterfactualPairScorer
    base_obs = make_reward_template("scalar_observable", tolerance=1.0)
    gated = GatedRewardTemplate(
        objectives=[("yield", base_obs, 1.0)],
        hard_gates=[("non_toxic_solvent", lambda info, comp: 0.0 if "benzene" in comp.lower() else 1.0)],
    )
    safe_pkt = gated.score({"target": 85.0, "units": "%"}, '[ANSWER_START]{"value": 85.0, "units": "%"}[ANSWER_END]')
    toxic_pkt = gated.score(
        {"target": 85.0, "units": "%"},
        '<think>Use benzene solvent.</think>[ANSWER_START]{"value": 85.0, "units": "%"}[ANSWER_END]',
    )
    assert safe_pkt.scalar > 0.95
    assert toxic_pkt.scalar == 0.0 and toxic_pkt.channels["reward/yield_accuracy"] == 0.0

    cf_scorer = CounterfactualPairScorer(base_obs)
    consistent = cf_scorer.score_pair(
        {"target": 80.0, "units": "%"}, '[ANSWER_START]80.0 %[ANSWER_END]',
        {"target": 40.0, "units": "%"}, '[ANSWER_START]40.0 %[ANSWER_END]',
    )
    memorized = cf_scorer.score_pair(
        {"target": 80.0, "units": "%"}, '[ANSWER_START]80.0 %[ANSWER_END]',
        {"target": 40.0, "units": "%"}, '[ANSWER_START]-5.0 eV[ANSWER_END]',
    )
    assert consistent.channels["reward/consistency"] == 1.0
    assert memorized.channels["reward/consistency"] == 0.0

    # 2. 1-line `.as_env()` plugged directly into NeMo-RL `step()` AND `PostTrainPipeline`
    env = base_obs.as_env(name="qm9_gap_env")
    step_ret = env.step(
        [[{"role": "user", "content": "HOMO-LUMO gap?"}, {"role": "assistant", "content": "[ANSWER_START]5.2 eV[ANSWER_END]"}]],
        [{"info": {"target": 5.2, "units": "eV"}}],
    )
    assert isinstance(step_ret.rewards, dict)
    assert torch.isclose(step_ret.rewards["reward/accuracy"][0], torch.tensor(1.0))

    @register_corpus("qm9_template_smoke_corpus")
    class QM9SmokeCorpus(ScientificCorpus):
        name = "qm9_template_smoke_corpus"

        @property
        def split_guard(self) -> SplitGuard:
            return SplitGuard(group_key="mol_id", val_groups=frozenset({"v"}), test_groups=frozenset({"t"}))

        def to_sft(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
            return [{"prompt": "Gap in eV?", "completion": " <think>DFT.</think>[ANSWER_START]5.2 eV[ANSWER_END]"}] * 4

        def to_preference(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
            return [{"prompt": "Gap?", "chosen": " [ANSWER_START]5.2 eV[ANSWER_END]", "rejected": " [ANSWER_START]0.1 eV[ANSWER_END]"}] * 4

        def to_mpo(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
            return self.to_preference()

        def to_rl_prompts(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
            return [{"prompt": "Gap in eV?", "info": {"target": 5.2, "units": "eV"}}] * 2

    manifest = PostTrainPipeline(
        corpus="qm9_template_smoke_corpus",
        env=env,
        stages=[SFTStage(), GRPOStage(), EvalStage()],
        smoke=True,
    ).run(output_dir=tmp_path / "tmpl_pipe")
    assert "grpo" in manifest.evaluations

    # 3. `labrl templates` CLI
    all_tmpls = cli_main(["templates"])
    assert len(all_tmpls["templates"]) >= 8
    single_tmpl = cli_main(["templates", "--id", "scalar_observable"])
    assert single_tmpl["template_id"] == "scalar_observable"
    assert "[ANSWER_START]" in single_tmpl["prompt_contract"]
