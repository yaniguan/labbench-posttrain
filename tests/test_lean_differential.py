"""Tests for Lean 4.36.0 compilation, `#print axioms`, 300-case `#eval` parity, and `train_grpo` checks (#55)."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from posttrain.batching import clone_frozen  # noqa: E402
from posttrain.grpo import TOY_PROMPTS, _advantages, train_grpo  # noqa: E402
from posttrain.rewards import toy_protocol_reward  # noqa: E402
from posttrain.rl_common import Rollout  # noqa: E402
from posttrain.tiny import build_tiny_lm  # noqa: E402
from scripts.verify_lean_proofs_and_parity import (  # noqa: E402
    find_lake_executable,
    parse_axiom_output,
    verify_lean_and_parity,
)


def test_train_grpo_invokes_check_kl_anchor_and_check_gdpo_pareto_weights() -> None:
    """#55: `train_grpo` logs `check_kl_anchor` violations and validates `check_gdpo_pareto_weights`."""
    # 1. Valid positive kl_coef and generous max_kl_drift -> zero KL anchor violations
    m_train, tok_train = build_tiny_lm(seed=0)
    ok_res = train_grpo(
        TOY_PROMPTS,
        toy_protocol_reward,
        model=m_train,
        tokenizer=tok_train,
        steps=1,
        group_size=2,
        max_new_tokens=4,
        kl_coef=0.02,
        max_kl_drift=1.0,
        reward_mode="random_p50",
        seed=0,
    )
    assert ok_res["kl_anchor_violations"] == []

    # 2. Ablation kl_coef == 0.0 -> logs `positive_kl_coef` rule from `check_kl_anchor`
    m_zero_kl, tok_zero_kl = build_tiny_lm(seed=1)
    zero_kl_res = train_grpo(
        TOY_PROMPTS,
        toy_protocol_reward,
        model=m_zero_kl,
        tokenizer=tok_zero_kl,
        steps=1,
        group_size=2,
        max_new_tokens=4,
        kl_coef=0.0,
        reward_mode="random_p50",
        seed=1,
    )
    assert any(v["rule"] == "positive_kl_coef" for v in zero_kl_res["kl_anchor_violations"])
    assert any("kl_anchor:positive_kl_coef" in r.hacking_flags for r in zero_kl_res["logger"].rows)

    # 3. Degenerate all-zero GDPO weights rejected by `check_gdpo_pareto_weights` inside `_advantages`
    rows = [
        Rollout(
            prompt="p",
            completion="c1",
            meta={},
            group_id=0,
            input_ids=[1, 2],
            completion_mask=[0, 1],
            truncated=False,
            reward=1.0,
            components={"reward/a": 1.0, "reward/b": 0.5},
        ),
        Rollout(
            prompt="p",
            completion="c2",
            meta={},
            group_id=0,
            input_ids=[1, 3],
            completion_mask=[0, 1],
            truncated=False,
            reward=0.0,
            components={"reward/a": 0.0, "reward/b": 0.0},
        ),
    ]
    with pytest.raises(ValueError, match="gdpo_positive_weight_sum"):
        _advantages(
            rows,
            torch.tensor([1.0, 0.0]),
            torch.tensor([0, 0]),
            "gdpo",
            "mean",
            True,
            gdpo_weights={"reward/a": 0.0, "reward/b": 0.0},
        )


def test_parse_axiom_output_rejects_sorry_and_custom_axioms() -> None:
    """#55: Unit check that `parse_axiom_output` catches `sorryAx` and custom axioms."""
    clean_out = (
        "'LabPostTrainLean.thm1' does not depend on any axioms\n"
        "'LabPostTrainLean.thm2' depends on axioms: [propext, Classical.choice, Quot.sound]\n"
    )
    assert parse_axiom_output(clean_out, ["thm1", "thm2"]) == []

    sorry_out = "'LabPostTrainLean.thm_bad' depends on axioms: [propext, sorryAx]\n"
    errs = parse_axiom_output(sorry_out, ["thm_bad"])
    assert any("sorryAx" in e for e in errs)

    custom_out = "'LabPostTrainLean.thm_ax' depends on axioms: [myCustomAxiom]\n"
    errs_custom = parse_axiom_output(custom_out, ["thm_ax"])
    assert any("myCustomAxiom" in e for e in errs_custom)


def test_lean_build_axioms_and_300_case_differential_parity() -> None:
    """#55: Compile `LabPostTrainLean`, check `#print axioms`, and run 300-case `#eval` parity."""
    if find_lake_executable() is None:
        pytest.skip("Lean 4 `lake` toolchain not installed in this environment")
    report = verify_lean_and_parity(n_cases=300, seed=20261127, self_test_sorry=True)
    assert report["ok"] is True, report["errors"]
    assert report["differential_cases"] >= 250
    assert report["theorems_checked"] >= 19
