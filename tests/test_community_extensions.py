"""Tests for Idea #14: Twin-Delta Step-Level Process Credit & LabPostTrainLean."""

from __future__ import annotations

from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from community_extensions import (  # noqa: E402
    ProcessStepCheck,
    ProtocolDraftState,
    count_digital_twin_violations,
    run_benchmark,
    score_step_process_bonus_permille,
    score_verifiable_trace_credit,
)


def test_protocol_draft_state_roundtrip_and_violation_count() -> None:
    for mask in range(16):
        state = ProtocolDraftState.from_mask(mask)
        assert state.to_mask() == mask
        assert count_digital_twin_violations(state) == mask.bit_count()
        assert count_digital_twin_violations(mask) == mask.bit_count()


def test_step_process_bonus_matches_lean_vacuous_theorem() -> None:
    # Productive step on verified, stripped, within-cap rollout receives positive bonus
    productive = ProcessStepCheck(
        violation_count_prev=4,
        violation_count_curr=1,
        unit_bonus_permille=120,
        core_scientific_verified=True,
        oracle_field_stripped=True,
        think_tokens=160,
        max_think_tokens=240,
    )
    assert score_step_process_bonus_permille(productive) == 360

    # Vacuous equal-violation step gets 0
    assert (
        score_step_process_bonus_permille(
            ProcessStepCheck(4, 4, 120, True, True, 160, 240)
        )
        == 0
    )
    # Violation-increasing step gets 0
    assert (
        score_step_process_bonus_permille(
            ProcessStepCheck(2, 4, 120, True, True, 160, 240)
        )
        == 0
    )
    # Unverified final answer gets 0 even if step reduced violations
    assert (
        score_step_process_bonus_permille(
            ProcessStepCheck(4, 1, 120, False, True, 160, 240)
        )
        == 0
    )
    # Oracle leak gets 0
    assert (
        score_step_process_bonus_permille(
            ProcessStepCheck(4, 1, 120, True, False, 160, 240)
        )
        == 0
    )
    # Over-budget <think> gets 0
    assert (
        score_step_process_bonus_permille(
            ProcessStepCheck(4, 1, 120, True, True, 241, 240)
        )
        == 0
    )


def test_trace_credit_anti_oscillation_and_zero_gates() -> None:
    # Attempting to farm delta credit via mid-trace regressions (3 -> 4 -> 2 -> 4 -> 0)
    # must telescope to at most v_0 = 3 total delta units.
    osc = score_verifiable_trace_credit(
        initial_violations=3,
        step_violations=[4, 2, 2, 4, 0],
        think_tokens=150,
        max_think_tokens=240,
    )
    assert osc.gate_passed is True
    assert osc.step_violation_deltas == (0, 1, 0, 0, 2)
    assert sum(osc.step_violation_deltas) == 3
    # Vacuous filler steps (indices 0, 2, 3) earn strictly 0.0 credit
    assert osc.step_credits[0] == 0.0
    assert osc.step_credits[2] == 0.0
    assert osc.step_credits[3] == 0.0

    # Unresolved final violation v_K > 0 zeroes all step and terminal credits
    unverified = score_verifiable_trace_credit(
        initial_violations=3,
        step_violations=[2, 1, 1],
        think_tokens=110,
        max_think_tokens=240,
    )
    assert unverified.gate_passed is False
    assert unverified.total_credit == 0.0
    assert all(c == 0.0 for c in unverified.step_credits)

    # Exceeding max_think_tokens zeroes all credits even when v_K == 0
    bloated = score_verifiable_trace_credit(
        initial_violations=3,
        step_violations=[2, 1, 0],
        think_tokens=280,
        max_think_tokens=240,
    )
    assert bloated.verified is True
    assert bloated.gate_passed is False
    assert bloated.total_credit == 0.0

    with pytest.raises(ValueError):
        score_verifiable_trace_credit(-1, [0], think_tokens=50)


def test_lean_module_hygiene_and_theorems_present() -> None:
    lean_file = REPO_ROOT / "lean" / "LabPostTrainLean.lean"
    assert lean_file.is_file()
    src = lean_file.read_text(encoding="utf-8")
    for required in (
        "structure RolloutRewardSpec",
        "def gatedRewardPermille",
        "def checkRolloutGate",
        "theorem unverified_rollout_gets_zero_reward",
        "def splitGuardDisjoint",
        "theorem splitGuard_no_train_test_overlap",
        "theorem vacuous_step_gets_zero_process_bonus",
    ):
        assert required in src
    assert re.search(r"\bsorry\b", src) is None
    assert re.search(r"\baxiom\b", src) is None


def test_run_benchmark_five_seeds_120_steps() -> None:
    report = run_benchmark(num_seeds=5, num_steps=120, max_think_tokens=240)
    assert report["num_seeds"] == 5
    assert report["num_steps"] == 120
    regimes = report["regimes"]

    add = regimes["additive_length_bonus"]
    sparse = regimes["sparse_terminal_only"]
    twin = regimes["twin_delta_process_credit"]

    assert twin["verified_repair_accuracy_pct"] > sparse["verified_repair_accuracy_pct"]
    assert sparse["verified_repair_accuracy_pct"] > add["verified_repair_accuracy_pct"]
    assert twin["verified_repair_accuracy_pct"] >= 90.0

    assert twin["mean_think_tokens"] < sparse["mean_think_tokens"]
    assert sparse["mean_think_tokens"] < add["mean_think_tokens"]
    assert twin["mean_think_tokens"] <= 240.0
    assert add["mean_think_tokens"] > 240.0
