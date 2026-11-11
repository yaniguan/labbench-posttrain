"""Milestone 2 tests:
- `bioenvs` v2 GDPO multi-channel anti-hack rewards (`PQA` abstain/calibration, `GEN` v1 vs v2, counterfactual pairs)
- `src/labpt/rewards.py` `ordered_action_reward` penalizing reversed wet-lab step order (`dispense` before `aspirate`)
- `posttrain` `<think>` reasoning length penalty (`zmad` & clamped `overlong_buffer`) on graded BioProBench rewards
- `posttrain.grpo` true DAPO dynamic resampling + GDPO integration with `BioProBenchEnv`
- `src/labpt_nemo_rl` `BioProBenchEnv` (`EnvironmentInterface`), 4-axis `compose_recipe`, and config pre-flight validation
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bioenvs import make_env
from labpt.rewards import counterfactual_consistency_reward, ordered_action_reward, total_reward
from labpt_nemo_rl import (
    BioProBenchEnv,
    compose_recipe,
    validate_nemo_rl_yaml,
    validate_sft_rl_pair,
)
from labpt_nemo_rl.config_validation import load_yaml
from posttrain.advantages import extract_reasoning_span, reasoning_length_penalty
from posttrain.grpo import train_grpo
from posttrain.rewards import RewardOutput


def test_pqa_v2_abstain_and_gdpo_channels_and_counterfactuals():
    env = make_env("pqa", split="test", seed=1, reward_version="v2")
    s = env.sample(0)
    wrong_choice = next(c for c in s.info["choices"] if c != s.info["answer"])

    r_abstain = env.score(s, "[ANSWER_START]ABSTAIN[ANSWER_END]")
    r_wrong_hedged = env.score(s, f"[ANSWER_START]{wrong_choice} & 20[ANSWER_END]")
    r_wrong_conf = env.score(s, f"[ANSWER_START]{wrong_choice} & 100[ANSWER_END]")
    r_right_conf = env.score(s, f"[ANSWER_START]{s.info['answer']} & 100[ANSWER_END]")

    # Incentive hierarchy: right (1.0) >> honest abstain (0.10) > hedged wrong (~0.048) > confident wrong (0.0)
    assert r_right_conf.value == pytest.approx(1.0)
    assert r_abstain.value > r_wrong_hedged.value > r_wrong_conf.value == 0.0
    assert r_abstain.gdpo_channels()["reward/calibration"] > r_wrong_hedged.gdpo_channels()["reward/calibration"]

    # Option-order counterfactual keeps the true answer string invariant while permuting choices
    orig, cf = env.counterfactual_pair(0)
    assert orig.info["answer"] == cf.info["answer"]
    assert orig.info["choices"] != cf.info["choices"]
    oracle_ans = f"[ANSWER_START]{orig.info['answer']} & 100[ANSWER_END]"
    assert counterfactual_consistency_reward(env.score(orig, oracle_ans).value, env.score(cf, oracle_ans).value) == 1.0


def test_ordered_action_reward_penalizes_reversed_aspirate_dispense():
    ref = json.dumps([
        {"action": "aspirate", "params": {"volume_uL": 50.0, "well": "A1", "labware": "plate1"}},
        {"action": "heat", "params": {"temperature_C": 37.0, "duration_s": 60.0}},
        {"action": "dispense", "params": {"volume_uL": 50.0, "well": "B1", "labware": "plate1"}},
    ])
    rev = json.dumps([
        {"action": "dispense", "params": {"volume_uL": 50.0, "well": "B1", "labware": "plate1"}},
        {"action": "heat", "params": {"temperature_C": 37.0, "duration_s": 60.0}},
        {"action": "aspirate", "params": {"volume_uL": 50.0, "well": "A1", "labware": "plate1"}},
    ])
    wrong_vol = json.dumps([
        {"action": "aspirate", "params": {"volume_uL": 500.0, "well": "A1", "labware": "plate1"}},
        {"action": "heat", "params": {"temperature_C": 95.0, "duration_s": 60.0}},
        {"action": "dispense", "params": {"volume_uL": 500.0, "well": "B1", "labware": "plate1"}},
    ])

    r_exact = ordered_action_reward(ref, ref)
    r_rev = ordered_action_reward(rev, ref)
    r_bad_param = ordered_action_reward(wrong_vol, ref)

    assert r_exact["total"] == pytest.approx(1.0)
    assert r_rev["order_tau"] == pytest.approx(0.0)
    assert r_rev["total"] < r_exact["total"] - 0.25
    assert r_bad_param["param_precision"] < 0.70
    assert r_bad_param["total"] < r_exact["total"] - 0.10
    # Legacy v1 bag-of-verbs gave identical reward to reversed protocol; v2 separates them cleanly
    assert total_reward(rev, "protocol2actions", ref, {}, reward_version="v1") == pytest.approx(2.0)
    assert total_reward(rev, "protocol2actions", ref, {}, reward_version="v2") < 1.5


def test_reasoning_length_penalty_operates_on_graded_rewards_and_clamps_overlong():
    assert extract_reasoning_span("<think>first step second step</think>[ANSWER_START]A[ANSWER_END]") == "first step second step"
    assert extract_reasoning_span("chain before close</think>final") == "chain before close"
    assert extract_reasoning_span("<think>truncated runaway thought") == "truncated runaway thought"

    graded = torch.tensor([0.75, 0.75, 0.75, 0.0])
    think_lens = torch.tensor([10.0, 40.0, 200.0, 300.0])
    groups = torch.tensor([0, 0, 0, 0])
    shaped, skipped = reasoning_length_penalty(graded, think_lens, groups, mode="zmad", coef=0.1, allow_graded=True)
    assert skipped == 0
    assert shaped[0] == pytest.approx(0.75)       # <= median receives zero penalty
    assert shaped[2] < shaped[0]                  # bloated <think> trace loses reward even on graded 0.75!
    assert shaped[3] == pytest.approx(0.0)        # zero-reward completion untouched

    # Clamped overlong buffer never exceeds -coef even on huge multi-turn token counts
    multi_lens = torch.tensor([20.0, 95.0, 500.0])
    out_buf, _ = reasoning_length_penalty(
        torch.ones(3), multi_lens, torch.zeros(3, dtype=torch.long),
        mode="overlong_buffer", coef=0.4, max_length=100, buffer_length=20,
    )
    assert out_buf[0] == pytest.approx(1.0)
    assert out_buf[1] == pytest.approx(1.0 - (15.0 / 20.0) * 0.4)
    assert out_buf[2] == pytest.approx(0.6)       # clamped at 1.0 - 0.4


def test_dapo_true_dynamic_resampling_recovers_active_group():
    calls = {"n": 0}

    def stateful_reward(prompt: str, completion: str, meta: dict) -> RewardOutput:
        calls["n"] += 1
        # Round 0 (first 4 rollouts) has zero variance (all 0.0); Round 1 has non-zero variance!
        val = 0.0 if calls["n"] <= 4 else float(calls["n"] % 2)
        return RewardOutput(value=val, components={"reward/a": val, "reward/b": 1.0 - val})

    out = train_grpo(
        [{"prompt": "Spin tube at"}],
        stateful_reward,
        group_size=4,
        max_new_tokens=4,
        steps=1,
        dynamic_sampling=True,
        dynamic_sampling_max_rounds=3,
        advantage_mode="gdpo",
        reasoning_length_coef=0.05,
        seed=0,
    )
    rec = out["logger"].rows[-1]
    assert rec.status == "ran"
    assert rec.loss_parts["resample_rounds"] == 2.0
    assert rec.loss_parts["dropped_groups"] == 1.0


def test_bioprobench_env_bridge_and_recipe_validation():
    env = BioProBenchEnv({"tasks": ["pqa", "ord", "err"], "split": "test", "max_items_per_task": 4})
    rows = env.build_dataset(n_per_task=2, include_counterfactuals=True)
    assert len(rows) == 12  # 3 tasks * 2 items * (orig + cf)

    msgs = [[{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": "[ANSWER_START]True[ANSWER_END]"}] for r in rows]
    metas = [r["metadata"] for r in rows]
    step_out = env.step(msgs, metas)
    assert isinstance(step_out.rewards, dict)
    assert set(step_out.rewards.keys()) == {"reward/accuracy", "reward/structure", "reward/precision"}
    assert step_out.terminateds.all()
    _, metrics = env.global_post_process_and_metrics({"metadata": step_out.metadata})
    assert "bioprobench/counterfactual_consistency" in metrics

    # All shipped NeMo-RL configs and fixed A100 SFT/GRPO pairs pass strict validation
    for p in Path("configs/posttrain/nemo_rl").glob("*.yaml"):
        validate_nemo_rl_yaml(load_yaml(p), source=str(p))
    validate_sft_rl_pair(load_yaml("configs/scientist/sft_a100.yaml"), load_yaml("configs/scientist/grpo_a100.yaml"))
    validate_sft_rl_pair(load_yaml("configs/operator/sft_a100.yaml"), load_yaml("configs/operator/grpo_a100.yaml"))

    # 4-axis composition succeeds across single-GPU, 8xA100, and multi-node profiles
    for algo in ("dapo", "gdpo_multireward", "gspo_longcot"):
        for hw in ("local_1gpu_colocated", "node_8xa100_colocated", "cluster_disagg_async"):
            cfg = compose_recipe(algo=algo, domain="bioenvs_pqa_ord_err", model="qwen3_8b", hardware=hw)
            assert cfg["policy"]["model_name"] == "Qwen/Qwen3-8B"

    # Pre-flight validator catches GSPO + token_level_loss conflict
    with pytest.raises(ValueError, match="GSPO"):
        compose_recipe(
            algo="gspo_longcot",
            domain="bioenvs_gen",
            model="qwen3_1.7b",
            hardware="local_1gpu_colocated",
            overrides=["loss_fn.token_level_loss=true"],
        )
