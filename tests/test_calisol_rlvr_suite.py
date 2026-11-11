"""Comprehensive verification of the CALiSol-23 Conductivity RLVR & SFT/DPO/MPO/PPO/GRPO suite."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from bioenvs import make_env
from bioenvs.calisol import (
    TEST_DOIS,
    VAL_DOIS,
    load_clean_calisol_records,
    load_smiles_catalog,
)
from labpt_nemo_rl.config_validation import compose_recipe, load_yaml, validate_nemo_rl_yaml
from labpt_nemo_rl.envs.calisol_env import CALiSolEnv
from scripts.posttrain.calisol_pipeline import (
    build_calisol_dpo_pairs,
    build_calisol_mpo_batch,
    build_calisol_sft_dataset,
    run_calisol_suite,
)
from scripts.posttrain.run_recipe import main as run_recipe_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _wrap_json(payload: dict) -> str:
    return f"<think>VTF transport check.</think>[ANSWER_START]{json.dumps(payload)}[ANSWER_END]"


def test_calisol_cleaning_smiles_and_zero_doi_leakage():
    catalog = load_smiles_catalog()
    assert len(catalog["salts"]) == 13
    assert "LiBPFPB" in catalog["salts"]
    assert "EC" in catalog["solvents"] and "DMC" in catalog["solvents"] and "DME" in catalog["solvents"]

    records = load_clean_calisol_records()
    assert len(records) >= 13_000
    # Strict physical hygiene: finite positive k, c, T; stripped salts; normalized solvents summing to ~1.0
    salts = set()
    dois_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    salts_in_train: set[str] = set()
    for r in records:
        assert math.isfinite(r["k_mS_cm"]) and r["k_mS_cm"] > 0.0
        assert math.isfinite(r["c"]) and r["c"] > 0.0
        assert math.isfinite(r["T_K"]) and r["T_K"] > 0.0
        assert r["salt"] == r["salt"].strip()
        assert abs(sum(r["solvents"].values()) - 1.0) < 1e-4
        salts.add(r["salt"])
        dois_by_split[r["split"]].add(r["doi"])
        if r["split"] == "train":
            salts_in_train.add(r["salt"])

    assert len(salts) == 13
    assert salts_in_train == salts  # All 13 salts covered in train!
    assert dois_by_split["train"].isdisjoint(dois_by_split["val"])
    assert dois_by_split["train"].isdisjoint(dois_by_split["test"])
    assert dois_by_split["val"].isdisjoint(dois_by_split["test"])
    assert dois_by_split["val"] == set(VAL_DOIS)
    assert dois_by_split["test"] == set(TEST_DOIS)


def test_calisol_design_verifiable_reward_and_anti_hacking():
    env = make_env("calisol_design", split="test", seed=0, reward_version="v2")
    assert len(env) >= 5
    s = env.sample(0)
    # Test split design tasks never touch train DOIs
    assert set(s.info["dois"]).issubset(TEST_DOIS)

    oracle_r = env.score(s, _wrap_json(s.info["oracle_formulation"]))
    worst_r = env.score(s, _wrap_json(s.info["worst_formulation"]))
    assert oracle_r.parsed is True
    assert oracle_r.value == pytest.approx(1.0, abs=0.03)
    assert oracle_r.components["reward/support"] == pytest.approx(1.0, abs=1e-5)
    assert worst_r.value < oracle_r.value - 0.15

    # Invalid stoichiometry (`sum(solvents) == 2.0` or negative `c`) fails closed -> 0.0
    bad_stoich = {
        **s.info["oracle_formulation"],
        "solvents": {k: 2.0 for k in list(s.info["oracle_formulation"]["solvents"])[:1]},
    }
    assert env.score(s, _wrap_json(bad_stoich)).value == 0.0
    bad_conc = {**s.info["oracle_formulation"], "c": -0.5}
    assert env.score(s, _wrap_json(bad_conc)).value == 0.0
    assert env.score(s, "unformatted prose without tags").value == 0.0

    # Out-of-support extrapolation guard: pick a distractor salt + distractor solvent not measured in this regime
    measured_salts = {r["salt"] for r in s.info["atlas_records"]}
    measured_solvents = {sol for r in s.info["atlas_records"] for sol in r["solvents"]}
    ood_salt = next(x for x in s.info["candidate_salts"] if x not in measured_salts)
    ood_solv = next(x for x in s.info["candidate_solvents"] if x not in measured_solvents)
    ood_form = {
        "salt": ood_salt,
        "c": 4.2,
        "c_units": s.info["c_units"],
        "solvent_ratio_type": s.info["solvent_ratio_type"],
        "solvents": {ood_solv: 1.0},
    }
    ood_r = env.score(s, _wrap_json(ood_form))
    assert ood_r.components["reward/support"] < 0.02
    assert ood_r.value < 0.02

    # Control modes (`format` and `incorrect`)
    inc_env = make_env("calisol_design", split="test", seed=0, reward_mode="incorrect")
    s_inc = inc_env.sample(0)
    assert inc_env.reward(s_inc, _wrap_json(s_inc.info["oracle_formulation"])).value < oracle_r.value - 0.15


def test_calisol_predict_verifiable_reward_and_proper_calibration():
    env = make_env("calisol_predict", split="test", seed=0, reward_version="v2")
    s = env.sample(0)
    assert s.info["doi"] in TEST_DOIS
    k_true = float(s.info["k_mS_cm"])

    exact = env.score(s, _wrap_json({"k_mS_cm": k_true, "sigma_mS_cm": max(0.1, 0.08 * k_true)}))
    assert exact.value == pytest.approx(1.0, abs=1e-5)

    # Fixed error e = 3.0 mS/cm: honest sigma=3.0 strictly beats overconfident sigma=0.05 AND lazy sigma=45.0
    k_err = k_true + 3.0
    honest = env.score(s, _wrap_json({"k_mS_cm": k_err, "sigma_mS_cm": 3.0}))
    overconf = env.score(s, _wrap_json({"k_mS_cm": k_err, "sigma_mS_cm": 0.05}))
    lazy = env.score(s, _wrap_json({"k_mS_cm": k_err, "sigma_mS_cm": 45.0}))
    assert honest.components["reward/calibration"] > overconf.components["reward/calibration"] + 0.4
    assert honest.components["reward/calibration"] > lazy.components["reward/calibration"] + 0.1
    assert honest.value > overconf.value
    assert honest.value > lazy.value

    # Counterfactual pair shifts operating temperature on real experimental sweep
    orig, cf = env.counterfactual_pair(0)
    assert orig.info["T_K"] != cf.info["T_K"]


def test_calisol_dpo_and_mpo_mining_and_nemo_rl_env():
    sft_rows = build_calisol_sft_dataset(split="train", max_design=4, max_predict=4)
    assert len(sft_rows) == 8

    pairs = build_calisol_dpo_pairs(split="train", max_pairs=12)
    assert len(pairs) == 12
    for p in pairs:
        assert p["k_chosen_mS_cm"] >= 1.8 * p["k_rejected_mS_cm"]
        assert p["k_chosen_mS_cm"] - p["k_rejected_mS_cm"] >= 2.0
        assert p["doi_chosen"] not in TEST_DOIS and p["doi_rejected"] not in TEST_DOIS

    mpo_rows = build_calisol_mpo_batch(split="train", max_pairs=4, max_unpaired=4)
    assert any(r["record_type"] == "paired" for r in mpo_rows)
    assert any(r["record_type"] == "unpaired" and r["label"] == 1 for r in mpo_rows)
    assert any(r["record_type"] == "unpaired" and r["label"] == 0 for r in mpo_rows)

    # NeMo-RL CALiSolEnv batched multi-reward step
    c_env = CALiSolEnv(tasks=["calisol_design", "calisol_predict"], split="train", multi_reward=True)
    ds = c_env.build_dataset(n_per_task=1, include_counterfactuals=True)
    assert len(ds) == 4
    design_meta = ds[0]["metadata"]
    ans = _wrap_json(design_meta["info"]["oracle_formulation"])
    step_out = c_env.step([[{"role": "assistant", "content": ans}]], [design_meta])
    assert isinstance(step_out.rewards, dict)
    assert set(step_out.rewards.keys()) == {"reward/conductivity", "reward/support", "reward/validity"}
    assert float(step_out.rewards["reward/conductivity"][0]) > 0.90
    assert step_out.terminateds.dtype == torch.bool and bool(step_out.terminateds[0])


def test_calisol_end_to_end_smoke_sft_dpo_mpo_ppo_grpo_and_recipes(tmp_path):
    out_dir = tmp_path / "calisol_smoke"
    res = run_calisol_suite(
        methods=("sft", "dpo", "mpo", "ppo", "grpo"),
        task_mode="both",
        smoke=True,
        seed=0,
        output_dir=out_dir,
    )
    for m in ("sft", "dpo", "mpo", "ppo", "grpo"):
        assert res["runs"][m]["status"] == "ran"
        assert math.isfinite(res["runs"][m]["final_loss"])
    assert res["runs"]["dpo"]["reference_unchanged"] is True
    assert res["runs"]["mpo"]["reference_unchanged"] is True
    assert res["runs"]["ppo"]["reference_unchanged"] is True
    assert res["runs"]["grpo"]["reference_unchanged"] is True
    assert "test/design_reward_mean" in res["eval_test"]
    assert (out_dir / "calisol_summary.json").exists()

    # Validate multi-GPU NeMo-RL YAML & 4-axis recipe composition for `calisol_conductivity`
    nemo_cfg = load_yaml(REPO_ROOT / "configs" / "posttrain" / "nemo_rl" / "grpo_calisol_gdpo_dapo.yaml")
    validate_nemo_rl_yaml(nemo_cfg, source="grpo_calisol_gdpo_dapo.yaml")

    composed = compose_recipe(
        algo="gdpo_multireward",
        domain="calisol_conductivity",
        model="qwen3_8b",
        hardware="node_8xa100_colocated",
    )
    assert composed["grpo"]["adv_estimator"]["gdpo_channel_weights"]["reward/conductivity"] == 1.2

    r_out = run_recipe_main([
        "--algo", "gdpo_multireward",
        "--domain", "calisol_conductivity",
        "--model", "qwen3_1.7b",
        "--hardware", "local_1gpu_colocated",
        "--smoke",
    ])
    assert r_out["result"]["status"] == "ran"
