"""Unit tests for the `yaniguan/RL` fork contract, GDPO `reward_weights` ordering, and `mpo_offline` rejection (#52)."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from labpt_nemo_rl.config_validation import compose_recipe  # noqa: E402
from labpt_nemo_rl.envs.bioprobench_env import BioProBenchEnv  # noqa: E402
from labpt_nemo_rl.envs.calisol_env import CALiSolEnv  # noqa: E402
from scripts.posttrain.check_rl_fork_contract import (  # noqa: E402
    check_ast_imports,
    check_setup_to_train_mapping,
    run_all_checks,
)
from scripts.posttrain.run_nemo_rl import main as run_nemo_main, resolve_nemo_entrypoint  # noqa: E402


def test_calisol_and_bioprobench_gdpo_reward_weights_match_sorted_env_channels() -> None:
    """#52: CALiSol (4 channels) and BioProBench (3 channels) GDPO `reward_weights` match sorted `reward/*` keys."""
    # 1. CALiSol-23: 4 sorted channels
    cal_cfg = compose_recipe(
        algo="gdpo_multireward",
        domain="calisol_conductivity",
        model="qwen3_1.7b",
        hardware="local_1gpu_colocated",
    )
    cal_env = CALiSolEnv(cal_cfg["env"]["calisol"], max_items_per_task=2)
    cal_rows = cal_env.build_dataset(n_per_task=1, include_counterfactuals=False)
    cal_step = cal_env.step(
        [[{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": "[ANSWER_START]ABSTAIN[ANSWER_END]"}] for r in cal_rows],
        [r["metadata"] for r in cal_rows],
    )
    assert isinstance(cal_step.rewards, dict)
    cal_sorted_keys = sorted(cal_step.rewards.keys())
    assert cal_sorted_keys == [
        "reward/calibration",
        "reward/conductivity",
        "reward/support",
        "reward/validity",
    ]
    cal_w_map = cal_cfg["grpo"]["adv_estimator"]["gdpo_channel_weights"]
    cal_rw = cal_cfg["grpo"]["adv_estimator"]["reward_weights"]
    assert cal_rw == [cal_w_map[k] for k in cal_sorted_keys]
    assert cal_rw == [0.6, 1.2, 0.8, 0.5]

    # 2. BioProBench (PQA + ORD + ERR): 3 sorted channels
    bio_cfg = compose_recipe(
        algo="gdpo_multireward",
        domain="bioenvs_pqa_ord_err",
        model="qwen3_1.7b",
        hardware="local_1gpu_colocated",
    )
    bio_env = BioProBenchEnv(bio_cfg["env"]["bioprobench"], max_items_per_task=2)
    bio_rows = bio_env.build_dataset(n_per_task=1, include_counterfactuals=False)
    bio_step = bio_env.step(
        [[{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": "[ANSWER_START]ABSTAIN[ANSWER_END]"}] for r in bio_rows],
        [r["metadata"] for r in bio_rows],
    )
    assert isinstance(bio_step.rewards, dict)
    bio_sorted_keys = sorted(bio_step.rewards.keys())
    assert bio_sorted_keys == [
        "reward/accuracy",
        "reward/precision",
        "reward/structure",
    ]
    bio_w_map = bio_cfg["grpo"]["adv_estimator"]["gdpo_channel_weights"]
    bio_rw = bio_cfg["grpo"]["adv_estimator"]["reward_weights"]
    assert bio_rw == [bio_w_map[k] for k in bio_sorted_keys]
    assert bio_rw == [1.0, 0.8, 0.6]

    # 3. Native-path GRPO recipes never pass forbidden keys in `grpo.length_penalty`
    for algo in ("gdpo_multireward", "gspo_longcot", "dapo"):
        composed = compose_recipe(
            algo=algo,
            domain="bioenvs_pqa_ord_err",
            model="qwen3_1.7b",
            hardware="local_1gpu_colocated",
        )
        assert composed["grpo"]["length_penalty"] is None
        assert composed["grpo"]["reward_shaping"]["max_response_length"] == composed["policy"]["max_total_sequence_length"]


def test_mpo_offline_fails_loudly_on_fork_launcher() -> None:
    """#52: `mpo_offline` must raise `NotImplementedError` rather than silently running plain DPO."""
    with pytest.raises(NotImplementedError, match="mpo_offline"):
        resolve_nemo_entrypoint("mpo_offline", {})

    with pytest.raises(NotImplementedError, match="mpo_offline"):
        run_nemo_main([
            "--algo", "mpo_offline",
            "--domain", "bioenvs_pqa_ord_err",
            "--model", "qwen3_1.7b",
            "--hardware", "local_1gpu_colocated",
            "--dry-run",
        ])


def test_ast_contract_checker_detects_missing_import_and_bad_setup_unpack(tmp_path: Path) -> None:
    """#52: Verify `check_rl_fork_contract` catches missing `setup_sft_data` and `*setup_res` positional bugs."""
    fake_fork = tmp_path / "fake_rl"
    (fake_fork / "nemo_rl" / "data").mkdir(parents=True)
    (fake_fork / "nemo_rl" / "algorithms").mkdir(parents=True)
    (fake_fork / "nemo_rl" / "__init__.py").write_text("", encoding="utf-8")
    (fake_fork / "nemo_rl" / "data" / "__init__.py").write_text("", encoding="utf-8")
    (fake_fork / "nemo_rl" / "data" / "utils.py").write_text(
        "def setup_response_data(*a, **k): pass\ndef setup_preference_data(*a, **k): pass\n",
        encoding="utf-8",
    )
    (fake_fork / "nemo_rl" / "algorithms" / "__init__.py").write_text("", encoding="utf-8")
    sft_stub = (
        "class MasterConfig: pass\n"
        "def setup(master_config, tokenizer, train_dataset, val_dataset):\n"
        "    return (policy, cluster, train_dataloader, val_dataloader, loss_fn, logger, checkpointer, sft_save_state, master_config)\n"
        "def sft_train(policy, train_dataloader, val_dataloader, tokenizer, loss_fn, master_config, logger, checkpointer, sft_save_state): pass\n"
    )
    dpo_stub = sft_stub.replace("sft_save_state", "dpo_save_state").replace("sft_train", "dpo_train")
    grpo_stub = (
        "class MasterConfig: pass\n"
        "def setup(master_config, tokenizer, dataset, val_dataset, processor=None, policy_factory=None): pass\n"
        "def grpo_train(policy, policy_generation, wrapped_dataloader, val_dataloader, tokenizer, loss_fn, task_to_env, val_task_to_env, logger, checkpointer, grpo_save_state, master_config, processor=None): pass\n"
    )
    (fake_fork / "nemo_rl" / "algorithms" / "sft.py").write_text(sft_stub, encoding="utf-8")
    (fake_fork / "nemo_rl" / "algorithms" / "dpo.py").write_text(dpo_stub, encoding="utf-8")
    (fake_fork / "nemo_rl" / "algorithms" / "grpo.py").write_text(grpo_stub, encoding="utf-8")

    bad_launcher = tmp_path / "bad_run_nemo_rl.py"
    bad_launcher.write_text(
        "from nemo_rl.data.utils import setup_sft_data\n"
        "def launch_nemo_rl(export_cfg, yaml_path, *, algo=None):\n"
        "    if entry['stage'] == 'sft':\n"
        "        setup_res = setup_fn(config, tokenizer, dataset, val_dataset)\n"
        "        train_fn(*setup_res)\n"
        "    if entry['stage'] == 'dpo':\n"
        "        (p, cluster, dl, vdl, lf, lg, ck, st, mc) = setup_fn(config, tokenizer, dataset, val_dataset)\n"
        "        train_fn(policy=p, train_dataloader=cluster, val_dataloader=vdl, tokenizer=tokenizer, loss_fn=lf, master_config=mc, logger=lg, checkpointer=ck, dpo_save_state=st)\n",
        encoding="utf-8",
    )

    imp_errs = check_ast_imports(fake_fork, bad_launcher)
    assert any("setup_sft_data" in e for e in imp_errs)

    map_errs = check_setup_to_train_mapping(fake_fork, bad_launcher)
    assert any("Stage 'sft'" in e and "positional/starred" in e for e in map_errs)
    assert any("Stage 'dpo'" in e and "`cluster` from `setup()` erroneously passed" in e for e in map_errs)

    env_fork = os.environ.get("RL_FORK_SRC")
    if env_fork and Path(env_fork).is_dir():
        report = run_all_checks(Path(env_fork))
        assert report["ok"] is True, report["errors"]
