"""Regression test suite verifying fixes across audit issues (#21..#42) and pre-existing issues (#12..#20)."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# #15: LAB-Bench no-model shortcut baselines & tie-splitting rule
# ---------------------------------------------------------------------------
def test_labbench_shortcut_tie_rule_and_protocolqa_numbers():
    from labeval.shortcuts import compute_labbench_shortcuts, score_item_length_shortcut

    # Exact 3-way tie including ideal -> 1/3 credit; 2-way tie excluding ideal -> 0.0 credit
    assert score_item_length_shortcut("abcd", ["efgh", "ijkl", "xy"], mode="longest_chars") == pytest.approx(1.0 / 3.0)
    assert score_item_length_shortcut("ab", ["efgh", "ijkl"], mode="longest_chars") == pytest.approx(0.0)
    assert score_item_length_shortcut("a b c", ["x y z", "one"], mode="longest_words") == pytest.approx(0.5)

    report = compute_labbench_shortcuts()
    assert "meta" in report and "git_commit" in report["meta"]
    pqa = report["families"]["ProtocolQA"]
    assert pqa["n"] == 108
    assert pqa["longest_chars"] == pytest.approx(0.332, abs=1e-3)
    assert pqa["chance_with_refuse"] == pytest.approx(0.188, abs=1e-3)


# ---------------------------------------------------------------------------
# #16: NeMo-RL launcher dry-run, pinned commit, and GSPO schema flags
# ---------------------------------------------------------------------------
def test_nemo_rl_launcher_dry_run_and_gspo_recipe_schema(tmp_path: Path):
    from labpt_nemo_rl.config_validation import compose_recipe, validate_nemo_rl_yaml
    from scripts.posttrain.run_nemo_rl import PINNED_RL_COMMIT, main as run_nemo_main

    dumped = tmp_path / "gspo_composed.yaml"
    res = run_nemo_main([
        "--algo", "gspo_longcot",
        "--domain", "bioenvs_pqa_ord_err",
        "--model", "qwen3_1.7b",
        "--hardware", "local_1gpu_colocated",
        "--dump-yaml", str(dumped),
        "--dry-run",
    ])
    assert res["summary"]["status"] == "dry_run_ok"
    assert res["summary"]["pinned_rl_commit"] == PINNED_RL_COMMIT
    assert set(res["summary"]["registered_envs"]) == {"labbench_bioprobench", "labbench_calisol"}
    assert res["summary"]["first_batch_size"] > 0
    assert dumped.is_file()

    cfg = compose_recipe(
        algo="gspo_longcot",
        domain="bioenvs_pqa_ord_err",
        model="qwen3_1.7b",
        hardware="local_1gpu_colocated",
    )
    validate_nemo_rl_yaml(cfg, source="gspo_test")
    loss_fn = cfg["loss_fn"]
    assert loss_fn["sequence_level_importance_ratios"] is True
    assert loss_fn["token_level_loss"] is False
    assert loss_fn["sequence_level_importance_sampling_geometric_mean"] is True

    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert PINNED_RL_COMMIT in pyproject_text


# ---------------------------------------------------------------------------
# #17: Pass-rate prompt filter (CPU smoke + cached file + train-only guard)
# ---------------------------------------------------------------------------
def test_pass_rate_prompt_filter_keeps_and_drops_and_guards_splits(tmp_path: Path):
    from bioenvs import make_env
    from scripts.posttrain.pass_rate_filter import (
        load_kept_ids,
        main as filter_main,
        make_pass_rate_filter_fn,
    )

    fake_file = tmp_path / "fake_pass_rates.jsonl"
    fake_rows = [
        {"env": "ord", "split": "train", "id": "keep_mid", "mean_reward": 0.50, "std_reward": 0.25},
        {"env": "ord", "split": "train", "id": "drop_easy", "mean_reward": 0.98, "std_reward": 0.05},
        {"env": "ord", "split": "train", "id": "drop_hard", "mean_reward": 0.02, "std_reward": 0.05},
        {"env": "ord", "split": "train", "id": "drop_zero_var", "mean_reward": 0.50, "std_reward": 0.0},
    ]
    fake_file.write_text("\n".join(json.dumps(r) for r in fake_rows) + "\n", encoding="utf-8")
    assert load_kept_ids(fake_file) == {"keep_mid"}

    fn = make_pass_rate_filter_fn(fake_file, split="train", env_name="ord")
    assert fn({"id": "keep_mid"}) is True
    assert fn({"id": "drop_easy"}) is False
    assert fn({"id": "drop_hard"}) is False
    assert fn({"id": "drop_zero_var"}) is False

    # Never filter val or test!
    with pytest.raises(ValueError, match="split='train'"):
        make_pass_rate_filter_fn(fake_file, split="val")
    with pytest.raises(ValueError, match="train-only"):
        make_env("ord", split="test", filter_fn=fn)

    # Cached completions run + tiny CPU smoke run
    cached_in = tmp_path / "cached_in.jsonl"
    cached_in.write_text(
        json.dumps({"env": "ord", "split": "train", "id": "p1", "rewards": [0.0, 0.5, 1.0, 0.5]}) + "\n"
        + json.dumps({"env": "ord", "split": "train", "id": "p2", "rewards": [1.0, 1.0, 1.0, 1.0]}) + "\n",
        encoding="utf-8",
    )
    cached_res = filter_main(["--cached", str(cached_in), "--out", str(tmp_path / "cached_out.jsonl")])
    assert cached_res["summary"]["per_env"]["ord"]["kept_fraction"] == pytest.approx(0.5)

    smoke_res = filter_main([
        "--envs", "ord", "err",
        "--smoke",
        "--k", "2",
        "--max-items", "3",
        "--out", str(tmp_path / "smoke_filter.jsonl"),
    ])
    assert set(smoke_res["summary"]["per_env"].keys()) == {"ord", "err"}
    assert (tmp_path / "smoke_filter.summary.json").is_file()


# ---------------------------------------------------------------------------
# #18: 3-seed GRPO vs GSPO + reward controls runner (--dry-run and --smoke)
# ---------------------------------------------------------------------------
def test_grpo_vs_gspo_3seeds_dry_run_and_cpu_smoke(tmp_path: Path):
    mod = _load_script_module("run_grpo_vs_gspo_3seeds", "scripts/run_grpo_vs_gspo_3seeds.py")

    dry = mod.main(["--dry-run", "--out-dir", str(tmp_path / "dry")])
    assert dry["summary"]["status"] == "not run"
    assert dry["summary"]["gpu_results"] == "PENDING"
    assert all(v["mean_reward"] == "not run" for v in dry["summary"]["arms"].values())

    smoke = mod.main(["--smoke", "--steps", "1", "--group-size", "2", "--out-dir", str(tmp_path / "smoke")])
    assert smoke["summary"]["gpu_results"] == "PENDING"
    assert len(smoke["runs"]) == 12  # 4 arms x 3 seeds
    assert (tmp_path / "smoke" / "grpo_true_seed0.json").is_file()
    assert (tmp_path / "smoke" / "gspo_true_seed2.json").is_file()
    assert "meta" in smoke["summary"]


# ---------------------------------------------------------------------------
# #19: Run provenance metadata (`common/run_meta.py`) in result JSON builders
# ---------------------------------------------------------------------------
def test_run_metadata_in_analyze_and_eval_suite(tmp_path: Path):
    from common.run_meta import build_run_meta
    from labeval import analyze as labeval_analyze
    from posttrain.eval_suite import build_report

    meta = build_run_meta(data_files=[REPO_ROOT / "pyproject.toml"], prompt_template="Q: {q}", reward_version="v3")
    required_keys = {
        "utc_timestamp",
        "git_commit",
        "git_dirty",
        "reward_version",
        "sampling",
        "prompt_template_sha256",
        "data_sha256",
        "library_versions",
    }
    assert required_keys.issubset(set(meta.keys()))

    rep = build_report(rows=[{"correct": False, "task": "pqa", "length": 4}])
    assert required_keys.issubset(set(rep["meta"].keys()))

    # labeval.analyze.main writes labbench_summary_test.json with a populated "meta" block
    fake_jsonl = tmp_path / "labbench_haiku-5.5_seed0.jsonl"
    fake_jsonl.write_text(
        json.dumps({
            "id": "q1",
            "family": "ProtocolQA",
            "subset": "ProtocolQA",
            "model": "haiku-5.5",
            "seed": 0,
            "n_choices": 5,
            "correct": True,
            "sure": True,
            "parse_fail": False,
            "finish_reason": "stop",
            "pred": "A",
            "gold": "A",
            "usage": {"prompt": 10, "completion": 5, "cost": 0.001},
        }) + "\n",
        encoding="utf-8",
    )
    labeval_analyze.main(["--families", "ProtocolQA", "--tag", "test"], results_dir=tmp_path)
    summary = json.loads((tmp_path / "labbench_summary_test.json").read_text(encoding="utf-8"))
    assert "meta" in summary and required_keys.issubset(set(summary["meta"].keys()))


# ---------------------------------------------------------------------------
# #20: Reward exploit audit thresholds fail closed when a reward opens a hole
# ---------------------------------------------------------------------------
def test_bioenvs_audit_thresholds_pass_and_fail_when_patched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from bioenvs import audit
    from bioenvs.base import Reward
    from bioenvs.err import ERR

    res = audit.audit(n=200)
    thresholds = audit.load_thresholds()
    assert audit.check_thresholds(res, thresholds) == []
    # Every threshold entry must carry a non-empty reason (#20)
    for env_name, block in thresholds["environments"].items():
        assert isinstance(block["oracle"].get("reason"), str) and len(block["oracle"]["reason"].strip()) > 5
        for pol, spec in (block.get("exploits") or {}).items():
            assert isinstance(spec.get("reason"), str) and len(spec["reason"].strip()) > 5, (env_name, pol)

    # Patch ERR.score to pay 1.0 unconditionally -> threshold checker MUST flag violation!
    monkeypatch.setattr(ERR, "score", lambda self, sample, completion: Reward(1.0, True, {"detect": 1.0}))
    broken_res = audit.audit(n=40)
    failures = audit.check_thresholds(broken_res, thresholds)
    assert any("err" in f and "always True" in f for f in failures)


# ---------------------------------------------------------------------------
# #24: Actuator type/bounds validation in labpt.rewards
# ---------------------------------------------------------------------------
def test_labpt_rewards_rejects_string_bool_nan_and_out_of_range_params():
    from labpt.rewards import ordered_action_reward, schema_valid_actions

    bad_str = json.dumps([{"action": "aspirate", "params": {"volume_uL": "5000", "well": "A1", "labware": "p"}}])
    bad_bool = json.dumps([{"action": "heat", "params": {"temperature_C": True, "duration_s": 10}}])
    assert schema_valid_actions(bad_str) == 0.0
    assert schema_valid_actions(bad_bool) == 0.0
    good = json.dumps([{"action": "heat", "params": {"temperature_C": 37.0, "duration_s": 10.0}}])
    assert ordered_action_reward(bad_bool, good)["param_precision"] == 0.0


# ---------------------------------------------------------------------------
# #27: tokenizers_share_vocab rejects distinct equal-sized vocabularies
# ---------------------------------------------------------------------------
def test_distill_rejects_permuted_equal_size_tokenizers():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from scientist.distill import distill_step_loss, tokenizers_share_vocab

    def _make_tok(vocab: dict[str, int]) -> PreTrainedTokenizerFast:
        t = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        t.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")

    tok_a = _make_tok({"<pad>": 0, "<eos>": 1, "<unk>": 2, "alpha": 3, "beta": 4})
    tok_b = _make_tok({"<pad>": 0, "<eos>": 1, "<unk>": 2, "beta": 3, "alpha": 4})  # swapped IDs 3 & 4!
    assert len(tok_a) == len(tok_b) == 5
    assert tokenizers_share_vocab(tok_a, tok_b) is False
    assert tokenizers_share_vocab(tok_a, tok_a) is True

    s_logits = torch.randn(1, 3, 5, requires_grad=True)
    t_logits = torch.randn(1, 3, 5)
    ids = torch.tensor([[3, 4, 1]])
    mask = torch.tensor([[0, 1, 1]])
    with pytest.raises(ValueError, match="Incompatible vocab"):
        distill_step_loss(s_logits, t_logits, ids, mask, student_tokenizer=tok_a, teacher_tokenizer=tok_b)


# ---------------------------------------------------------------------------
# #29 & #30: TROPD KL guard + PPO checkpoint value_head & step-seeded rollouts
# ---------------------------------------------------------------------------
def test_tropd_requires_kl_and_ppo_saves_and_resumes_value_head(tmp_path: Path):
    from posttrain.distill_opd import train_tropd
    from posttrain.ppo import train_ppo
    from posttrain.rewards import toy_protocol_reward
    from posttrain.tiny import build_tiny_lm

    model, tok = build_tiny_lm(seed=4)
    teacher, _ = build_tiny_lm(seed=5)
    with pytest.raises(ValueError, match="non-negative"):
        train_tropd(
            [{"prompt": "Heat sample"}],
            toy_protocol_reward,
            teachers=[teacher],
            model=model,
            tokenizer=tok,
            kl_coef=-0.01,
            steps=1,
        )

    ckpt_t = tmp_path / "tropd_ckpt.pt"
    res_t = train_tropd(
        [{"prompt": "Heat sample"}],
        toy_protocol_reward,
        teachers=[teacher],
        model=model,
        tokenizer=tok,
        kl_coef=0.02,
        steps=1,
        group_size=2,
        max_new_tokens=3,
        checkpoint_path=str(ckpt_t),
    )
    assert ckpt_t.is_file() and res_t["reference"] is not None

    # PPO checkpoint must persist value_head weights and step-dependent random rewards (#30)
    m1, tok1 = build_tiny_lm(seed=9)
    ckpt_ppo = tmp_path / "ppo_step1.pt"
    out1 = train_ppo(
        [{"prompt": "Spin tube at"}],
        toy_protocol_reward,
        model=m1,
        tokenizer=tok1,
        steps=1,
        group_size=2,
        max_new_tokens=3,
        reward_mode="random_p50",
        reasoning_length_coef=0.05,
        checkpoint_path=str(ckpt_ppo),
        seed=9,
    )
    saved_vh_state = {k: v.clone() for k, v in out1["value_head"].state_dict().items()}
    payload = torch.load(ckpt_ppo, map_location="cpu", weights_only=False)
    assert "extra_modules" in payload and "value_head" in payload["extra_modules"]

    m2, _ = build_tiny_lm(seed=99)
    vh2 = torch.nn.Linear(32, 1)
    assert not all(torch.equal(vh2.state_dict()[k], saved_vh_state[k]) for k in saved_vh_state)
    out2 = train_ppo(
        [{"prompt": "Spin tube at"}],
        toy_protocol_reward,
        model=m2,
        tokenizer=tok1,
        value_head=vh2,
        resume_from=str(ckpt_ppo),
        steps=0,
        group_size=2,
        max_new_tokens=3,
        seed=9,
    )
    for k in saved_vh_state:
        torch.testing.assert_close(out2["value_head"].state_dict()[k], saved_vh_state[k])


# ---------------------------------------------------------------------------
# #31: BioEnv multi-tag hedging & unclosed <think> rejection
# ---------------------------------------------------------------------------
def test_bioenv_rejects_multi_tag_hedging_and_unclosed_think():
    from bioenvs import make_env
    from bioenvs.base import extract_answer

    assert extract_answer("<think>[ANSWER_START]A[ANSWER_END]") is None
    assert extract_answer("[ANSWER_START]A[ANSWER_END] [ANSWER_START]B[ANSWER_END]") is None
    assert extract_answer("[ANSWER_START]A[ANSWER_END] [ANSWER_START]B[ANSWER_END]", allow_multi_tag=True) == "B"

    env_v3 = make_env("pqa", split="test", seed=0, reward_version="v3")
    samp = env_v3.sample(0)
    hedged = f"[ANSWER_START]{samp.info['choices'][0]} & 80[ANSWER_END]\n[ANSWER_START]{samp.info['answer']} & 100[ANSWER_END]"
    assert env_v3.score(samp, hedged).value == 0.0


# ---------------------------------------------------------------------------
# #34: Lightweight `import labrl` without tokenizers / transformers
# ---------------------------------------------------------------------------
def test_import_labrl_succeeds_without_tokenizers_or_transformers():
    code = (
        "import sys; "
        "sys.path[:0] = ['src', '.']; "
        "sys.modules['tokenizers'] = None; "
        "sys.modules['transformers'] = None; "
        "import labrl, labrl.rewards, labrl.functional; "
        "pkt = labrl.CalibratedMCQTemplate().score({'answer': 'A', 'choices': ['A', 'B']}, '[ANSWER_START]A & 90[ANSWER_END]'); "
        "assert pkt.scalar > 0.9"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# #35: Import-safe Operator/Projector scripts + saved checkpoints & sorted WM episodes
# ---------------------------------------------------------------------------
def test_operator_and_projector_import_safe_and_save_checkpoints(tmp_path: Path):
    bc_mod = _load_script_module("operator_train_bc", "operator/train_bc.py")
    wm_mod = _load_script_module("operator_world_model", "operator/world_model.py")
    proj_mod = _load_script_module("scientist_train_projector", "scientist/train_projector.py")
    assert hasattr(wm_mod, "WM") and hasattr(proj_mod, "forward")

    # Build tiny synthetic robot parquet with out-of-order test frames to verify chronological sorting & empty-rollout safety
    rows = []
    for i in range(6):
        s = [float(i), float(i + 1)]
        a = [1.0]
        ns = [float(i + 1), float(i + 2)]
        rows.append({"repo": "toy_bot", "episode": "ep_tr", "frame": i, "split": "train", "state": s, "action": a, "next_state": ns})
    # Deliberately scramble test frame order (3, 1, 2, 0)
    for i in (3, 1, 2, 0):
        s = [float(i), float(i + 1)]
        rows.append({"repo": "toy_bot", "episode": "ep_te", "frame": i, "split": "test", "state": s, "action": [1.0], "next_state": [float(i + 1), float(i + 2)]})
    pq_path = tmp_path / "frames.parquet"
    pd.DataFrame(rows).to_parquet(pq_path)

    bc_cfg = tmp_path / "bc.yaml"
    bc_out = tmp_path / "bc_out"
    bc_cfg.write_text(yaml.safe_dump({
        "device": "cpu", "frames": str(pq_path), "repo": "toy_bot",
        "hidden": 16, "lr": 1e-3, "steps": 2, "batch_size": 2, "output_dir": str(bc_out),
    }), encoding="utf-8")
    bc_res = bc_mod.main([str(bc_cfg)])
    assert (bc_out / "checkpoint.pt").is_file()
    bc_ckpt = torch.load(bc_out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert {"state_dict", "mu", "sd", "state_dim", "action_dim"}.issubset(bc_ckpt.keys())
    assert np.isfinite(bc_res["test_mse"])

    # World model with rollout_k=50 (> episode length 4) must emit valid JSON (None, not NaN!) and save checkpoint.pt
    wm_cfg = tmp_path / "wm.yaml"
    wm_out = tmp_path / "wm_out"
    wm_cfg.write_text(yaml.safe_dump({
        "device": "cpu", "frames": str(pq_path), "repo": "toy_bot",
        "hidden": 16, "lr": 1e-3, "steps": 2, "batch_size": 2, "rollout_k": 50, "output_dir": str(wm_out),
    }), encoding="utf-8")
    wm_res = wm_mod.main([str(wm_cfg)])
    assert wm_res["rollout_norm_mse"] is None and wm_res["n_rollouts"] == 0
    assert (wm_out / "checkpoint.pt").is_file()
    json.loads((wm_out / "metrics.json").read_text(encoding="utf-8"))  # valid strict JSON!


# ---------------------------------------------------------------------------
# #36 & #37: Non-overlapping SFT/DPO batching + balanced MPO BCO & label validation
# ---------------------------------------------------------------------------
def test_sft_non_overlapping_batches_and_mpo_balanced_bco():
    from posttrain.mpo import mpo_loss, train_mpo
    from posttrain.sft import train_sft
    from posttrain.tiny import build_tiny_lm

    model, tok = build_tiny_lm(seed=2)
    examples = [{"prompt": f"Step {i}:", "completion": f" Action {i}."} for i in range(6)]
    res = train_sft(examples, model=model, tokenizer=tok, steps=3, batch_size=2, seed=2)
    assert res["status"] == "ran" and len(res["losses"]) == 3

    # Balanced BCO check: when pos term = -logsigmoid(0) = log(2) and neg term = -logsigmoid(0) = log(2),
    # 0.5 * (pos + neg) == log(2) (~0.6931), NOT 2 * log(2) (~1.3863)!
    lp_pos = torch.tensor([-1.0], requires_grad=True)
    ref_pos = torch.tensor([-1.0])
    lp_neg = torch.tensor([-1.0], requires_grad=True)
    ref_neg = torch.tensor([-1.0])
    loss, metrics = mpo_loss(
        logp_chosen=None,
        logp_rejected=None,
        ref_chosen=None,
        ref_rejected=None,
        logp_unpaired_pos=lp_pos,
        ref_unpaired_pos=ref_pos,
        logp_unpaired_neg=lp_neg,
        ref_unpaired_neg=ref_neg,
        reward_shift=0.0,
        preference_weight=0.0,
        bco_weight=1.0,
        sft_weight=0.0,
    )
    assert float(loss.item()) == pytest.approx(float(np.log(2.0)), rel=1e-5)
    assert metrics["bco_loss"] == pytest.approx(float(np.log(2.0)), rel=1e-5)

    # Unlabeled unpaired record raises ValueError instead of silently defaulting to negative label=0 (#37)
    m_mpo, _ = build_tiny_lm(seed=3)
    with pytest.raises(ValueError, match="binary label"):
        train_mpo(
            [{"prompt": "Wash cells", "completion": " with PBS", "record_type": "unpaired"}],
            model=m_mpo,
            tokenizer=tok,
            steps=1,
        )


# ---------------------------------------------------------------------------
# #38 & #39: check_gpu H100 support, smoke_all python fallback, LLM 401 fail-fast & run lock
# ---------------------------------------------------------------------------
def test_check_gpu_h100_and_llm_4xx_fail_fast_and_empty_redo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from common.llm import LLM
    from labpt.config import check_gpu

    # H100 passes check_gpu; small 8 GB consumer card fails check_gpu (#38)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx=0: SimpleNamespace(total_memory=80 * (1024 ** 3)))
    check_gpu({"device": "a100"})

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA GeForce RTX 3070")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx=0: SimpleNamespace(total_memory=8 * (1024 ** 3)))
    with pytest.raises(AssertionError, match="High-VRAM"):
        check_gpu({"device": "a100"})

    # smoke_all resolves a valid existing Python executable on Windows/POSIX (#38)
    smoke_mod = _load_script_module("smoke_all", "scripts/smoke_all.py")
    py_bin = smoke_mod._resolve_python_executable()
    assert Path(py_bin).exists()

    # LLM.complete fails fast (1 attempt, 0 sleeps) on HTTP 401 unauthorized (#39)
    calls = {"post": 0, "sleep": 0}

    async def _run_401():
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-dummy-key")
        llm = LLM("haiku-5.5", use_cache=False)

        async def fake_post(url: str, **kw: Any):
            calls["post"] += 1
            req = httpx.Request("POST", url)
            return httpx.Response(401, json={"error": {"message": "Invalid API key", "code": 401}}, request=req)

        async def fake_sleep(sec: float):
            calls["sleep"] += 1

        monkeypatch.setattr(llm.client, "post", fake_post)
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        res = await llm.complete("Hello")
        await llm.aclose()
        return res

    out_401 = asyncio.run(_run_401())
    assert calls["post"] == 1 and calls["sleep"] == 0
    assert "401" in str(out_401["error"])

    # labeval.run handles empty existing JSONL under --redo-truncated without IndexError (#39)
    from labeval import run as labeval_run

    monkeypatch.setattr(labeval_run, "RESULTS", tmp_path)
    empty_out = tmp_path / "labbench_haiku-5.5_seed0.jsonl"
    empty_out.write_text("", encoding="utf-8")

    class _DummyLLM:
        def __init__(self, *a: Any, **kw: Any) -> None:
            self.cost = 0.0
            self.tokens = {"prompt": 1, "completion": 1, "reasoning": 0}

        async def complete(self, *a: Any, **kw: Any) -> dict[str, Any]:
            return {"text": "[ANSWER]A[/ANSWER]", "finish_reason": "stop", "usage": {"prompt": 1, "completion": 1, "cost": 0.0}, "error": None}

        async def aclose(self) -> None:
            return None

    fake_item = SimpleNamespace(
        id="i1", family="ProtocolQA", subset="ProtocolQA", question="Q?", prompt_question="Q?",
        choices=["x", "y"], answer="A", unsure="B", images=[],
    )
    monkeypatch.setattr(labeval_run, "LLM", _DummyLLM)
    monkeypatch.setattr(labeval_run, "load", lambda seed, families=None, limit=None: [fake_item])
    args = SimpleNamespace(redo_truncated=True, families=["ProtocolQA"], limit=1, max_tokens=64, concurrency=2)
    asyncio.run(labeval_run.run_model("haiku-5.5", 0, args))
    written = [json.loads(line) for line in empty_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(written) == 1 and written[0]["id"] == "i1"


# ---------------------------------------------------------------------------
# #40: Chemical abbreviation protection in protocol_parser & no yield leak in safety_failure
# ---------------------------------------------------------------------------
def test_protocol_parser_abbreviations_and_safety_failure_no_yield_leak():
    from labpt.protocol_parser import parse_procedure
    from labpt.schema import Annotation, Outcome, Record
    from labpt.tasks import safety_failure

    steps = parse_procedure("Quench with sat. NH4Cl (10 mL) and stir for 15 min. Heat to 80 C for 2 h and filter.")
    actions = [(s.action, s.params) for s in steps]
    # 'sat. NH4Cl (10 mL)' stays attached to 'quench' in the first sentence instead of cleaving on 'sat.'
    quench_params = next(p for name, p in actions if name == "quench")
    assert "10 mL" in quench_params.get("amounts", [])
    # In 'Heat to 80 C for 2 h and filter', 80 C binds to 'heat', NOT 'filter'!
    heat_params = next(p for name, p in actions if name == "heat")
    filter_params = next(p for name, p in actions if name == "filter")
    assert heat_params.get("temperature_C") == pytest.approx(80.0)
    assert "temperature_C" not in filter_params

    rec = Record(
        id="r1",
        source="ord",
        source_id="ord_1",
        license="CC0-1.0",
        split="train",
        modality="reaction",
        provenance="measurement",
        protocol_text="Add conc. HCl dropwise and heat to 60 C.",
        annotations=[Annotation(key="safety_level", value="medium", annotator="rule:v0", auto=True)],
        outcome=Outcome(success=False, value=0.0, units="%", metric="yield"),
    )
    ex = safety_failure(rec)
    assert ex is not None
    prompt_str = json.dumps(ex["prompt"])
    assert "reported yield" not in prompt_str
    assert "0.0%" not in prompt_str


# ---------------------------------------------------------------------------
# #41: BioProBench SplitGuard on test, eval_suite zero-accuracy SE, contamination guards
# ---------------------------------------------------------------------------
def test_bioprobench_split_guard_eval_zero_acc_and_contamination_bounds():
    from labrl import SplitLeakageError, make_corpus
    from posttrain.contamination import option_order, prefix_fractions
    from posttrain.eval_suite import build_report

    bio_corpus = make_corpus("bioprobench")
    with pytest.raises(SplitLeakageError):
        bio_corpus.to_sft(split="test", limit=2)
    with pytest.raises(SplitLeakageError):
        bio_corpus.to_preference(split="test", limit=2)
    with pytest.raises(SplitLeakageError):
        bio_corpus.to_mpo(split="test", limit=2)

    # Zero accuracy (0.0) must produce noise_note with SE=0.000, NOT coin-flip p=0.50 SE!
    from posttrain.eval_suite import noise_note

    zero_rep = build_report(rows=[{"correct": False, "task": "pqa", "length": 5} for _ in range(10)])
    assert zero_rep["accuracy"]["mean"] == 0.0
    assert zero_rep["noise"] == noise_note(10, 0.0)
    assert zero_rep["noise"] != noise_note(10, 0.5)

    # option_order rejects >26 options cleanly; prefix_fractions("") returns empty list
    with pytest.raises(ValueError, match="26"):
        option_order([f"opt_{i}" for i in range(27)], "opt_0")
    assert prefix_fractions("") == []


# ---------------------------------------------------------------------------
# #42: L3 <-> L4/L6 typed interchange contracts & scripts/train_grpo.py fixes
# ---------------------------------------------------------------------------
def test_rehearsal_protocol_interchange_epistemic_prediction_and_train_grpo_helpers():
    from labpt.actions import from_rehearsal_protocol, to_rehearsal_protocol
    from labrl import EpistemicPrediction, RewardPacket
    from scripts.train_grpo import _text, r_overlap

    l3_plan = [
        {"action": "aspirate", "params": {"volume_uL": 50.0, "well": "A1", "labware": "plate"}},
        {"action": "dispense", "params": {"volume_uL": 50.0, "well": "B2", "labware": "plate"}},
    ]
    rehearsal_ops = to_rehearsal_protocol(l3_plan, auto_tips=True)
    assert [op["op"] for op in rehearsal_ops] == ["pick_up_tip", "aspirate", "dispense", "drop_tip"]
    assert rehearsal_ops[1] == {"op": "aspirate", "loc": "plate:A1", "vol": 50.0}
    roundtrip = from_rehearsal_protocol(rehearsal_ops)
    assert [a.model_dump() for a in roundtrip] == l3_plan

    pkt = RewardPacket(
        scalar=0.92,
        parsed=True,
        channels={"reward/accuracy": 0.95, "reward/calibration": 0.85},
        telemetry={"estimated_k_mS_cm": 11.4, "sigma_mS_cm": 0.5, "units": "mS/cm"},
    )
    ep = pkt.to_epistemic_prediction(z_score=2.0, rationale_spans=["VTF viscosity regime"])
    assert isinstance(ep, EpistemicPrediction)
    assert ep.value == pytest.approx(11.4)
    assert ep.sigma == pytest.approx(0.5)
    assert ep.interval_lo == pytest.approx(10.4)
    assert ep.interval_hi == pytest.approx(12.4)
    assert ep.abstain is False
    assert ep.to_dict()["rationale_spans"] == ["VTF viscosity regime"]

    # scripts/train_grpo.py _text reads c[-1]["content"] safely and r_overlap penalizes reversed order
    assert _text([]) == ""
    assert _text([{"role": "system", "content": "sys"}, {"role": "assistant", "content": "final"}]) == "final"
    ref_json = json.dumps(l3_plan)
    rev_json = json.dumps(list(reversed(l3_plan)))
    scores = r_overlap([ref_json, rev_json], ["protocol2actions", "protocol2actions"], [ref_json, ref_json])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < scores[0] - 0.2
