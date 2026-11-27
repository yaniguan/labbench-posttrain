"""End-to-end verification of the `labrl` SFT-to-RLVR Python SDK and `labrl` CLI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch

import labrl
from labrl import (
    DPOStage,
    EvalStage,
    GRPOStage,
    MPOStage,
    NeMoRLEngine,
    PPOStage,
    PostTrainPipeline,
    RewardPacket,
    SFTStage,
    ScientificCorpus,
    SplitGuard,
    SplitLeakageError,
    TROPDStage,
    VerifiableEnv,
    functional,
    make_corpus,
    make_verifiable_env,
    register_corpus,
    register_env,
)
from labrl.cli import main as cli_main
from posttrain.hacking_checks import CANARY_NEEDLES


def test_builtin_corpora_and_split_guard() -> None:
    for cname in ("calisol23", "bioprobench", "lab_v0"):
        corpus = make_corpus(cname)
        assert len(corpus.split_guard.signature) == 16
        sft_rows = corpus.to_sft(split="train", limit=4)
        pref_rows = corpus.to_preference(split="train", limit=4)
        mpo_rows = corpus.to_mpo(split="train", limit=6)
        rl_rows = corpus.to_rl_prompts(split="train", limit=4)
        assert len(sft_rows) >= 2 and "prompt" in sft_rows[0] and "completion" in sft_rows[0]
        assert len(pref_rows) >= 2 and "chosen" in pref_rows[0] and "rejected" in pref_rows[0]
        assert len(mpo_rows) >= 2
        assert len(rl_rows) >= 2 and "prompt" in rl_rows[0]

    guard = SplitGuard(group_key="doi", val_groups=frozenset({"10.1000/val"}), test_groups=frozenset({"10.1000/test"}))
    guard.assert_disjoint(["10.1000/train1", "10.1000/train2"])
    with pytest.raises(SplitLeakageError):
        guard.assert_disjoint(["10.1000/train1", "10.1000/test"])
    with pytest.raises(SplitLeakageError):
        guard.assert_no_canary(["safe prompt", f"contaminated {CANARY_NEEDLES[1]}"])


def test_custom_verifiable_env_and_sequential_full_lifecycle(tmp_path: Path) -> None:
    @register_env("toy_electrochem_test")
    class ToyElectrochemEnv(VerifiableEnv):
        name = "toy_electrochem_test"

        def score_observable(self, info: dict[str, Any], completion: str) -> RewardPacket:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", completion)
            if not m:
                return RewardPacket(scalar=0.0, parsed=False, channels={"accuracy": 0.0, "validity": 0.0})
            pred = float(m.group(1))
            target = float(info.get("target_k", 12.0))
            acc = max(0.0, 1.0 - abs(pred - target) / max(target, 1.0))
            return RewardPacket(
                scalar=0.8 * acc + 0.2,
                parsed=True,
                channels={"accuracy": acc, "validity": 1.0},
                telemetry={"pred_k": pred},
            )

    @register_corpus("toy_electrochem_corpus_test")
    class ToyElectrochemCorpus(ScientificCorpus):
        name = "toy_electrochem_corpus_test"

        @property
        def split_guard(self) -> SplitGuard:
            return SplitGuard(group_key="cell_id", val_groups=frozenset({"v1"}), test_groups=frozenset({"t1"}))

        def default_env_name(self) -> str:
            return "toy_electrochem_test"

        def to_sft(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
            return [{"prompt": "Predict k (mS/cm) for 1M LiPF6 EC/DMC:", "completion": " <think>Fast transport.</think>11.8"}] * 4

        def to_preference(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
            return [{
                "prompt": "Predict k (mS/cm) for 1M LiPF6 EC/DMC:",
                "chosen": " <think>Optimal salt dissociation.</think>11.8",
                "rejected": " <think>Too viscous.</think>1.2",
                "record_type": "paired",
            }] * 4

        def to_mpo(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
            return self.to_preference()[:2] + [
                {"prompt": "Predict k:", "completion": " 11.8", "label": 1, "record_type": "unpaired"},
                {"prompt": "Predict k:", "completion": " 0.5", "label": 0, "record_type": "unpaired"},
            ]

        def to_rl_prompts(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
            return [{"prompt": "Predict k (mS/cm) for 1M LiPF6 EC/DMC:", "info": {"target_k": 11.8}}] * 2

    # 1. Verify NeMo-RL batched `step()` contract on custom 15-line VerifiableEnv
    env = make_verifiable_env("toy_electrochem_test")
    ret = env.step(
        [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "11.8"}]],
        [{"info": {"target_k": 11.8}}],
    )
    assert isinstance(ret.rewards, dict)
    assert set(ret.rewards.keys()) == {"reward/accuracy", "reward/validity"}
    assert torch.isclose(ret.rewards["reward/accuracy"][0], torch.tensor(1.0))

    # 2. Verify sequential SFT -> DPO -> MPO -> TROPD -> PPO -> GRPO -> Eval chain
    pipe = PostTrainPipeline(
        corpus="toy_electrochem_corpus_test",
        stages=[SFTStage(), DPOStage(), MPOStage(), TROPDStage(), PPOStage(), GRPOStage(), EvalStage()],
        smoke=True,
        branch_from_sft=False,
    )
    manifest = pipe.run(output_dir=tmp_path / "seq_run")
    assert len(manifest.stages) == 7
    assert "grpo" in manifest.evaluations
    assert manifest.artifacts["dpo"].metrics["reference_unchanged"] is True
    assert manifest.artifacts["mpo"].metrics["reference_unchanged"] is True
    assert manifest.artifacts["ppo"].metrics["sft_anchor_unchanged"] is True
    assert manifest.artifacts["grpo"].metrics["sft_anchor_unchanged"] is True
    assert (tmp_path / "seq_run" / "tropd.pt").is_file()
    assert (tmp_path / "seq_run" / "lineage_manifest.json").exists()


def test_calisol_branched_comparison_and_nemo_engine(tmp_path: Path) -> None:
    # 1. Branched head-to-head SFT -> {DPO, MPO, PPO, GRPO} -> Eval on TorchLocalEngine
    pipe_local = PostTrainPipeline.from_methods(
        ["sft", "dpo", "mpo", "ppo", "grpo"],
        corpus="calisol23",
        smoke=True,
    )
    manifest_local = pipe_local.run(output_dir=tmp_path / "calisol_local")
    assert set(manifest_local.evaluations.keys()) == {"sft", "dpo", "mpo", "ppo", "grpo"}
    for m_name, ev in manifest_local.evaluations.items():
        assert "test/scalar_reward_mean" in ev, m_name
        assert "test/parse_rate" in ev, m_name

    # 1b. LabV0Corpus end-to-end pipeline (#22)
    manifest_v0 = PostTrainPipeline.from_methods(
        ["sft", "mpo", "grpo"],
        corpus="lab_v0",
        smoke=True,
    ).run(output_dir=tmp_path / "lab_v0_local")
    assert set(manifest_v0.evaluations.keys()) == {"sft", "mpo", "grpo"}
    assert manifest_v0.evaluations["grpo"]["test/scalar_reward_mean"] >= 0.0

    # 2. Same pipeline compiled & validated for 8xA100 cluster via NeMoRLEngine (#23)
    pipe_nemo = PostTrainPipeline(
        corpus="calisol23",
        stages=[SFTStage(), DPOStage(), PPOStage(), GRPOStage(), GRPOStage(advantage="gspo")],
        engine=NeMoRLEngine(hardware_axis="node_8xa100_colocated", model_axis="qwen3_8b"),
        base_model="Qwen/Qwen3-8B",
    )
    manifest_nemo = pipe_nemo.run(output_dir=tmp_path / "calisol_nemo")
    assert manifest_nemo.engine_name == "nemo"
    for s_name, art in manifest_nemo.artifacts.items():
        assert art.metrics["status"] == "compiled"
    import yaml

    sft_yaml = yaml.safe_load((tmp_path / "calisol_nemo" / "nemo_configs" / "sft.yaml").read_text(encoding="utf-8"))
    dpo_yaml = yaml.safe_load((tmp_path / "calisol_nemo" / "nemo_configs" / "dpo.yaml").read_text(encoding="utf-8"))
    ppo_yaml = yaml.safe_load((tmp_path / "calisol_nemo" / "nemo_configs" / "ppo.yaml").read_text(encoding="utf-8"))
    assert "sft" in sft_yaml and "grpo" not in sft_yaml
    assert "dpo" in dpo_yaml and "grpo" not in dpo_yaml
    assert "ppo" in ppo_yaml and "grpo" not in ppo_yaml

    # Unmapped stage raises ValueError instead of silently emitting gdpo_multireward (#23)
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="Unsupported NeMo-RL stage"):
        PostTrainPipeline(
            corpus="calisol23",
            stages=[SimpleNamespace(name="rm", method="reward_model")],  # type: ignore[list-item]
            engine=NeMoRLEngine(hardware_axis="node_8xa100_colocated", model_axis="qwen3_8b"),
        ).run(output_dir=tmp_path / "bad_nemo")


def test_labrl_cli_and_functional_facade(tmp_path: Path) -> None:
    # Stateless functional facade sanity check
    adv = functional.gdpo_advantages(
        {
            "reward/a": torch.tensor([1.0, 0.0]),
            "reward/b": torch.tensor([0.0, 1.0]),
        },
        group_ids=torch.tensor([0, 0]),
        weights={"reward/a": 2.0, "reward/b": 1.0},
    )
    assert adv.shape == (2,) and float(adv[0]) > float(adv[1])

    # CLI 1: `labrl list`
    info = cli_main(["list"])
    assert info["version"] == labrl.__version__
    assert {"calisol23", "bioprobench", "lab_v0"}.issubset(set(info["corpora"]))
    assert "calisol_conductivity" in info["environments"]

    # CLI 2: `labrl pipeline --smoke`
    pipe_out = cli_main([
        "pipeline",
        "--corpus", "calisol23",
        "--methods", "sft,dpo,grpo",
        "--smoke",
        "--output-dir", str(tmp_path / "cli_pipe"),
    ])
    assert set(pipe_out["evaluations"].keys()) == {"sft", "dpo", "grpo"}
    assert (tmp_path / "cli_pipe" / "lineage_manifest.json").exists()

    # CLI 3: `labrl recipe --smoke --dump-yaml` + `labrl validate`
    dumped = tmp_path / "exported_calisol.yaml"
    rec_out = cli_main([
        "recipe",
        "--algo", "gdpo_multireward",
        "--domain", "calisol_conductivity",
        "--model", "qwen3_1.7b",
        "--hardware", "local_1gpu_colocated",
        "--smoke",
        "--dump-yaml", str(dumped),
    ])
    assert rec_out["status"] == "ran"
    assert dumped.exists()

    val_out = cli_main(["validate", str(dumped)])
    assert val_out["valid"] is True
    assert json.loads((tmp_path / "cli_pipe" / "lineage_manifest.json").read_text(encoding="utf-8"))["corpus_name"] == "calisol23"


def test_sequential_sft_mpo_gdpo_handoffs_and_calisol_channel_isolation(tmp_path: Path) -> None:
    """#49: Sequential SFT -> MPO -> GDPO chains MPO weights into GRPO while pinning KL anchor to SFT."""
    import yaml
    from labpt_nemo_rl import compose_recipe
    from scripts.posttrain.run_nemo_rl import main as run_nemo_main

    # 1. `from_methods(..., branch_from_sft=False)` and CLI `--sequential` chain SFT -> MPO -> GRPO
    pipe_seq = PostTrainPipeline.from_methods(
        ["sft", "mpo", "gdpo"],
        corpus="calisol23",
        smoke=True,
        branch_from_sft=False,
    )
    assert pipe_seq.branch_from_sft is False
    manifest_seq = pipe_seq.run(output_dir=tmp_path / "seq_sft_mpo_gdpo")
    assert list(manifest_seq.evaluations.keys()) == ["grpo"]
    assert manifest_seq.artifacts["mpo"].metrics["upstream_stage"] == "sft"
    assert manifest_seq.artifacts["grpo"].metrics["upstream_stage"] == "mpo"
    assert manifest_seq.artifacts["grpo"].metrics["upstream_checkpoint"] == manifest_seq.artifacts["mpo"].checkpoint_path
    assert manifest_seq.artifacts["grpo"].sft_anchor_path == manifest_seq.artifacts["sft"].sft_anchor_path
    assert manifest_seq.artifacts["grpo"].sft_reference is manifest_seq.artifacts["sft"].sft_reference
    assert manifest_seq.artifacts["grpo"].metrics["sft_anchor_unchanged"] is True

    cli_seq = cli_main([
        "pipeline",
        "--corpus", "calisol23",
        "--methods", "sft,mpo,gdpo",
        "--sequential",
        "--smoke",
        "--output-dir", str(tmp_path / "cli_seq"),
    ])
    assert list(cli_seq["evaluations"].keys()) == ["grpo"]
    grpo_stage_meta = next(s for s in cli_seq["stages"] if s["stage_name"] == "grpo")
    mpo_stage_meta = next(s for s in cli_seq["stages"] if s["stage_name"] == "mpo")
    sft_stage_meta = next(s for s in cli_seq["stages"] if s["stage_name"] == "sft")
    assert grpo_stage_meta["metrics"]["upstream_stage"] == "mpo"
    assert grpo_stage_meta["metrics"]["upstream_checkpoint"] == mpo_stage_meta["checkpoint_path"]
    assert grpo_stage_meta["sft_anchor_path"] == sft_stage_meta["sft_anchor_path"]

    # 2. NeMoRLEngine sequential SFT -> DPO -> GDPO wires actor_init_path (DPO) and reference_model_name (SFT),
    #    while MPO fails loudly on the yaniguan/RL fork backend (#52).
    pipe_nemo_mpo = PostTrainPipeline.from_methods(
        ["sft", "mpo", "gdpo"],
        corpus="calisol23",
        base_model="Qwen/Qwen3-8B",
        engine=NeMoRLEngine(hardware_axis="node_8xa100_colocated", model_axis="qwen3_8b"),
        branch_from_sft=False,
    )
    with pytest.raises(NotImplementedError, match="MPO"):
        pipe_nemo_mpo.run(output_dir=tmp_path / "nemo_mpo_fail")

    pipe_nemo_seq = PostTrainPipeline.from_methods(
        ["sft", "dpo", "gdpo"],
        corpus="calisol23",
        base_model="Qwen/Qwen3-8B",
        engine=NeMoRLEngine(hardware_axis="node_8xa100_colocated", model_axis="qwen3_8b"),
        branch_from_sft=False,
    )
    nemo_manifest = pipe_nemo_seq.run(output_dir=tmp_path / "nemo_seq")
    sft_ckpt = nemo_manifest.artifacts["sft"].checkpoint_path
    dpo_ckpt = nemo_manifest.artifacts["dpo"].checkpoint_path
    dpo_cfg = yaml.safe_load((tmp_path / "nemo_seq" / "nemo_configs" / "dpo.yaml").read_text(encoding="utf-8"))
    gdpo_cfg = yaml.safe_load((tmp_path / "nemo_seq" / "nemo_configs" / "grpo.yaml").read_text(encoding="utf-8"))
    assert dpo_cfg["policy"]["model_name"] == sft_ckpt
    assert dpo_cfg["policy"]["reference_model_name"] == sft_ckpt
    assert gdpo_cfg["policy"]["model_name"] == dpo_ckpt
    assert gdpo_cfg["policy"]["reference_model_name"] == sft_ckpt
    assert nemo_manifest.artifacts["sft"].metrics["nemo_entrypoint"] == "nemo_rl.algorithms.sft.sft_train"
    assert nemo_manifest.artifacts["dpo"].metrics["nemo_entrypoint"] == "nemo_rl.algorithms.dpo.dpo_train"
    assert nemo_manifest.artifacts["grpo"].metrics["nemo_entrypoint"] == "nemo_rl.algorithms.grpo.grpo_train"

    # 3. Domain-clean GDPO channel weights on CALiSol-23 (including reward/calibration, zero BioProBench pollution)
    calisol_gdpo = compose_recipe(
        algo="gdpo_multireward",
        domain="calisol_conductivity",
        model="qwen3_8b",
        hardware="node_8xa100_colocated",
    )
    assert set(calisol_gdpo["grpo"]["adv_estimator"]["gdpo_channel_weights"].keys()) == {
        "reward/conductivity",
        "reward/support",
        "reward/validity",
        "reward/calibration",
    }
    assert calisol_gdpo["grpo"]["adv_estimator"]["reward_weights"] == [0.6, 1.2, 0.8, 0.5]

    # 4. scripts/posttrain/run_nemo_rl.py stage-specific entrypoint dispatch for sft, dpo, and gdpo_multireward
    for algo_name, exp_stage in (("sft", "sft"), ("dpo", "dpo"), ("gdpo_multireward", "grpo")):
        res = run_nemo_main([
            "--algo", algo_name,
            "--domain", "calisol_conductivity",
            "--model", "qwen3_8b",
            "--hardware", "node_8xa100_colocated",
            "--init-checkpoint", "ckpts/dpo_step400",
            "--sft-anchor", "ckpts/sft_anchor",
            "--dry-run",
        ])
        assert res["summary"]["status"] == "dry_run_ok"
        assert res["summary"]["stage"] == exp_stage
        assert res["summary"]["actor_init"] == "ckpts/dpo_step400"
        if algo_name != "sft":
            assert res["summary"]["sft_anchor"] == "ckpts/sft_anchor"

    with pytest.raises(NotImplementedError, match="mpo_offline"):
        run_nemo_main([
            "--algo", "mpo_offline",
            "--domain", "calisol_conductivity",
            "--model", "qwen3_8b",
            "--hardware", "node_8xa100_colocated",
            "--dry-run",
        ])
