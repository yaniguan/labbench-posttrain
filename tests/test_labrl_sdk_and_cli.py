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

    # 2. Same pipeline compiled & validated for 8xA100 cluster via NeMoRLEngine
    pipe_nemo = PostTrainPipeline(
        corpus="calisol23",
        stages=[GRPOStage(), MPOStage(), TROPDStage()],
        engine=NeMoRLEngine(hardware_axis="node_8xa100_colocated", model_axis="qwen3_8b"),
        base_model="Qwen/Qwen3-8B",
    )
    manifest_nemo = pipe_nemo.run(output_dir=tmp_path / "calisol_nemo")
    assert manifest_nemo.engine_name == "nemo"
    for s_name in ("grpo", "mpo", "tropd"):
        yaml_file = tmp_path / "calisol_nemo" / "nemo_configs" / f"{s_name}.yaml"
        assert yaml_file.exists()


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
