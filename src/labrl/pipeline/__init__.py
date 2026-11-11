"""First-class SFT-to-RL lifecycle orchestrator (`PostTrainPipeline`).

Automatically enforces:
  1. Cryptographic `SplitGuard` holdout invariance across all stages.
  2. Automatic freezing of Stage 1 `SFTStage` into the downstream KL reference anchor (`sft_reference`).
  3. Base-model compatibility verification across stage handoffs.
  4. Support for both sequential chains (`SFT -> MPO -> GRPO -> Eval`) and head-to-head benchmark forks
     (`branch_from_sft=True`: runs SFT once, then forks independent copies into DPO / MPO / PPO / GRPO).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from labrl.core.types import PipelineManifest, SplitLeakageError, StageArtifact
from labrl.data import ScientificCorpus, make_corpus
from labrl.engines import ExecutionEngine, resolve_engine
from labrl.envs import make_verifiable_env
from labrl.stages import (
    DPOStage,
    EvalStage,
    GRPOStage,
    MPOStage,
    PPOStage,
    SFTStage,
    Stage,
    TROPDStage,
)


class PostTrainPipeline:
    """End-to-end SFT -> Preference/Distill -> RLVR -> Scientific Evaluation orchestrator."""

    def __init__(
        self,
        *,
        corpus: str | ScientificCorpus,
        stages: Sequence[Stage],
        env: str | Any | None = None,
        base_model: str = "Qwen/Qwen3-1.7B",
        engine: str | ExecutionEngine = "auto",
        smoke: bool = False,
        branch_from_sft: bool = False,
        seed: int = 0,
    ) -> None:
        if not stages:
            raise ValueError("PostTrainPipeline requires at least one Stage")
        self.corpus = make_corpus(corpus)
        self.env_name = env if isinstance(env, str) else (env.name if env is not None else self.corpus.default_env_name())
        self.env = make_verifiable_env(env or self.env_name)
        self.stages = list(stages)
        self.base_model = base_model
        self.engine = resolve_engine(engine, smoke=smoke)
        self.smoke = bool(smoke)
        self.branch_from_sft = bool(branch_from_sft)
        self.seed = int(seed)

    @classmethod
    def from_methods(
        cls,
        methods: Sequence[str],
        *,
        corpus: str | ScientificCorpus = "calisol23",
        env: str | Any | None = None,
        base_model: str = "Qwen/Qwen3-1.7B",
        engine: str | ExecutionEngine = "auto",
        smoke: bool = False,
        include_eval: bool = True,
        seed: int = 0,
    ) -> "PostTrainPipeline":
        """Convenience builder constructing a head-to-head SFT-anchored comparison pipeline."""
        stage_map: dict[str, type[Stage]] = {
            "sft": SFTStage,
            "dpo": DPOStage,
            "mpo": MPOStage,
            "tropd": TROPDStage,
            "ppo": PPOStage,
            "grpo": GRPOStage,
        }
        norm = [m.strip().lower() for m in methods if m.strip()]
        stages: list[Stage] = [SFTStage()]
        for m in norm:
            if m == "sft":
                continue
            if m not in stage_map:
                raise ValueError(f"Unknown method {m!r}; choose from {sorted(stage_map)}")
            stages.append(stage_map[m]())
        if include_eval:
            stages.append(EvalStage())
        return cls(
            corpus=corpus,
            env=env,
            stages=stages,
            base_model=base_model,
            engine=engine,
            smoke=smoke,
            branch_from_sft=True,
            seed=seed,
        )

    def _verify_handoff(self, prev: StageArtifact | None, cur: StageArtifact) -> None:
        if cur.split_signature != self.corpus.split_guard.signature:
            raise SplitLeakageError(
                f"Stage {cur.stage_name!r} split signature {cur.split_signature} != corpus {self.corpus.split_guard.signature}"
            )
        if prev is not None and prev.base_model_name != cur.base_model_name:
            raise ValueError(
                f"Base-model mismatch across stages: {prev.base_model_name!r} -> {cur.base_model_name!r}"
            )

    def run(self, *, output_dir: str | Path | None = None) -> PipelineManifest:
        out_root = Path(output_dir) if output_dir else None
        if out_root:
            out_root.mkdir(parents=True, exist_ok=True)

        manifest = PipelineManifest(
            corpus_name=self.corpus.name,
            env_name=str(self.env_name),
            base_model_name=self.base_model,
            engine_name=self.engine.name,
            split_signature=self.corpus.split_guard.signature,
        )

        sft_anchor_art: StageArtifact | None = None
        last_art: StageArtifact | None = None

        for stage in self.stages:
            if stage.method == "eval" and self.branch_from_sft:
                # Evaluate every trained branch head-to-head on identical held-out splits
                targets = [a for a in manifest.artifacts.values() if a.method != "eval"]
                if not targets and sft_anchor_art is not None:
                    targets = [sft_anchor_art]
                last_eval_art: StageArtifact | None = None
                for target in targets:
                    eval_art = self.engine.run_stage(
                        stage,
                        corpus=self.corpus,
                        env=self.env,
                        base_model_name=self.base_model,
                        upstream=target,
                        output_dir=out_root,
                        seed=self.seed,
                    )
                    self._verify_handoff(target, eval_art)
                    manifest.evaluations[target.stage_name] = dict(eval_art.metrics)
                    last_eval_art = eval_art
                if last_eval_art is not None:
                    manifest.stages.append(last_eval_art.to_manifest_dict())
                    manifest.artifacts[stage.name] = last_eval_art
                continue

            upstream = (sft_anchor_art if (self.branch_from_sft and stage.method != "sft") else last_art)
            art = self.engine.run_stage(
                stage,
                corpus=self.corpus,
                env=self.env,
                base_model_name=self.base_model,
                upstream=upstream,
                output_dir=out_root,
                seed=self.seed,
            )
            self._verify_handoff(upstream, art)
            if stage.method == "sft" and sft_anchor_art is None:
                sft_anchor_art = art
            if stage.method == "eval":
                manifest.evaluations[last_art.stage_name if last_art else "base"] = dict(art.metrics)
            else:
                last_art = art
            manifest.stages.append(art.to_manifest_dict())
            manifest.artifacts[stage.name] = art

        if out_root:
            (out_root / "lineage_manifest.json").write_text(
                json.dumps(manifest.to_dict(), indent=2),
                encoding="utf-8",
            )
        return manifest


__all__ = ["PostTrainPipeline"]
