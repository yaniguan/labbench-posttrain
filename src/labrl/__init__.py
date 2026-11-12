"""`labrl`: End-to-End SFT-to-RLVR Post-Training Library for Scientific & Wet-Lab Observables.

Quickstart:
    >>> from labrl import PostTrainPipeline
    >>> manifest = PostTrainPipeline.from_methods(
    ...     ["sft", "dpo", "mpo", "ppo", "grpo"],
    ...     corpus="calisol23",
    ...     smoke=True,
    ... ).run()
"""
from __future__ import annotations

from labrl import functional, rewards
from labrl.core import (
    CORPUS_REGISTRY,
    ENV_REGISTRY,
    REWARD_TEMPLATE_REGISTRY,
    STAGE_REGISTRY,
    PipelineManifest,
    RewardPacket,
    SplitLeakageError,
    StageArtifact,
    compose_recipe,
    register_corpus,
    register_env,
    register_reward_template,
    register_stage,
    validate_composed_recipe,
    validate_nemo_rl_yaml,
    validate_sft_rl_pair,
)
from labrl.data import (
    BioProBenchCorpus,
    CALiSolCorpus,
    LabV0Corpus,
    ScientificCorpus,
    SplitGuard,
    make_corpus,
)
from labrl.engines import (
    ExecutionEngine,
    NeMoRLEngine,
    TorchLocalEngine,
    resolve_engine,
)
from labrl.envs import (
    BioProBenchEnv,
    CALiSolEnv,
    VerifiableEnv,
    make_verifiable_env,
)
from labrl.pipeline import PostTrainPipeline
from labrl.rewards import (
    CalibratedMCQTemplate,
    CounterfactualPairScorer,
    FaultDiagnosisRepairTemplate,
    GatedRewardTemplate,
    InverseDesignSupportTemplate,
    MolecularValiditySimilarityTemplate,
    PermutationOrderTemplate,
    ScalarObservableTemplate,
    SpectralPeakSetTemplate,
    StructuredActionPlanTemplate,
    VerifiedRewardTemplate,
    list_reward_templates,
    make_reward_template,
)
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

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "BioProBenchCorpus",
    "BioProBenchEnv",
    "CALiSolCorpus",
    "CALiSolEnv",
    "CORPUS_REGISTRY",
    "CalibratedMCQTemplate",
    "CounterfactualPairScorer",
    "DPOStage",
    "ENV_REGISTRY",
    "EvalStage",
    "ExecutionEngine",
    "FaultDiagnosisRepairTemplate",
    "GRPOStage",
    "GatedRewardTemplate",
    "InverseDesignSupportTemplate",
    "LabV0Corpus",
    "MPOStage",
    "MolecularValiditySimilarityTemplate",
    "NeMoRLEngine",
    "PPOStage",
    "PermutationOrderTemplate",
    "PipelineManifest",
    "PostTrainPipeline",
    "REWARD_TEMPLATE_REGISTRY",
    "RewardPacket",
    "SFTStage",
    "STAGE_REGISTRY",
    "ScalarObservableTemplate",
    "ScientificCorpus",
    "SpectralPeakSetTemplate",
    "SplitGuard",
    "SplitLeakageError",
    "Stage",
    "StageArtifact",
    "StructuredActionPlanTemplate",
    "TROPDStage",
    "TorchLocalEngine",
    "VerifiableEnv",
    "VerifiedRewardTemplate",
    "compose_recipe",
    "functional",
    "list_reward_templates",
    "make_corpus",
    "make_reward_template",
    "make_verifiable_env",
    "register_corpus",
    "register_env",
    "register_reward_template",
    "register_stage",
    "resolve_engine",
    "rewards",
    "validate_composed_recipe",
    "validate_nemo_rl_yaml",
    "validate_sft_rl_pair",
]
