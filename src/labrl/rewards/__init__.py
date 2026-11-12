"""`labrl.rewards`: Anti-Hack Verified Scientific & Robotic Reward Templates.

Includes:
  - 8 canonical templates covering all 205 benchmarks in `docs/benchmark_map.md`
  - Dimensional physical unit normalizer (`convert_scientific_unit`)
  - Pure-Python SMILES & stoichiometry verifier (`parse_smiles_invariants`)
  - Hierarchical safety-gated & counterfactual combinators (`GatedRewardTemplate`, `CounterfactualPairScorer`)
  - Instant 1-line `.as_env()` conversion into dual-engine `VerifiableEnv`
"""
from __future__ import annotations

from labrl.rewards.composers import (
    CounterfactualPairScorer,
    GatedRewardTemplate,
    list_reward_templates,
    make_reward_template,
)
from labrl.rewards.extractors import (
    ExtractedAnswer,
    convert_scientific_unit,
    extract_verified_answer,
    normalize_unit_symbol,
    parse_smiles_invariants,
    parse_strict_json,
    smiles_similarity,
    strip_think_blocks,
    units_compatible,
)
from labrl.rewards.templates import (
    CalibratedMCQTemplate,
    FaultDiagnosisRepairTemplate,
    InverseDesignSupportTemplate,
    MolecularValiditySimilarityTemplate,
    PermutationOrderTemplate,
    ScalarObservableTemplate,
    SpectralPeakSetTemplate,
    StructuredActionPlanTemplate,
    TemplateVerifiableEnv,
    VerifiedRewardTemplate,
)

__all__ = [
    "CalibratedMCQTemplate",
    "CounterfactualPairScorer",
    "ExtractedAnswer",
    "FaultDiagnosisRepairTemplate",
    "GatedRewardTemplate",
    "InverseDesignSupportTemplate",
    "MolecularValiditySimilarityTemplate",
    "PermutationOrderTemplate",
    "ScalarObservableTemplate",
    "SpectralPeakSetTemplate",
    "StructuredActionPlanTemplate",
    "TemplateVerifiableEnv",
    "VerifiedRewardTemplate",
    "convert_scientific_unit",
    "extract_verified_answer",
    "list_reward_templates",
    "make_reward_template",
    "normalize_unit_symbol",
    "parse_smiles_invariants",
    "parse_strict_json",
    "smiles_similarity",
    "strip_think_blocks",
    "units_compatible",
]
