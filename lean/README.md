# `LabPostTrainLean` — Formal Anti-Hack RLVR, GDPO Pareto Simplex, Reverse-KL & SplitGuard (`L3`)

Self-contained **Lean 4** (`leanprover/lean4:v4.36.0`) verification kernel for `labbench-posttrain`, synchronized with the central Auto-Lab Trust Regime at [`yaniguan/auto-lab/tree/main/lean`](https://github.com/yaniguan/auto-lab/tree/main/lean) (`AutoLabLean/L3_PostTrain.lean`).

## Formalized Guardrails & Proved Theorems

| Component | Definitions | Machine-Checked Theorems |
| :--- | :--- | :--- |
| **Shared Diagnostics** | `RegimeViolation`, `firstViolation` | `firstViolation_nil`, `firstViolation_cons`, `firstViolation_none_iff` |
| **SplitGuard Holdout** | `splitGuardDisjoint`, `SplitManifest`, `checkSplitGuard` | `splitGuard_no_train_test_overlap`, `splitGuard_no_val_test_or_canary_overlap` |
| **Anti-Hack Rollout Gate** | `RolloutRewardSpec`, `gatedRewardPermille`, `checkRolloutGate` | `unverified_rollout_gets_zero_reward`, `checkRolloutGate_ok_iff` |
| **Twin-Delta Process Credit** | `ProcessStepSpec`, `stepProcessBonusPermille`, `frontierTraceDeltaSum`, `traceProcessBonusPermille` | `vacuous_step_gets_zero_process_bonus`, `frontierTraceDeltaSum_le_initial`, `unverified_trace_gets_zero_process_bonus` |
| **Multi-Objective GDPO Simplex** | `GDPOMultiRewardSpec`, `checkGDPOParetoWeights`, `gatedGDPOChannelsPermille`, `weightedDotNat`, `paretoNormalizedPermille`, `intListSum`, `groupCenteredScaled` | `checkGDPOParetoWeights_ok_iff`, `gdpo_unverified_all_channels_zero`, `weightedDotNat_le_sum_mul_max`, `gdpo_pareto_normalized_le_max`, `intListSum_affine_map`, `group_centered_scaled_sum_zero` |
| **Reverse-KL SFT-Anchor Bound** | `KLAnchorSpec`, `checkKLAnchor` | `checkKLAnchor_ok_iff`, `kl_anchor_drift_bound`, `kl_anchor_permille_le_div` |

## Build & Verify

```bash
cd lean
lake build
```

Zero external dependencies, `0 sorry`, `0 axiom`, and 20 compile-time `#guard` checks.
