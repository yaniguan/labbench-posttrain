# `LabPostTrainLean` — Formal Anti-Hack RLVR, GDPO Standardization, Reverse-KL & SplitGuard (`L3`)

Self-contained **Lean 4** (`leanprover/lean4:v4.36.0`) verification kernel for `labbench-posttrain`. Sections 1–3 correspond to the base Layer-L3 specification in [`yaniguan/auto-lab/tree/main/lean`](https://github.com/yaniguan/auto-lab/tree/main/lean) (`AutoLabLean/L3_PostTrain.lean`), while Sections 4–5 add local `labbench-posttrain` extensions for multi-objective GDPO (both raw-channel simplex bounds and within-group standardized GDPO conservation) and Reverse-KL SFT-anchor trust-region bounds.

## Formalized Guardrails & Proved Theorems

| Component | Definitions | Machine-Checked Theorems |
| :--- | :--- | :--- |
| **Shared Diagnostics** | `RegimeViolation`, `firstViolation` | `firstViolation_nil`, `firstViolation_cons`, `firstViolation_none_iff` |
| **SplitGuard Holdout** | `splitGuardDisjoint`, `SplitManifest`, `checkSplitGuard` | `splitGuard_no_train_test_overlap`, `splitGuard_no_val_test_or_canary_overlap` |
| **Anti-Hack Rollout Gate** | `RolloutRewardSpec`, `gatedRewardPermille`, `checkRolloutGate` | `unverified_rollout_gets_zero_reward`, `checkRolloutGate_ok_iff` |
| **Twin-Delta Process Credit** | `ProcessStepSpec`, `stepProcessBonusPermille`, `frontierTraceDeltaSum`, `traceProcessBonusPermille` | `vacuous_step_gets_zero_process_bonus`, `frontierTraceDeltaSum_le_initial`, `unverified_trace_gets_zero_process_bonus` |
| **Multi-Objective GDPO (Raw Simplex & Standardized Z-Score)** | `GDPOMultiRewardSpec`, `checkGDPOParetoWeights`, `gatedGDPOChannelsPermille`, `weightedDotNat`, `paretoNormalizedPermille`, `intListSum`, `groupCenteredScaled`, `gdpoChannelZScoreScaled`, `zipAddInt`, `gdpoStandardizedGroupSum` | `checkGDPOParetoWeights_ok_iff`, `gdpo_unverified_all_channels_zero`, `weightedDotNat_le_sum_mul_max`, `raw_channel_pareto_normalized_le_max` (`gdpo_pareto_normalized_le_max`), `intListSum_affine_map`, `group_centered_scaled_sum_zero`, `gdpo_zero_spread_channel_zero`, `gdpo_channel_zscore_scaled_sum_zero`, `gdpo_standardized_multi_channel_sum_zero` |
| **Reverse-KL SFT-Anchor Bound** | `KLAnchorSpec`, `checkKLAnchor` | `checkKLAnchor_ok_iff`, `kl_anchor_drift_bound`, `kl_anchor_permille_le_div` |

## Build & Verify

```bash
cd lean
lake build
python ../scripts/verify_lean_proofs_and_parity.py --cases 300 --self-test-sorry
```

Zero external dependencies, `0 sorry`, no custom axioms (uses only Lean's standard kernel axioms `propext`, `Classical.choice`, and `Quot.sound`), 20 compile-time `#guard` checks, and a 300-case random `#eval` differential parity check against the Python mirrors in `autolab_wire.py`.
