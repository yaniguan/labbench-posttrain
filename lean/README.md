# `LabPostTrainLean` — Formal Anti-Hack RLVR, SplitGuard & Process Credit (`L3`)

Self-contained **Lean 4** (`leanprover/lean4:v4.36.0`) verification kernel for `labbench-posttrain`, synchronized with the central Auto-Lab Trust Regime at [`yaniguan/auto-lab/tree/main/lean`](https://github.com/yaniguan/auto-lab/tree/main/lean) (`AutoLabLean/L3_PostTrain.lean`).

## Formalized Guardrails & Proved Theorems

| Component | Definitions | Machine-Checked Theorems |
| :--- | :--- | :--- |
| **Shared Diagnostics** | `RegimeViolation`, `firstViolation` | `firstViolation_nil`, `firstViolation_cons`, `firstViolation_none_iff` |
| **SplitGuard Holdout** | `splitGuardDisjoint`, `SplitManifest`, `checkSplitGuard` | `splitGuard_no_train_test_overlap`, `splitGuard_no_val_test_or_canary_overlap` |
| **Anti-Hack Rollout Gate** | `RolloutRewardSpec`, `gatedRewardPermille`, `checkRolloutGate` | `unverified_rollout_gets_zero_reward`, `checkRolloutGate_ok_iff` |
| **Twin-Delta Process Credit** | `ProcessStepSpec`, `stepProcessBonusPermille`, `frontierTraceDeltaSum`, `traceProcessBonusPermille` | `vacuous_step_gets_zero_process_bonus`, `frontierTraceDeltaSum_le_initial`, `unverified_trace_gets_zero_process_bonus` |

## Build & Verify

```bash
cd lean
lake build
```

Zero external dependencies, `0 sorry`, `0 axiom`, and 10 compile-time `#guard` checks.
