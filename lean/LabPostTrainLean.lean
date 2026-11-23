/-!
# LabPostTrainLean: Leak-Free Splits, Anti-Hack RLVR Rewards & Twin-Delta Process Credit
(`labbench-posttrain` · Layer L3)

Self-contained Lean 4.36.0 formalization synced with the central Auto-Lab Trust Regime
(`yaniguan/auto-lab/tree/main/lean`, `AutoLabLean/L3_PostTrain.lean`), covering:

1. **SplitGuard (`splitGuardDisjoint`, `checkSplitGuard`)**:
   Pairwise disjoint group/scaffold/DOI IDs across `train`, `val`, `test`, plus strict
   separation from `canaryGroups` (`LAB-Bench` evaluation-only holdout).
2. **Multiplicative Anti-Hack Reward (`RolloutRewardSpec`, `gatedRewardPermille`, `checkRolloutGate`)**:
   Format and auxiliary bonuses are gated to strictly `0` unless `coreScientificVerified = true`,
   `oracleFieldStripped = true`, and `thinkTokens ≤ maxThinkTokens`.
3. **Twin-Delta Step-Level Process Credit (`ProcessStepSpec`, `stepProcessBonusPermille`)**:
   Intermediate `<think>` drafts receive `0` process bonus whenever they fail to strictly
   reduce `violationCount` (`prevViolations ≤ currViolations`), exceed `maxThinkTokens`,
   leak oracle fields, or fail final `coreScientificVerified = true`, with anti-oscillation
   frontier telescoping bounded by initial violation count `v0`.
-/
namespace LabPostTrainLean

/-- Structured boundary violation reporting rule identifier and diagnostic detail. -/
structure RegimeViolation where
  repo   : String
  rule   : String
  detail : String
  deriving DecidableEq, Repr

/-- Return the first failing check in a list of `(holds, violation)` pairs. -/
def firstViolation : List (Bool × RegimeViolation) → Option RegimeViolation
  | [] => none
  | (true, _) :: rest => firstViolation rest
  | (false, v) :: _ => some v

@[simp] theorem firstViolation_nil : firstViolation [] = none := rfl

@[simp] theorem firstViolation_cons (b : Bool) (v : RegimeViolation)
    (rest : List (Bool × RegimeViolation)) :
    firstViolation ((b, v) :: rest) = none ↔ b = true ∧ firstViolation rest = none := by
  cases b <;> simp [firstViolation]

theorem firstViolation_none_iff (checks : List (Bool × RegimeViolation)) :
    firstViolation checks = none ↔ ∀ p ∈ checks, p.1 = true := by
  induction checks with
  | nil => simp [firstViolation]
  | cons hd tl ih =>
    rcases hd with ⟨b, v⟩
    cases b with
    | false => simp [firstViolation]
    | true => simp [firstViolation, ih]

/-! ## 1. SplitGuard: Group, Scaffold, DOI & Canary Disjointness -/

/-- Decidable disjointness predicate for split group/scaffold/DOI ID lists. -/
def splitGuardDisjoint (xs ys : List Nat) : Bool :=
  xs.all (fun x => !ys.contains x)

structure SplitManifest where
  trainGroups  : List Nat
  valGroups    : List Nat
  testGroups   : List Nat
  canaryGroups : List Nat
  deriving DecidableEq, Repr

def checkSplitGuard (m : SplitManifest) : Option RegimeViolation :=
  firstViolation
    [ (splitGuardDisjoint m.trainGroups m.valGroups &&
       splitGuardDisjoint m.trainGroups m.testGroups &&
       splitGuardDisjoint m.valGroups m.testGroups,
        { repo := "labbench-posttrain", rule := "splitguard_zero_group_leak",
          detail := "shared molecule scaffold, DOI, or protocol group leaked across train/val/test" }),
      (splitGuardDisjoint m.trainGroups m.canaryGroups &&
       splitGuardDisjoint m.valGroups m.canaryGroups,
        { repo := "labbench-posttrain", rule := "canary_eval_holdout",
          detail := "LAB-Bench canary item detected inside training or validation split" }) ]

/-- SplitGuard Zero-Leakage Theorem: if `checkSplitGuard m = none`, no group ID can appear
    in both `trainGroups` and `testGroups` (nor in `trainGroups` and `canaryGroups`). -/
theorem splitGuard_no_train_test_overlap (m : SplitManifest) (g : Nat)
    (hok : checkSplitGuard m = none)
    (htrain : g ∈ m.trainGroups) :
    g ∉ m.testGroups ∧ g ∉ m.canaryGroups := by
  unfold checkSplitGuard at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true, Bool.and_eq_true,
             splitGuardDisjoint, List.all_eq_true] at hok
  rcases hok with ⟨⟨⟨_, hnotTestB⟩, _⟩, ⟨hnotCanB, _⟩⟩
  have hnotTest := hnotTestB g htrain
  have hnotCan  := hnotCanB g htrain
  simp only [Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not] at hnotTest hnotCan
  exact ⟨hnotTest, hnotCan⟩

/-- Validation-split isolation: if `checkSplitGuard m = none`, no group ID in `valGroups`
    appears in `testGroups` or `canaryGroups`. -/
theorem splitGuard_no_val_test_or_canary_overlap (m : SplitManifest) (g : Nat)
    (hok : checkSplitGuard m = none)
    (hval : g ∈ m.valGroups) :
    g ∉ m.testGroups ∧ g ∉ m.canaryGroups := by
  unfold checkSplitGuard at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true, Bool.and_eq_true,
             splitGuardDisjoint, List.all_eq_true] at hok
  rcases hok with ⟨⟨_, hvalTestB⟩, ⟨_, hvalCanB⟩⟩
  have hnotTest := hvalTestB g hval
  have hnotCan  := hvalCanB g hval
  simp only [Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not] at hnotTest hnotCan
  exact ⟨hnotTest, hnotCan⟩

/-! ## 2. Multiplicative Anti-Hack Rollout Reward Gate -/

structure RolloutRewardSpec where
  coreScientificVerified : Bool
  oracleFieldStripped    : Bool
  thinkTokens            : Nat
  maxThinkTokens         : Nat
  taskRewardPermille     : Nat
  formatBonusPermille    : Nat
  deriving DecidableEq, Repr

/-- Multiplicative anti-hack reward composer (in per-mille `10^-3`):
    unverified rollouts, oracle leaks, or `<think>` padding receive strictly `0`. -/
def gatedRewardPermille (r : RolloutRewardSpec) : Nat :=
  if r.coreScientificVerified && r.oracleFieldStripped && decide (r.thinkTokens ≤ r.maxThinkTokens) then
    r.taskRewardPermille + r.formatBonusPermille
  else
    0

def checkRolloutGate (r : RolloutRewardSpec) : Option RegimeViolation :=
  firstViolation
    [ (r.oracleFieldStripped,
        { repo := "labbench-posttrain", rule := "no_oracle_prompt_leak",
          detail := "task record still exposes oracle_formulation or gold answer to policy" }),
      (decide (r.thinkTokens ≤ r.maxThinkTokens),
        { repo := "labbench-posttrain", rule := "think_length_cap",
          detail := "scratchpad <think> length exceeds anti-verbosity cap" }),
      (r.coreScientificVerified,
        { repo := "labbench-posttrain", rule := "core_verifier_passed",
          detail := "candidate rollout failed domain code/physics checker; bonus gated to 0" }) ]

/-- Anti-Reward-Hacking Theorem: Any rollout that fails the core scientific checker,
    exceeds `<think>` budget, or leaks the oracle field earns strictly `0` total reward
    regardless of how large `formatBonusPermille` is. -/
theorem unverified_rollout_gets_zero_reward (r : RolloutRewardSpec)
    (hfail : r.coreScientificVerified = false ∨
             r.oracleFieldStripped = false ∨
             r.maxThinkTokens < r.thinkTokens) :
    gatedRewardPermille r = 0 := by
  unfold gatedRewardPermille
  rcases hfail with hc | ho | ht
  · simp [hc]
  · simp [ho]
  · have hnd : decide (r.thinkTokens ≤ r.maxThinkTokens) = false :=
      decide_eq_false (by omega)
    simp [hnd]

theorem checkRolloutGate_ok_iff (r : RolloutRewardSpec) :
    checkRolloutGate r = none ↔
      r.oracleFieldStripped = true ∧
      r.thinkTokens ≤ r.maxThinkTokens ∧
      r.coreScientificVerified = true := by
  unfold checkRolloutGate
  simp [firstViolation_cons, firstViolation_nil, decide_eq_true_eq]

/-! ## 3. Twin-Delta Step-Level Process Credit & Anti-`<think>`-Bloat Shaping -/

/-- Specification for a single intermediate `<think>` reasoning draft step transitioning
    digital-twin violation count from `violationCountPrev` to `violationCountCurr`. -/
structure ProcessStepSpec where
  violationCountPrev     : Nat
  violationCountCurr     : Nat
  unitBonusPermille      : Nat
  coreScientificVerified : Bool
  oracleFieldStripped    : Bool
  thinkTokens            : Nat
  maxThinkTokens         : Nat
  deriving DecidableEq, Repr

/-- Twin-delta step-level process bonus (in per-mille):
    awards `(violationCountPrev - violationCountCurr) * unitBonusPermille` ONLY when
    the step strictly decreases `violationCount` (`violationCountCurr < violationCountPrev`),
    final verification passes (`coreScientificVerified = true`), the oracle field is stripped,
    and total `<think>` tokens satisfy `thinkTokens ≤ maxThinkTokens`. -/
def stepProcessBonusPermille (s : ProcessStepSpec) : Nat :=
  if s.coreScientificVerified &&
     s.oracleFieldStripped &&
     decide (s.thinkTokens ≤ s.maxThinkTokens) &&
     decide (s.violationCountCurr < s.violationCountPrev) then
    (s.violationCountPrev - s.violationCountCurr) * s.unitBonusPermille
  else
    0

/-- Process-Credit Anti-Bloat Theorem:
    Any intermediate `<think>` step that fails to strictly decrease `violationCount`
    (`violationCountPrev ≤ violationCountCurr`), lacks final `coreScientificVerified = true`,
    leaks the oracle field, or exceeds `maxThinkTokens` receives strictly `0` process bonus. -/
theorem vacuous_step_gets_zero_process_bonus (s : ProcessStepSpec)
    (hvac : s.violationCountPrev ≤ s.violationCountCurr ∨
            s.coreScientificVerified = false ∨
            s.oracleFieldStripped = false ∨
            s.maxThinkTokens < s.thinkTokens) :
    stepProcessBonusPermille s = 0 := by
  unfold stepProcessBonusPermille
  rcases hvac with hv | hc | ho | ht
  · have hnd : decide (s.violationCountCurr < s.violationCountPrev) = false :=
      decide_eq_false (by omega)
    simp [hnd]
  · simp [hc]
  · simp [ho]
  · have hnd : decide (s.thinkTokens ≤ s.maxThinkTokens) = false :=
      decide_eq_false (by omega)
    simp [hnd]

/-- Running-minimum frontier violation reduction across a multi-step `<think>` trace:
    tracks `bestMin` so temporarily injecting violations mid-trace cannot farm extra credit. -/
def frontierTraceDeltaSum (bestMin : Nat) : List Nat → Nat
  | [] => 0
  | v :: vs =>
    if v < bestMin then
      (bestMin - v) + frontierTraceDeltaSum v vs
    else
      frontierTraceDeltaSum bestMin vs

/-- Anti-Oscillation Telescoping Bound:
    No matter how many intermediate `<think>` steps or temporary regressions an agent emits,
    total frontier violation reduction starting from `bestMin` never exceeds `bestMin`. -/
theorem frontierTraceDeltaSum_le_initial (bestMin : Nat) (vs : List Nat) :
    frontierTraceDeltaSum bestMin vs ≤ bestMin := by
  induction vs generalizing bestMin with
  | nil => simp [frontierTraceDeltaSum]
  | cons v rest ih =>
    unfold frontierTraceDeltaSum
    split
    · next hlt =>
      have hsub := ih v
      omega
    · next hge =>
      exact ih bestMin

/-- Total gated process bonus over a full violation trace `v0 :: vs`. -/
def traceProcessBonusPermille (v0 : Nat) (vs : List Nat) (unitBonusPermille : Nat)
    (coreScientificVerified oracleFieldStripped : Bool) (thinkTokens maxThinkTokens : Nat) : Nat :=
  if coreScientificVerified && oracleFieldStripped && decide (thinkTokens ≤ maxThinkTokens) then
    frontierTraceDeltaSum v0 vs * unitBonusPermille
  else
    0

theorem unverified_trace_gets_zero_process_bonus (v0 : Nat) (vs : List Nat)
    (unitBonusPermille : Nat) (coreScientificVerified oracleFieldStripped : Bool)
    (thinkTokens maxThinkTokens : Nat)
    (hfail : coreScientificVerified = false ∨
             oracleFieldStripped = false ∨
             maxThinkTokens < thinkTokens) :
    traceProcessBonusPermille v0 vs unitBonusPermille
      coreScientificVerified oracleFieldStripped thinkTokens maxThinkTokens = 0 := by
  unfold traceProcessBonusPermille
  rcases hfail with hc | ho | ht
  · simp [hc]
  · simp [ho]
  · have hnd : decide (thinkTokens ≤ maxThinkTokens) = false :=
      decide_eq_false (by omega)
    simp [hnd]

/-! ## 4. Compile-Time `#guard` Witnesses -/

#guard checkSplitGuard
  { trainGroups := [101, 102, 103], valGroups := [201], testGroups := [301, 302], canaryGroups := [999] }
  == none

#guard (checkSplitGuard
  { trainGroups := [101, 302], valGroups := [201], testGroups := [301, 302], canaryGroups := [999] }).isSome

#guard (checkSplitGuard
  { trainGroups := [101, 999], valGroups := [201], testGroups := [301], canaryGroups := [999] }).isSome

#guard gatedRewardPermille
  { coreScientificVerified := true, oracleFieldStripped := true,
    thinkTokens := 180, maxThinkTokens := 256,
    taskRewardPermille := 1000, formatBonusPermille := 100 } == 1100

#guard gatedRewardPermille
  { coreScientificVerified := false, oracleFieldStripped := true,
    thinkTokens := 120, maxThinkTokens := 256,
    taskRewardPermille := 1000, formatBonusPermille := 500 } == 0

#guard gatedRewardPermille
  { coreScientificVerified := true, oracleFieldStripped := true,
    thinkTokens := 400, maxThinkTokens := 256,
    taskRewardPermille := 1000, formatBonusPermille := 100 } == 0

#guard stepProcessBonusPermille
  { violationCountPrev := 4, violationCountCurr := 2, unitBonusPermille := 150,
    coreScientificVerified := true, oracleFieldStripped := true,
    thinkTokens := 140, maxThinkTokens := 256 } == 300

#guard stepProcessBonusPermille
  { violationCountPrev := 2, violationCountCurr := 2, unitBonusPermille := 150,
    coreScientificVerified := true, oracleFieldStripped := true,
    thinkTokens := 140, maxThinkTokens := 256 } == 0

#guard stepProcessBonusPermille
  { violationCountPrev := 3, violationCountCurr := 1, unitBonusPermille := 150,
    coreScientificVerified := false, oracleFieldStripped := true,
    thinkTokens := 140, maxThinkTokens := 256 } == 0

#guard traceProcessBonusPermille 3 [7, 2, 2, 0] 100 true true 160 256 == 300

end LabPostTrainLean
