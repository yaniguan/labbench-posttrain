/-!
# LabPostTrainLean: Leak-Free Splits, Anti-Hack RLVR Rewards & Twin-Delta Process Credit
(`labbench-posttrain` · Layer L3)

Self-contained Lean 4.36.0 formalization extending the base L3 specification in the central
Auto-Lab Trust Regime (`yaniguan/auto-lab/tree/main/lean`, `AutoLabLean/L3_PostTrain.lean`
for Sections 1-3) with local `labbench-posttrain` multi-objective GDPO (both raw-channel
simplex aggregation and within-group standardized GDPO conservation) and Reverse-KL
SFT-anchor trust-region theorems (Sections 4-5):

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
4. **Multi-Objective GDPO Raw-Channel Simplex & Within-Group Standardized Advantage (`#47`, `#55`)**:
   Enforces `≥ 2` channels, matching weight/channel arity, and strictly positive weight sum;
   distinguishes raw-channel simplex convex-combination upper bounds (`raw_channel_pareto_normalized_le_max` /
   `gdpo_pareto_normalized_le_max`) from per-channel dispersion-scaled within-group standardized
   GDPO advantages (`gdpoChannelZScoreScaled`, `gdpo_zero_spread_channel_zero`,
   `gdpo_standardized_multi_channel_sum_zero`).
5. **Reverse-KL SFT-Anchor Trust-Region Bound (`KLAnchorSpec`, `checkKLAnchor`)**:
   Enforces frozen SFT reference weights, strictly positive `klCoefPermille`, and bounded
   `klToSftPermille ≤ maxKlDriftPermille`, proving deterministic policy-drift bounds relative
   to the frozen `π_SFT` anchor.
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

/-! ## 4. Multi-Objective GDPO Simplex Pareto Normalization & Group-Centering -/

/-- Multi-objective GDPO rollout specification pairing `K ≥ 2` reward channels (in per-mille)
    with non-negative integer Pareto weights (`#47`). -/
structure GDPOMultiRewardSpec where
  coreScientificVerified : Bool
  oracleFieldStripped    : Bool
  thinkTokens            : Nat
  maxThinkTokens         : Nat
  channels               : List Nat
  weights                : List Nat
  deriving DecidableEq, Repr

/-- Validate multi-objective GDPO Pareto weights: requires at least 2 channels, matching
    weight/channel lengths, and strictly positive simplex denominator `0 < weights.sum`. -/
def checkGDPOParetoWeights (s : GDPOMultiRewardSpec) : Option RegimeViolation :=
  firstViolation
    [ (decide (2 ≤ s.channels.length),
        { repo := "labbench-posttrain", rule := "gdpo_min_two_channels",
          detail := "GDPO multi-reward decomposition requires at least 2 reward channels" }),
      (decide (s.weights.length = s.channels.length),
        { repo := "labbench-posttrain", rule := "gdpo_weight_channel_arity",
          detail := "GDPO Pareto weight vector length does not match reward channel count" }),
      (decide (0 < s.weights.sum),
        { repo := "labbench-posttrain", rule := "gdpo_positive_weight_sum",
          detail := "GDPO Pareto weight vector sums to zero (degenerate simplex)" }) ]

theorem checkGDPOParetoWeights_ok_iff (s : GDPOMultiRewardSpec) :
    checkGDPOParetoWeights s = none ↔
      2 ≤ s.channels.length ∧
      s.weights.length = s.channels.length ∧
      0 < s.weights.sum := by
  unfold checkGDPOParetoWeights
  simp [firstViolation_cons, firstViolation_nil, decide_eq_true_eq]

/-- Per-channel multiplicative anti-hack gate: zeroes every GDPO channel simultaneously
    whenever `coreScientificVerified`, `oracleFieldStripped`, or `thinkTokens ≤ maxThinkTokens` fails. -/
def gatedGDPOChannelsPermille (s : GDPOMultiRewardSpec) : List Nat :=
  if s.coreScientificVerified && s.oracleFieldStripped && decide (s.thinkTokens ≤ s.maxThinkTokens) then
    s.channels
  else
    s.channels.map (fun _ => 0)

/-- Multi-Channel Gate Collapse Theorem: failing the anti-hack rollout gate zeroes every
    individual GDPO channel simultaneously, preventing auxiliary channel variance leakage. -/
theorem gdpo_unverified_all_channels_zero (s : GDPOMultiRewardSpec)
    (hfail : s.coreScientificVerified = false ∨
             s.oracleFieldStripped = false ∨
             s.maxThinkTokens < s.thinkTokens) :
    ∀ g ∈ gatedGDPOChannelsPermille s, g = 0 := by
  unfold gatedGDPOChannelsPermille
  rcases hfail with hc | ho | ht
  · simp [hc]
  · simp [ho]
  · have hnd : decide (s.thinkTokens ≤ s.maxThinkTokens) = false :=
      decide_eq_false (by omega)
    simp [hnd]

/-- Non-negative integer inner product `∑_i w_i * c_i` for Pareto aggregation. -/
def weightedDotNat : List Nat → List Nat → Nat
  | [], _ => 0
  | _, [] => 0
  | w :: ws, c :: cs => w * c + weightedDotNat ws cs

/-- Simplex-normalized Pareto scalar `(∑_i w_i * c_i) / (∑_i w_i)` in per-mille (`0` if `w.sum = 0`). -/
def paretoNormalizedPermille (weights channels : List Nat) : Nat :=
  let wSum := weights.sum
  if wSum == 0 then 0 else weightedDotNat weights channels / wSum

theorem weightedDotNat_le_sum_mul_max (weights channels : List Nat) (maxChannelPermille : Nat)
    (hbound : ∀ c ∈ channels, c ≤ maxChannelPermille) :
    weightedDotNat weights channels ≤ weights.sum * maxChannelPermille := by
  induction weights generalizing channels with
  | nil =>
    simp [weightedDotNat]
  | cons w ws ih =>
    cases channels with
    | nil =>
      simp [weightedDotNat]
    | cons c cs =>
      simp only [weightedDotNat, List.sum_cons]
      have hc : c ≤ maxChannelPermille := hbound c (by simp)
      have hwc : w * c ≤ w * maxChannelPermille := Nat.mul_le_mul_left w hc
      have htail : weightedDotNat ws cs ≤ ws.sum * maxChannelPermille :=
        ih cs (fun x hx => hbound x (by simp [hx]))
      have hdistrib : (w + ws.sum) * maxChannelPermille =
          w * maxChannelPermille + ws.sum * maxChannelPermille := Nat.add_mul w ws.sum maxChannelPermille
      omega

/-- Raw-Channel Simplex Convex-Combination Envelope Theorem (`#55`):
    whenever every raw channel satisfies `c ≤ maxChannelPermille`, simplex-normalized raw-channel
    Pareto aggregation `paretoNormalizedPermille weights channels` never exceeds `maxChannelPermille`.
    (Models weighted raw-channel aggregation; per-channel standardized GDPO advantages are
    modeled below via `gdpoChannelZScoreScaled` and `gdpoStandardizedGroupSum`.) -/
theorem raw_channel_pareto_normalized_le_max (weights channels : List Nat) (maxChannelPermille : Nat)
    (hbound : ∀ c ∈ channels, c ≤ maxChannelPermille) :
    paretoNormalizedPermille weights channels ≤ maxChannelPermille := by
  unfold paretoNormalizedPermille
  by_cases hz : weights.sum = 0
  · simp [hz]
  · have hpos : 0 < weights.sum := Nat.pos_of_ne_zero hz
    have hbeq : (weights.sum == 0) = false := beq_false_of_ne hz
    simp only [hbeq]
    have hdot := weightedDotNat_le_sum_mul_max weights channels maxChannelPermille hbound
    have hdiv := Nat.div_le_div_right (c := weights.sum) hdot
    rw [Nat.mul_div_cancel_left maxChannelPermille hpos] at hdiv
    exact hdiv

/-- Backward-compatible alias for `raw_channel_pareto_normalized_le_max`. -/
theorem gdpo_pareto_normalized_le_max (weights channels : List Nat) (maxChannelPermille : Nat)
    (hbound : ∀ c ∈ channels, c ≤ maxChannelPermille) :
    paretoNormalizedPermille weights channels ≤ maxChannelPermille :=
  raw_channel_pareto_normalized_le_max weights channels maxChannelPermille hbound

/-- Explicit integer list sum for algebraic group-centering conservation proofs. -/
def intListSum : List Int → Int
  | [] => 0
  | x :: xs => x + intListSum xs

/-- Integer-scaled within-group centered advantages `G * r_i - ∑_j r_j` for a group of size `G`. -/
def groupCenteredScaled (rewards : List Int) : List Int :=
  let g : Int := Int.ofNat rewards.length
  let total : Int := intListSum rewards
  rewards.map (fun r => g * r - total)

theorem intListSum_affine_map (xs : List Int) (scale shift : Int) :
    intListSum (xs.map (fun r => scale * r - shift)) =
      scale * intListSum xs - (Int.ofNat xs.length) * shift := by
  induction xs with
  | nil =>
    simp [intListSum]
  | cons x rest ih =>
    simp only [List.map_cons, intListSum, List.length_cons, ih]
    have hmul : scale * (x + intListSum rest) = scale * x + scale * intListSum rest :=
      Int.mul_add scale x (intListSum rest)
    have hlen : Int.ofNat (rest.length + 1) * shift = Int.ofNat rest.length * shift + shift := by
      have hcast : Int.ofNat (rest.length + 1) = Int.ofNat rest.length + 1 := rfl
      rw [hcast, Int.add_mul, Int.one_mul]
    omega

/-- Within-Group Zero-Sum Conservation Theorem:
    integer-scaled centered advantages `G * r_i - ∑_j r_j` sum identically to `0 : Int`
    for every rollout group `rewards`. -/
theorem group_centered_scaled_sum_zero (rewards : List Int) :
    intListSum (groupCenteredScaled rewards) = 0 := by
  unfold groupCenteredScaled
  rw [intListSum_affine_map]
  omega

/-- Per-channel standardized GDPO group component (`#55`, mirroring `posttrain/advantages.py::gdpo_advantages`):
    when `spreadScale = 0` (zero within-group spread), suppresses the channel to all-zeros so `0/0`
    rounding noise cannot be amplified; when `0 < spreadScale` (representing `w_k * ∏_{m ≠ k} std_m`),
    scales the centered group deviations `G * r_i - ∑_j r_j` by `spreadScale`. -/
def gdpoChannelZScoreScaled (rewards : List Int) (spreadScale : Nat) : List Int :=
  if spreadScale == 0 then
    rewards.map (fun _ => 0)
  else
    (groupCenteredScaled rewards).map (fun c => Int.ofNat spreadScale * c)

theorem intListSum_zeros (xs : List α) :
    intListSum (xs.map (fun _ => (0 : Int))) = 0 := by
  induction xs with
  | nil => simp [intListSum]
  | cons _ rest ih => simp [intListSum, ih]

theorem intListSum_scale_map (xs : List Int) (scale : Int) :
    intListSum (xs.map (fun c => scale * c)) = scale * intListSum xs := by
  induction xs with
  | nil => simp [intListSum]
  | cons x rest ih =>
    simp only [List.map_cons, intListSum, ih]
    have hmul : scale * (x + intListSum rest) = scale * x + scale * intListSum rest :=
      Int.mul_add scale x (intListSum rest)
    omega

/-- Zero-Spread Channel Suppression Theorem (`#55`):
    a zero-variance channel (`spreadScale = 0`) contributes identically `0` to every rollout in the group. -/
theorem gdpo_zero_spread_channel_zero (rewards : List Int) :
    ∀ a ∈ gdpoChannelZScoreScaled rewards 0, a = 0 := by
  unfold gdpoChannelZScoreScaled
  simp

/-- Per-Channel Standardized GDPO Zero-Sum Theorem (`#55`):
    for any channel rewards and any dispersion-scaled weight `spreadScale`,
    the standardized group component sums identically to `0 : Int`. -/
theorem gdpo_channel_zscore_scaled_sum_zero (rewards : List Int) (spreadScale : Nat) :
    intListSum (gdpoChannelZScoreScaled rewards spreadScale) = 0 := by
  unfold gdpoChannelZScoreScaled
  by_cases hz : spreadScale = 0
  · simp [hz, intListSum_zeros]
  · have hbeq : (spreadScale == 0) = false := beq_false_of_ne hz
    simp [hbeq, intListSum_scale_map, group_centered_scaled_sum_zero]

/-- Elementwise integer vector addition for multi-channel GDPO aggregation. -/
def zipAddInt : List Int → List Int → List Int
  | [], ys => ys
  | xs, [] => xs
  | x :: xs, y :: ys => (x + y) :: zipAddInt xs ys

theorem intListSum_zipAddInt (xs ys : List Int) :
    intListSum (zipAddInt xs ys) = intListSum xs + intListSum ys := by
  induction xs generalizing ys with
  | nil => simp [zipAddInt, intListSum]
  | cons x xrest ih =>
    cases ys with
    | nil => simp [zipAddInt, intListSum]
    | cons y yrest =>
      simp only [zipAddInt, intListSum, ih yrest]
      omega

/-- Multi-channel standardized GDPO group aggregation across `K` `(rewards_k, spreadScale_k)` channels. -/
def gdpoStandardizedGroupSum : List (List Int × Nat) → List Int
  | [] => []
  | (r, s) :: rest => zipAddInt (gdpoChannelZScoreScaled r s) (gdpoStandardizedGroupSum rest)

/-- Multi-Channel Standardized GDPO Zero-Sum Conservation Theorem (`#55`):
    aggregating arbitrary dispersion-standardized weighted channels preserves exact within-group
    zero-sum conservation `∑_i A_i = 0` across every prompt group. -/
theorem gdpo_standardized_multi_channel_sum_zero (channels : List (List Int × Nat)) :
    intListSum (gdpoStandardizedGroupSum channels) = 0 := by
  induction channels with
  | nil => simp [gdpoStandardizedGroupSum, intListSum]
  | cons hd tl ih =>
    rcases hd with ⟨r, s⟩
    simp only [gdpoStandardizedGroupSum, intListSum_zipAddInt,
               gdpo_channel_zscore_scaled_sum_zero, ih]
    omega

/-! ## 5. Reverse-KL SFT-Anchor Guardrail & Policy-Drift Bound -/

/-- Specification for frozen-SFT Reverse-KL trust-region enforcement (`#47`). -/
structure KLAnchorSpec where
  klCoefPermille     : Nat
  klToSftPermille    : Nat
  maxKlDriftPermille : Nat
  sftAnchorFrozen    : Bool
  deriving DecidableEq, Repr

/-- Verify that the SFT anchor remains frozen, `klCoefPermille > 0`, and Reverse-KL drift
    stays within `maxKlDriftPermille`. -/
def checkKLAnchor (k : KLAnchorSpec) : Option RegimeViolation :=
  firstViolation
    [ (k.sftAnchorFrozen,
        { repo := "labbench-posttrain", rule := "sft_anchor_frozen",
          detail := "SFT reference policy weights mutated during RLVR optimization" }),
      (decide (0 < k.klCoefPermille),
        { repo := "labbench-posttrain", rule := "positive_kl_coef",
          detail := "Reverse-KL SFT anchor coefficient must be strictly positive" }),
      (decide (k.klToSftPermille ≤ k.maxKlDriftPermille),
        { repo := "labbench-posttrain", rule := "kl_drift_envelope",
          detail := "Reverse-KL divergence to frozen SFT anchor exceeded trust-region cap" }) ]

theorem checkKLAnchor_ok_iff (k : KLAnchorSpec) :
    checkKLAnchor k = none ↔
      k.sftAnchorFrozen = true ∧
      0 < k.klCoefPermille ∧
      k.klToSftPermille ≤ k.maxKlDriftPermille := by
  unfold checkKLAnchor
  simp [firstViolation_cons, firstViolation_nil, decide_eq_true_eq]

/-- Non-Negative Reverse-KL SFT-Anchor Drift Bound Theorem:
    For any policy `π` whose KL-penalized objective improves over the frozen SFT anchor `π_SFT`
    (`KL(π_SFT || π_SFT) = 0`, so `(sftRewardPermille : Int) ≤ rewardPermille - klCoefPermille * klPermille`)
    under bounded task reward `rewardPermille ≤ rMaxPermille`, the non-negative KL penalty
    `klCoefPermille * klPermille` is deterministically bounded by `rMaxPermille - sftRewardPermille`. -/
theorem kl_anchor_drift_bound
    (rewardPermille sftRewardPermille rMaxPermille klCoefPermille klPermille : Nat)
    (hrmax : rewardPermille ≤ rMaxPermille)
    (himprove : (sftRewardPermille : Int) ≤
      (rewardPermille : Int) - (klCoefPermille : Int) * (klPermille : Int)) :
    0 ≤ klCoefPermille * klPermille ∧
    klCoefPermille * klPermille ≤ rMaxPermille - sftRewardPermille := by
  have hprod_nonneg : (0 : Int) ≤ (klCoefPermille : Int) * (klPermille : Int) :=
    Int.mul_nonneg (Int.natCast_nonneg klCoefPermille) (Int.natCast_nonneg klPermille)
  have hprod_cast : ((klCoefPermille * klPermille : Nat) : Int) =
      (klCoefPermille : Int) * (klPermille : Int) := Int.natCast_mul klCoefPermille klPermille
  omega

/-- Explicit KL-divergence ceiling corollary when `0 < klCoefPermille`. -/
theorem kl_anchor_permille_le_div
    (rewardPermille sftRewardPermille rMaxPermille klCoefPermille klPermille : Nat)
    (hcoef : 0 < klCoefPermille)
    (hrmax : rewardPermille ≤ rMaxPermille)
    (himprove : (sftRewardPermille : Int) ≤
      (rewardPermille : Int) - (klCoefPermille : Int) * (klPermille : Int)) :
    klPermille ≤ (rMaxPermille - sftRewardPermille) / klCoefPermille := by
  have hbound := (kl_anchor_drift_bound
    rewardPermille sftRewardPermille rMaxPermille klCoefPermille klPermille hrmax himprove).2
  have hdiv := Nat.div_le_div_right (c := klCoefPermille) hbound
  rw [Nat.mul_div_cancel_left klPermille hcoef] at hdiv
  exact hdiv

/-! ## 6. Compile-Time `#guard` Witnesses -/

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

#guard checkGDPOParetoWeights
  { coreScientificVerified := true, oracleFieldStripped := true,
    thinkTokens := 150, maxThinkTokens := 512,
    channels := [900, 800, 1000, 700], weights := [12, 8, 5, 5] } == none

#guard (checkGDPOParetoWeights
  { coreScientificVerified := true, oracleFieldStripped := true,
    thinkTokens := 150, maxThinkTokens := 512,
    channels := [900, 800], weights := [0, 0] }).isSome

#guard (checkGDPOParetoWeights
  { coreScientificVerified := true, oracleFieldStripped := true,
    thinkTokens := 150, maxThinkTokens := 512,
    channels := [900], weights := [10] }).isSome

#guard gatedGDPOChannelsPermille
  { coreScientificVerified := false, oracleFieldStripped := true,
    thinkTokens := 150, maxThinkTokens := 512,
    channels := [900, 800, 1000], weights := [12, 8, 5] } == [0, 0, 0]

#guard paretoNormalizedPermille [12, 8, 5, 5] [1000, 750, 800, 600] == 833

#guard intListSum (groupCenteredScaled [1000, 250, 0, 750]) == 0

#guard checkKLAnchor
  { klCoefPermille := 10, klToSftPermille := 45, maxKlDriftPermille := 200, sftAnchorFrozen := true }
  == none

#guard (checkKLAnchor
  { klCoefPermille := 0, klToSftPermille := 45, maxKlDriftPermille := 200, sftAnchorFrozen := true }).isSome

#guard (checkKLAnchor
  { klCoefPermille := 10, klToSftPermille := 45, maxKlDriftPermille := 200, sftAnchorFrozen := false }).isSome

#guard (checkKLAnchor
  { klCoefPermille := 10, klToSftPermille := 250, maxKlDriftPermille := 200, sftAnchorFrozen := true }).isSome

end LabPostTrainLean
