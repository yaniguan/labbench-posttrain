"""Idea #14: Twin-Delta Step-Level Process Credit & Anti-<think>-Bloat Shaping.

Inspired by DeepSeek-R1, DAPO, and GSPO analyses of scratchpad (<think>) length
hacking and sparse-reward credit assignment in multi-step scientific reasoning.

Each intermediate protocol draft inside a rollout is checked against a deterministic
wet-lab digital twin to obtain exact violation counts v_0, v_1, ..., v_K:
  - If final verification fails (v_K != 0), oracle fields are leaked, or total
    <think> length exceeds `max_think_tokens`, EVERY step bonus and terminal reward
    is strictly 0 (matching `LabPostTrainLean.unverified_rollout_gets_zero_reward`
    and `LabPostTrainLean.vacuous_step_gets_zero_process_bonus`).
  - When v_K == 0 and `think_tokens <= max_think_tokens`, step k earns positive
    process credit proportional to frontier violation reduction
    max(0, min_{0<=j<k} v_j - v_k), so vacuous verbosity steps and oscillation
    detours earn zero process bonus while true repair steps receive direct,
    low-variance policy-gradient credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# 4-bit wet-lab digital-twin violation flags:
VIOL_FROZEN_REAGENT = 1 << 0       # Aspirating from unthawed -20 C stock
VIOL_UNSEALED_HIGH_TEMP = 1 << 1   # >= 80 C denaturation without plate seal
VIOL_DAG_BARRIER_SWAP = 1 << 2     # Read/wash scheduled before upstream mix
VIOL_PIPETTE_RANGE = 1 << 3        # Transfer volume outside mounted pipette range

ACTION_THAW_REAGENT = 0
ACTION_SEAL_HIGH_TEMP = 1
ACTION_FIX_DAG_ORDER = 2
ACTION_CALIBRATE_PIPETTE = 3
ACTION_SPECULATIVE_DETOUR = 4
ACTION_PAD_VERBOSE_THINK = 5
ACTION_FINALIZE_CONCISE = 6
NUM_ACTIONS = 7

ACTION_TOKEN_COSTS = (28, 28, 28, 28, 36, 76, 12)


@dataclass(frozen=True)
class ProtocolDraftState:
    """Intermediate wet-lab protocol draft checked by the L3/L4 digital twin."""

    frozen_reagent: bool = False
    unsealed_high_temp: bool = False
    dag_barrier_swapped: bool = False
    pipette_out_of_range: bool = False

    def to_mask(self) -> int:
        mask = 0
        if self.frozen_reagent:
            mask |= VIOL_FROZEN_REAGENT
        if self.unsealed_high_temp:
            mask |= VIOL_UNSEALED_HIGH_TEMP
        if self.dag_barrier_swapped:
            mask |= VIOL_DAG_BARRIER_SWAP
        if self.pipette_out_of_range:
            mask |= VIOL_PIPETTE_RANGE
        return mask

    @classmethod
    def from_mask(cls, mask: int) -> ProtocolDraftState:
        return cls(
            frozen_reagent=bool(mask & VIOL_FROZEN_REAGENT),
            unsealed_high_temp=bool(mask & VIOL_UNSEALED_HIGH_TEMP),
            dag_barrier_swapped=bool(mask & VIOL_DAG_BARRIER_SWAP),
            pipette_out_of_range=bool(mask & VIOL_PIPETTE_RANGE),
        )


@dataclass(frozen=True)
class ProcessStepCheck:
    """Exact Python mirror of `LabPostTrainLean.ProcessStepSpec`."""

    violation_count_prev: int
    violation_count_curr: int
    unit_bonus_permille: int
    core_scientific_verified: bool
    oracle_field_stripped: bool
    think_tokens: int
    max_think_tokens: int


@dataclass(frozen=True)
class TraceCreditResult:
    """Step-level and rollout-level verifiable reward summary."""

    verified: bool
    gate_passed: bool
    initial_violations: int
    final_violations: int
    think_tokens: int
    step_violation_deltas: tuple[int, ...]
    step_credits: tuple[float, ...]
    terminal_reward: float
    total_credit: float


def count_digital_twin_violations(state: ProtocolDraftState | int) -> int:
    """Count active physical and causal violations in a protocol draft."""
    mask = state if isinstance(state, int) else state.to_mask()
    return int(mask.bit_count())


def score_step_process_bonus_permille(spec: ProcessStepCheck) -> int:
    """Exact mirror of `LabPostTrainLean.stepProcessBonusPermille`."""
    if (
        spec.core_scientific_verified
        and spec.oracle_field_stripped
        and spec.think_tokens <= spec.max_think_tokens
        and spec.violation_count_curr < spec.violation_count_prev
    ):
        return (spec.violation_count_prev - spec.violation_count_curr) * spec.unit_bonus_permille
    return 0


def score_verifiable_trace_credit(
    initial_violations: int,
    step_violations: list[int] | tuple[int, ...],
    think_tokens: int,
    max_think_tokens: int = 240,
    oracle_field_stripped: bool = True,
    process_weight: float = 1.0,
    terminal_weight: float = 0.5,
) -> TraceCreditResult:
    """Compute anti-hack gated twin-delta step credits across v_0, v_1, ..., v_K.

    Uses running-minimum frontier tracking m_k = min(m_{k-1}, v_k) so that:
      1. Monotone repairs satisfy delta_k = max(0, v_{k-1} - v_k).
      2. Vacuous <think> filler steps (v_k >= v_{k-1}) earn strictly 0.0 credit.
      3. Temporary mid-trace regressions cannot farm extra delta credit (> v_0).
      4. Any rollout with v_K != 0, leaked oracle, or think_tokens > max_think_tokens
         receives strictly 0.0 across every step and terminal term.
    """
    if initial_violations < 0:
        raise ValueError("initial_violations must be non-negative")
    final_v = initial_violations if len(step_violations) == 0 else int(step_violations[-1])
    verified = final_v == 0
    gate_passed = bool(
        verified and oracle_field_stripped and (0 <= think_tokens <= max_think_tokens)
    )

    best_min = int(initial_violations)
    deltas: list[int] = []
    for v_raw in step_violations:
        v = int(v_raw)
        if v < 0:
            raise ValueError("step violation count must be non-negative")
        if v < best_min:
            deltas.append(best_min - v)
            best_min = v
        else:
            deltas.append(0)

    if not gate_passed:
        zeros = tuple(0.0 for _ in deltas)
        return TraceCreditResult(
            verified=verified,
            gate_passed=False,
            initial_violations=initial_violations,
            final_violations=final_v,
            think_tokens=think_tokens,
            step_violation_deltas=tuple(deltas),
            step_credits=zeros,
            terminal_reward=0.0,
            total_credit=0.0,
        )

    denom = float(max(1, initial_violations))
    credits = [process_weight * (float(d) / denom) for d in deltas]
    # Award concise terminal completion credit to the final step that seals v_K == 0
    if credits:
        credits[-1] += terminal_weight
    total = float(sum(credits))
    return TraceCreditResult(
        verified=True,
        gate_passed=True,
        initial_violations=initial_violations,
        final_violations=0,
        think_tokens=think_tokens,
        step_violation_deltas=tuple(deltas),
        step_credits=tuple(credits),
        terminal_reward=terminal_weight,
        total_credit=total,
    )


def _step_env(mask: int, action: int) -> tuple[int, int, bool]:
    """Transition digital-twin violation bitmask for one <think> draft action."""
    tok_cost = ACTION_TOKEN_COSTS[action]
    if action == ACTION_THAW_REAGENT:
        return mask & ~VIOL_FROZEN_REAGENT, tok_cost, False
    if action == ACTION_SEAL_HIGH_TEMP:
        return mask & ~VIOL_UNSEALED_HIGH_TEMP, tok_cost, False
    if action == ACTION_FIX_DAG_ORDER:
        return mask & ~VIOL_DAG_BARRIER_SWAP, tok_cost, False
    if action == ACTION_CALIBRATE_PIPETTE:
        return mask & ~VIOL_PIPETTE_RANGE, tok_cost, False
    if action == ACTION_SPECULATIVE_DETOUR:
        return mask | VIOL_PIPETTE_RANGE, tok_cost, False
    if action == ACTION_PAD_VERBOSE_THINK:
        return mask, tok_cost, False
    if action == ACTION_FINALIZE_CONCISE:
        # Commits immediately once all violations are resolved; otherwise premature no-op.
        if mask == 0:
            return 0, tok_cost, True
        return mask, 24, False
    raise ValueError(f"Unknown action index: {action}")


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def _sample_rollout(
    probs_table: np.ndarray,
    rng: np.random.Generator,
    init_mask: int,
    max_draft_steps: int,
) -> tuple[list[int], list[int], list[int], int]:
    """Sample one multi-step reasoning rollout from tabular softmax policy."""
    states: list[int] = []
    actions: list[int] = []
    post_violations: list[int] = []
    total_tokens = 0
    mask = init_mask

    for _ in range(max_draft_steps):
        p = probs_table[mask]
        act = int(rng.choice(NUM_ACTIONS, p=p))
        next_mask, tok_cost, done = _step_env(mask, act)
        states.append(mask)
        actions.append(act)
        post_violations.append(int(next_mask.bit_count()))
        total_tokens += tok_cost
        mask = next_mask
        if done:
            break

    return states, actions, post_violations, total_tokens


def _train_single_seed(
    mode: str,
    seed: int,
    num_steps: int = 120,
    group_size: int = 24,
    max_draft_steps: int = 8,
    max_think_tokens: int = 240,
    lr: float = 0.32,
) -> dict[str, float]:
    """Run policy-gradient training for one seed under the specified reward regime."""
    rng = np.random.default_rng(seed)
    # Contextual tabular policy pi(a | violation_mask) over the 16 twin states
    logits = np.zeros((16, NUM_ACTIONS), dtype=np.float64)
    full_mask = (
        VIOL_FROZEN_REAGENT
        | VIOL_UNSEALED_HIGH_TEMP
        | VIOL_DAG_BARRIER_SWAP
        | VIOL_PIPETTE_RANGE
    )
    init_masks = (
        full_mask,
        VIOL_FROZEN_REAGENT | VIOL_UNSEALED_HIGH_TEMP | VIOL_DAG_BARRIER_SWAP,
        VIOL_FROZEN_REAGENT | VIOL_UNSEALED_HIGH_TEMP | VIOL_PIPETTE_RANGE,
        VIOL_UNSEALED_HIGH_TEMP | VIOL_DAG_BARRIER_SWAP | VIOL_PIPETTE_RANGE,
    )

    for step_idx in range(num_steps):
        probs = _softmax_rows(logits)
        grad = np.zeros_like(logits)

        rollouts: list[tuple[list[int], list[int], list[int], int, int]] = []
        traj_returns = np.zeros(group_size, dtype=np.float64)

        for g in range(group_size):
            init_mask = init_masks[(step_idx * group_size + g) % len(init_masks)]
            v0 = int(init_mask.bit_count())
            states, actions, post_vs, think_toks = _sample_rollout(
                probs, rng, init_mask=init_mask, max_draft_steps=max_draft_steps
            )
            rollouts.append((states, actions, post_vs, think_toks, v0))

            final_verified = post_vs[-1] == 0
            if mode == "additive_length_bonus":
                # Classic un-gated additive <think> length bonus (susceptible to verbosity hacking)
                task_r = 1.0 if final_verified else 0.0
                len_bonus = 0.95 * min(float(think_toks), 560.0) / 560.0
                traj_returns[g] = task_r + len_bonus
            elif mode == "sparse_terminal_only":
                traj_returns[g] = (
                    1.0 if (final_verified and think_toks <= max_think_tokens) else 0.0
                )
            elif mode == "twin_delta_process_credit":
                res = score_verifiable_trace_credit(
                    initial_violations=v0,
                    step_violations=post_vs,
                    think_tokens=think_toks,
                    max_think_tokens=max_think_tokens,
                )
                traj_returns[g] = res.total_credit
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        if mode in ("additive_length_bonus", "sparse_terminal_only"):
            mean_r = float(np.mean(traj_returns))
            std_r = float(np.std(traj_returns))
            if std_r > 1e-8:
                advs = (traj_returns - mean_r) / (std_r + 1e-6)
                for g, (states, actions, _, _, _) in enumerate(rollouts):
                    adv = float(advs[g])
                    for s, a in zip(states, actions):
                        grad[s] -= adv * probs[s]
                        grad[s, a] += adv
        else:
            # Step-level twin-delta credit assignment with state-conditional baseline
            step_records: list[tuple[int, int, float]] = []
            state_sums = np.zeros(16, dtype=np.float64)
            state_counts = np.zeros(16, dtype=np.float64)

            for states, actions, post_vs, think_toks, v0 in rollouts:
                res = score_verifiable_trace_credit(
                    initial_violations=v0,
                    step_violations=post_vs,
                    think_tokens=think_toks,
                    max_think_tokens=max_think_tokens,
                )
                for k, (s, a) in enumerate(zip(states, actions)):
                    c_k = float(res.step_credits[k])
                    # Only award concise-finalize terminal bonus if finalize action ended at v_K==0
                    if (
                        res.gate_passed
                        and k == len(actions) - 1
                        and a != ACTION_FINALIZE_CONCISE
                    ):
                        c_k -= res.terminal_reward * 0.5
                    step_records.append((s, a, c_k))
                    state_sums[s] += c_k
                    state_counts[s] += 1.0

            all_c = np.array([r[2] for r in step_records], dtype=np.float64)
            std_c = float(np.std(all_c))
            if std_c > 1e-8:
                state_means = np.divide(
                    state_sums,
                    np.maximum(state_counts, 1.0),
                )
                for s, a, c_k in step_records:
                    adv = (c_k - float(state_means[s])) / (std_c + 1e-6)
                    grad[s] -= adv * probs[s]
                    grad[s, a] += adv

        logits += lr * (grad / float(group_size))

    # Deterministic evaluation rollouts across 200 episodes
    eval_probs = _softmax_rows(logits)
    eval_rng = np.random.default_rng(seed + 10_000)
    verified_hits = 0
    raw_solve_hits = 0
    token_lengths: list[int] = []

    for ep in range(200):
        init_mask = init_masks[ep % len(init_masks)]
        _, _, post_vs, think_toks = _sample_rollout(
            eval_probs, eval_rng, init_mask=init_mask, max_draft_steps=max_draft_steps
        )
        is_solved = post_vs[-1] == 0
        raw_solve_hits += int(is_solved)
        verified_hits += int(is_solved and think_toks <= max_think_tokens)
        token_lengths.append(think_toks)

    return {
        "verified_repair_accuracy_pct": 100.0 * float(verified_hits) / 200.0,
        "ungated_solve_pct": 100.0 * float(raw_solve_hits) / 200.0,
        "mean_think_tokens": float(np.mean(token_lengths)),
    }


def run_benchmark(
    num_seeds: int = 5,
    num_steps: int = 120,
    max_think_tokens: int = 240,
) -> dict[str, Any]:
    """Simulate 5-seed policy-gradient training across 120 steps for all 3 regimes."""
    modes = (
        "additive_length_bonus",
        "sparse_terminal_only",
        "twin_delta_process_credit",
    )
    regimes: dict[str, dict[str, Any]] = {}

    for mode in modes:
        seed_runs = [
            _train_single_seed(
                mode=mode,
                seed=100 + s,
                num_steps=num_steps,
                max_think_tokens=max_think_tokens,
            )
            for s in range(num_seeds)
        ]
        accs = [r["verified_repair_accuracy_pct"] for r in seed_runs]
        toks = [r["mean_think_tokens"] for r in seed_runs]
        regimes[mode] = {
            "verified_repair_accuracy_pct": round(float(np.mean(accs)), 2),
            "verified_repair_accuracy_std_pct": round(float(np.std(accs)), 2),
            "mean_think_tokens": round(float(np.mean(toks)), 2),
            "std_think_tokens": round(float(np.std(toks)), 2),
            "per_seed": seed_runs,
        }

    return {
        "idea": "Idea #14: Twin-Delta Step-Level Process Credit & Anti-<think>-Bloat Shaping",
        "num_seeds": num_seeds,
        "num_steps": num_steps,
        "max_think_tokens": max_think_tokens,
        "regimes": regimes,
        "accuracy_gain_vs_sparse_pct": round(
            regimes["twin_delta_process_credit"]["verified_repair_accuracy_pct"]
            - regimes["sparse_terminal_only"]["verified_repair_accuracy_pct"],
            2,
        ),
        "think_token_reduction_vs_additive_pct": round(
            100.0
            * (
                regimes["additive_length_bonus"]["mean_think_tokens"]
                - regimes["twin_delta_process_credit"]["mean_think_tokens"]
            )
            / max(1.0, regimes["additive_length_bonus"]["mean_think_tokens"]),
            2,
        ),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_benchmark(), indent=2))
