"""KL estimators between the current policy and a frozen reference.

The reference is the SFT checkpoint. See docs/posttrain/kl_anchor.md.

Names follow Schulman, "Approximating KL Divergence" (2020),
http://joschu.net/blog/kl-approx.html, and the NeMo RL loss block
(`docs/guides/grpo.md`, On-Policy KL Approximation; default
`reference_policy_kl_type: k3` in `examples/configs/grpo_math_1B.yaml`).

`kl_k1`, `kl_k2`, `kl_k3`, and `reverse_kl_loss_term` operate per token:
- `reverse_kl_loss_term` yields an unbiased Reverse-KL gradient per conditional
  token step (`E_{y_t ~ pi}[grad reverse_kl_loss_term_t] = grad KL(pi_t || pi_ref,t)`),
  matching the GPU recipes (`yaniguan/RL` `reference_policy_kl_type: k3`).
- Summing per-token terms over a multi-token completion omits the causal term where
  an early token changes the conditional KL of later tokens (biased at the sequence
  level; arXiv:2506.09477, #54).
- `sequence_reverse_kl_reward_to_go` and `sequence_reverse_kl_loss_term`
  (`kl_mode="sequence_exact"`) restore the exact multi-step sequence-level gradient
  `grad KL(Pi_theta(y|x) || Pi_ref(y|x))` by carrying future KL suffix sums.
"""
from __future__ import annotations

import sys

import torch

# Defaults. KL is on. These are not zero.
# DPO beta is the original Rafailov default (0.1): it multiplies the log-ratio
# margin, so 0.1 is a moderate pull and 0 removes the preference gradient.
# PPO subtracts KL from a reward that lives in [0, 1]. 0.04 nats is a visible
# tax on a token that has wandered, close to the InstructGPT scale (0.02) but
# a bit tighter because a lab policy should keep the scientist's format.
# GRPO adds KL on top of standardised advantages. NeMo's recipe uses 0.01
# (`examples/configs/grpo_math_1B.yaml`, loss_fn.reference_policy_kl_penalty).
# The same number is not the same strength in the three losses.
DEFAULT_KL_COEF = {"dpo": 0.1, "ppo": 0.04, "grpo": 0.01}
DEFAULT_KL_ESTIMATOR = "k3"


def warn_if_kl_disabled(coef: float, method: str) -> None:
    """Print a loud line when a run turns the KL anchor off."""
    if coef is None or float(coef) != 0.0:
        return
    print(
        f"WARNING: {method} KL coefficient is 0. This is an ablation, not a recommendation. "
        "The policy can drift away from the SFT reference. "
        "In a lab that has no absolute ground truth, that drift is the failure mode.",
        file=sys.stderr,
        flush=True,
    )


def kl_k1(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """log pi - log pi_ref.

    Unbiased in expectation, but a single sample can be negative.
    The gradient w.r.t. log pi is +1 everywhere, so gradient descent always
    lowers log pi. It does not pull a too-small probability back up.
    """
    return logp - logp_ref


def kl_k2(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Half the squared log-ratio. Always >= 0.

    Gradient w.r.t. log pi is (log pi - log pi_ref), so descent walks log pi
    toward the reference. This is a local approximation of KL, not an
    unbiased estimator.
    """
    delta = logp - logp_ref
    return 0.5 * delta * delta


def kl_k3(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """pi_ref/pi - log(pi_ref/pi) - 1. Always >= 0.

    This is the low-variance estimator. Under samples from pi it is unbiased
    for KL(pi || pi_ref): E[pi_ref/pi] = 1, so the expectation collapses to
    E[log pi - log pi_ref].

    Let u = log pi_ref - log pi. Then k3 = exp(u) - u - 1.
    d(k3)/d(log pi) = 1 - pi_ref/pi.
    Descent decreases log pi when pi > pi_ref and increases it when pi < pi_ref.
    """
    log_ratio = logp_ref - logp  # log(pi_ref / pi)
    return torch.exp(log_ratio) - log_ratio - 1.0


def kl_term(logp: torch.Tensor, logp_ref: torch.Tensor, estimator: str = "k3") -> torch.Tensor:
    fn = {"k1": kl_k1, "k2": kl_k2, "k3": kl_k3}[estimator]
    return fn(logp, logp_ref.detach())


def reverse_kl_loss_term(
    logp: torch.Tensor,
    logp_ref: torch.Tensor,
    *,
    estimator: str = "k3",
    logp_behavior: torch.Tensor | None = None,
    input_clamp_value: float | None = 20.0,
    output_clamp_value: float | None = 10.0,
) -> torch.Tensor:
    """Per-token Reverse-KL surrogate matching NeMo-RL `calculate_kl` + `ClippedPGLossFn`.

    Differentiating `kl_k3(logp, logp_ref)` directly w.r.t. `logp` yields
    `1 - pi_ref/pi`, whose expectation under `a ~ pi` is the gradient of the
    *Forward* KL `KL(pi_ref || pi)`, NOT Reverse KL `KL(pi || pi_ref)`.

    Multiplying `k1`/`k3` by a value-1 score-function factor `w = exp(logp - sg(logp))`
    (or off-policy importance weight `exp(logp - sg(logp_behavior))`) restores the
    unbiased Reverse-KL gradient **per conditional token step** via the product rule:
        d/d(logp) [w * (exp(u) - 1 - u)]|_{w=1} = (exp(u) - 1 - u) + (1 - exp(u))
                                                = -u = log(pi / pi_ref).

    Note (#54): on multi-token autoregressive completions, summing this per-token term
    across steps is unbiased on the final token and partial on earlier tokens because
    it drops the effect of token `t` on future conditional KL terms `s > t`
    (arXiv:2506.09477). Use `sequence_reverse_kl_reward_to_go` or
    `sequence_reverse_kl_loss_term` (`kl_mode="sequence_exact"`) when an exact
    sequence-level Reverse-KL gradient is required.
    """
    if estimator not in ("k1", "k2", "k3"):
        raise KeyError(estimator)
    ref = logp_ref.detach()
    raw_logr = ref - logp  # u = log(pi_ref / pi)

    if estimator == "k2":
        if logp_behavior is None:
            w = None
        else:
            w = torch.exp((logp.detach() - logp_behavior.detach()).clamp(-20.0, 20.0))
    else:
        if logp_behavior is None:
            w = torch.exp(logp - logp.detach())
        else:
            log_w = logp - logp_behavior.detach()
            log_w_clamped = log_w.clamp(-20.0, 20.0)
            w = torch.exp(log_w_clamped)
            w = torch.where(log_w == log_w_clamped, w, w.detach())

    if input_clamp_value is not None:
        logr_clamped = raw_logr.clamp(min=-input_clamp_value, max=input_clamp_value)
        if w is not None:
            w = torch.where(raw_logr == logr_clamped, w, w.detach())
        logr = logr_clamped
    else:
        logr = raw_logr

    if estimator == "k1":
        kl = -logr
    elif estimator == "k2":
        kl = 0.5 * logr * logr
    else:
        kl = torch.exp(logr) - 1.0 - logr

    if w is not None:
        kl = w * kl
    if output_clamp_value is not None:
        kl = kl.clamp(min=-output_clamp_value, max=output_clamp_value)
    return kl


def sequence_reverse_kl_reward_to_go(
    logp: torch.Tensor,
    logp_ref: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Detached per-token Reverse-KL suffix sum `sum_{s >= t} m_s * sg(log pi_s - log pi_ref,s)` (#54).

    When weighted by `logp` in a policy-gradient objective (or negated into token
    reward-to-go), its expectation under `y ~ Pi_theta(.|x)` equals the exact
    sequence-level Reverse-KL gradient `grad_theta KL(Pi_theta || Pi_ref)`:
    past terms `s < t` have zero expectation under `E_{y_t ~ pi_t}[grad log pi_t] = 0`,
    while future terms `s > t` carry the state-transition KL coupling.
    """
    m = torch.ones_like(logp) if mask is None else mask.to(logp.dtype)
    per_tok = (logp.detach() - logp_ref.detach()) * m
    rtg = per_tok.flip(-1).cumsum(-1).flip(-1)
    return rtg * m


def sequence_reverse_kl_loss_term(
    logp: torch.Tensor,
    logp_ref: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    estimator: str = "k3",
    logp_behavior: torch.Tensor | None = None,
    input_clamp_value: float | None = 20.0,
    output_clamp_value: float | None = 10.0,
) -> torch.Tensor:
    """Sequence-exact Reverse-KL surrogate combining step-`t` KL with future KL suffix (#54).

    Forward value equals `reverse_kl_loss_term(...) * mask` (so logged KL magnitudes stay
    unchanged), while the backward pass adds `(logp - sg(logp)) * sum_{s > t} m_s * sg(k_s)`
    so `E_{y ~ Pi_theta}[grad sum_t sequence_reverse_kl_loss_term_t]` equals the exact
    multi-step sequence gradient `grad_theta KL(Pi_theta || Pi_ref)` for both `k1` and `k3`.
    """
    if estimator not in ("k1", "k3"):
        raise ValueError(
            f"sequence_reverse_kl_loss_term requires unbiased estimator 'k1' or 'k3', got {estimator!r}"
        )
    m = torch.ones_like(logp) if mask is None else mask.to(logp.dtype)
    local_term = reverse_kl_loss_term(
        logp,
        logp_ref,
        estimator=estimator,
        logp_behavior=logp_behavior,
        input_clamp_value=input_clamp_value,
        output_clamp_value=output_clamp_value,
    ) * m

    ref = logp_ref.detach()
    raw_u = ref - logp.detach()
    if input_clamp_value is not None:
        raw_u = raw_u.clamp(min=-input_clamp_value, max=input_clamp_value)
    fut_k = (-raw_u) if estimator == "k1" else (torch.exp(raw_u) - 1.0 - raw_u)
    if output_clamp_value is not None:
        fut_k = fut_k.clamp(min=-output_clamp_value, max=output_clamp_value)

    if logp_behavior is not None:
        log_iw = (logp.detach() - logp_behavior.detach()) * m
        prefix_iw = torch.exp(log_iw.cumsum(-1).clamp(-20.0, 20.0))
        weighted_fut = fut_k * prefix_iw * m
    else:
        weighted_fut = fut_k * m

    inclusive_suffix = weighted_fut.flip(-1).cumsum(-1).flip(-1)
    strict_future_suffix = (inclusive_suffix - weighted_fut) * m
    zero_val_score = (logp - logp.detach()) * m
    return local_term + zero_val_score * strict_future_suffix.detach()


def kl_grad_wrt_logp(logp: torch.Tensor, logp_ref: torch.Tensor, estimator: str) -> torch.Tensor:
    """Closed form used by the unit tests. Same shape as logp."""
    if estimator == "k1":
        return torch.ones_like(logp)
    if estimator == "k2":
        return logp - logp_ref
    if estimator == "k3":
        return 1.0 - torch.exp(logp_ref - logp)
    raise KeyError(estimator)


class AdaptiveKLController:
    """Move the KL coefficient toward a target KL.

    Same idea as the controller in Ziegler et al. (2019) and InstructGPT:
    a clipped proportional step. If the measured KL is above the target, the
    coefficient grows and the next update is punished more.

    `rate` is the step size on the clipped error. The error is clipped to
    +/- 0.2 so one noisy batch cannot explode the coefficient.
    """

    def __init__(self, coef: float, target: float, rate: float = 0.2):
        if coef < 0:
            raise ValueError("KL coefficient must be >= 0")
        if target <= 0:
            raise ValueError("target KL must be > 0; a target of 0 is the KL-off ablation")
        self.coef = float(coef)
        self.target = float(target)
        self.rate = float(rate)

    def update(self, measured_kl: float) -> float:
        error = measured_kl / self.target - 1.0
        error = max(-0.2, min(0.2, error))
        self.coef *= 1.0 + self.rate * error
        return self.coef
