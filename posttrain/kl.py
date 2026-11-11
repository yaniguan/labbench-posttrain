"""KL estimators between the current policy and a frozen reference.

The reference is the SFT checkpoint. See docs/posttrain/kl_anchor.md.

Names follow Schulman, "Approximating KL Divergence" (2020),
http://joschu.net/blog/kl-approx.html, and the NeMo RL loss block
(`docs/guides/grpo.md`, On-Policy KL Approximation; default
`reference_policy_kl_type: k3` in `examples/configs/grpo_math_1B.yaml`).

All three functions take token log-probs and return a per-token tensor.
The expectation of k1 and of k3 under the current policy equals
KL(pi || pi_ref). k2 is a local quadratic approximation. It is not unbiased.
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
