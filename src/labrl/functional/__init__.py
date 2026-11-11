"""Pure functional stateless post-training math kernels (`labrl.functional`).

Strictly separated from model state, dataloaders, and distributed engines,
matching `nemo_rl.algorithms.loss` and `nemo_rl.algorithms.advantage_estimator`.
"""
from posttrain.advantages import (
    clip_advantages,
    extract_reasoning_span,
    gae,
    gdpo_advantages,
    grpo_advantages,
    reasoning_length_penalty,
    reinforce_pp_advantages,
    vapo_length_adaptive_lambda,
    zero_variance_mask,
)
from posttrain.distill_opd import tropd_token_advantages
from posttrain.kl import AdaptiveKLController, kl_k1, kl_k2, kl_k3, kl_term, reverse_kl_loss_term
from posttrain.losses import (
    compute_is_weights,
    dpo_loss,
    likelihood_displacement,
    policy_surrogate,
    reduce_token_values,
    sft_nll,
    token_logprobs,
    value_loss,
)
from posttrain.mpo import mpo_loss, update_reward_shift
from scientist.distill import distill_step_loss

__all__ = [
    "AdaptiveKLController",
    "clip_advantages",
    "compute_is_weights",
    "distill_step_loss",
    "dpo_loss",
    "extract_reasoning_span",
    "gae",
    "gdpo_advantages",
    "grpo_advantages",
    "kl_k1",
    "kl_k2",
    "kl_k3",
    "kl_term",
    "likelihood_displacement",
    "mpo_loss",
    "policy_surrogate",
    "reasoning_length_penalty",
    "reduce_token_values",
    "reinforce_pp_advantages",
    "reverse_kl_loss_term",
    "sft_nll",
    "token_logprobs",
    "tropd_token_advantages",
    "update_reward_shift",
    "value_loss",
    "vapo_length_adaptive_lambda",
    "zero_variance_mask",
]
