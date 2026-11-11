"""Loss, advantage, and KL tests. No downloads."""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from posttrain.advantages import (
    apply_overlong_filter,
    apply_stop_properly,
    clip_advantages,
    gae,
    gdpo_advantages,
    grpo_advantages,
    group_length_penalty,
    reinforce_pp_advantages,
    zero_variance_mask,
)
from posttrain.kl import (
    DEFAULT_KL_COEF,
    AdaptiveKLController,
    kl_grad_wrt_logp,
    kl_k1,
    kl_k2,
    kl_k3,
    warn_if_kl_disabled,
)
from posttrain.losses import dpo_loss, policy_surrogate, reduce_token_values, sft_nll


def test_sft_prompt_tokens_have_zero_logit_gradient():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 7, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    # Predict positions 3 and 4 only. Positions 1 and 2 are prompt targets.
    comp = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]])
    loss = sft_nll(logits, ids, comp, token_level=True)
    loss.backward()
    # logits[:, t] predicts token t+1.
    assert torch.all(logits.grad[0, 0] == 0)
    assert torch.all(logits.grad[0, 1] == 0)
    assert torch.any(logits.grad[0, 2] != 0)


def test_token_vs_sequence_reduction():
    values = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    # Tokens: 1, 1, 0. Mean 2/3. Sequences: mean(1, 1)=1 and mean(0)=0. Mean 0.5.
    assert torch.allclose(reduce_token_values(values, mask, token_level=True), torch.tensor(2 / 3))
    assert torch.allclose(reduce_token_values(values, mask, token_level=False), torch.tensor(0.5))


def test_dpo_gradient_matches_formula_and_finite_difference():
    chosen = torch.tensor([0.3, -0.2], requires_grad=True)
    rejected = torch.tensor([-0.1, 0.4], requires_grad=True)
    ref_c = torch.tensor([0.0, 0.0])
    ref_r = torch.tensor([0.1, -0.2])
    beta = 0.1
    loss, _ = dpo_loss(chosen, rejected, ref_c, ref_r, beta)
    loss.backward()
    margin = (chosen.detach() - ref_c) - (rejected.detach() - ref_r)
    weight = torch.sigmoid(beta * margin)
    n = chosen.numel()
    # d(-log sigmoid(beta * m))/d logp_chosen = -(1 - sigmoid(beta * m)) * beta / n
    expected = -(1 - weight) * beta / n
    assert torch.allclose(chosen.grad, expected, atol=1e-5)
    assert torch.allclose(rejected.grad, -expected, atol=1e-5)

    def value(c, r):
        out, _ = dpo_loss(c, r, ref_c, ref_r, beta)
        return out

    eps = 1e-4
    base_c = chosen.detach()
    base_r = rejected.detach()
    fd = (value(base_c + torch.tensor([eps, 0.0]), base_r) - value(base_c - torch.tensor([eps, 0.0]), base_r)) / (2 * eps)
    assert torch.allclose(chosen.grad[0], fd, atol=1e-4)


def test_clip_region_gradient_is_zero_and_cispo_is_not():
    logp_old = torch.tensor([0.0])
    advantage = torch.tensor([1.0])
    logp = torch.tensor([1.0], requires_grad=True)  # ratio = e > 1.2
    loss = policy_surrogate(logp, logp_old, advantage, eps_low=0.2, eps_high=0.2, clip=True)
    loss.sum().backward()
    assert logp.grad.item() == 0.0

    logp_c = torch.tensor([1.0], requires_grad=True)
    cispo = policy_surrogate(logp_c, logp_old, advantage, eps_low=0.2, eps_high=0.2, clip=True, cispo=True)
    cispo.sum().backward()
    assert logp_c.grad.item() != 0.0

    logp_off = torch.tensor([1.0], requires_grad=True)
    unclipped = policy_surrogate(logp_off, logp_old, advantage, clip=False)
    unclipped.sum().backward()
    assert logp_off.grad.item() != 0.0


def test_asymmetric_clip_binds_only_on_the_tight_side():
    # ratio = exp(0.15) ≈ 1.16. Upper clip at 1.1 binds. Upper clip at 1.3 does not.
    logp = torch.tensor([0.15], requires_grad=True)
    old = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    tight = policy_surrogate(logp, old, adv, eps_low=0.2, eps_high=0.1, clip=True)
    tight.sum().backward()
    assert logp.grad.item() == 0.0
    logp2 = torch.tensor([0.15], requires_grad=True)
    wide = policy_surrogate(logp2, old, adv, eps_low=0.2, eps_high=0.3, clip=True)
    wide.sum().backward()
    assert logp2.grad.item() != 0.0


def test_lower_clip_binds_for_negative_advantage():
    # ratio = exp(-1) < 0.8 and advantage is negative, so the lower clip binds.
    logp = torch.tensor([-1.0], requires_grad=True)
    loss = policy_surrogate(logp, torch.tensor([0.0]), torch.tensor([-1.0]), 0.2, 0.2, clip=True)
    loss.sum().backward()
    assert logp.grad.item() == 0.0


def test_grpo_advantages_sum_to_zero_for_a_shared_scale():
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.2])
    groups = torch.zeros(4, dtype=torch.long)
    for baseline in ("mean", "loo"):
        for normalize in (False, True):
            if baseline == "loo" and normalize:
                continue
            adv = grpo_advantages(rewards, groups, baseline=baseline, normalize_std=normalize)
            assert torch.allclose(adv.sum(), torch.tensor(0.0), atol=1e-5), (baseline, normalize, adv)
    # Per-row leave-one-out stds differ, so the sum is not zero.
    skewed = torch.tensor([0.0, 0.0, 1.0, 5.0])
    loo = grpo_advantages(skewed, groups, baseline="loo", normalize_std=True)
    assert not torch.allclose(loo.sum(), torch.tensor(0.0), atol=1e-3)


def test_two_groups_each_sum_to_zero():
    rewards = torch.tensor([1.0, 0.0, 0.5, 3.0, 1.0])
    groups = torch.tensor([0, 0, 0, 1, 1])
    adv = grpo_advantages(rewards, groups, baseline="mean", normalize_std=True)
    assert torch.allclose(adv[:3].sum(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(adv[3:].sum(), torch.tensor(0.0), atol=1e-5)


def test_dynamic_sampling_drops_zero_variance_groups():
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0])
    groups = torch.tensor([0, 0, 0, 1, 1])
    drop = zero_variance_mask(rewards, groups)
    assert torch.equal(drop, torch.tensor([True, True, True, False, False]))
    assert int(drop.sum()) == 3


def test_stop_properly_and_overlong_and_advantage_clip():
    rewards = torch.tensor([1.0, 0.5])
    truncated = torch.tensor([True, False])
    stopped = apply_stop_properly(rewards, truncated, 0.0)
    assert torch.equal(stopped, torch.tensor([0.0, 0.5]))
    kept = apply_stop_properly(rewards, truncated, 1.0)
    assert torch.equal(kept, rewards)
    mask = torch.ones(2, 3)
    filtered = apply_overlong_filter(mask, truncated, enabled=True)
    assert torch.all(filtered[0] == 0)
    assert torch.all(filtered[1] == 1)
    clipped = clip_advantages(torch.tensor([10.0, -10.0, 0.2]), -1.0, 1.0)
    assert torch.equal(clipped, torch.tensor([1.0, -1.0, 0.2]))


def test_reinforce_pp_and_gdpo():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    groups = torch.tensor([0, 0, 1, 1])
    adv = reinforce_pp_advantages(rewards, groups, minus_baseline=True)
    assert abs(float(adv.mean())) < 1e-5
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-4
    comps = {
        "reward/correct": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "reward/format": torch.tensor([0.0, 100.0, 0.0, 100.0]),
    }
    mixed = gdpo_advantages(comps, groups, normalize_std=True)
    # After per-component norm the 100-scale term does not dominate the 0/1 term
    # by two orders of magnitude. The batch std is 1.
    assert abs(float(mixed.std(unbiased=False)) - 1.0) < 1e-4


def test_length_penalty_skips_graded_rewards():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.4, 0.6])
    lengths = torch.tensor([10.0, 2.0, 5.0, 3.0, 9.0])
    groups = torch.tensor([0, 0, 0, 1, 1])
    out, skipped = group_length_penalty(rewards, lengths, groups, coef=0.2)
    assert skipped == 1
    assert out[1].item() == pytest.approx(1.0)  # shortest correct completion
    assert out[0].item() < 1.0  # longest correct completion pays the penalty
    assert out[2].item() == pytest.approx(0.0)
    assert out[3].item() == pytest.approx(0.4)  # graded group unchanged


def test_gae_puts_outcome_on_the_last_token():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.zeros(1, 3)
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    adv = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    # With V=0, gamma=1, lambda=1, every token's advantage is the remaining return, which is 1.
    assert torch.allclose(adv, torch.ones(1, 3))


def test_kl_estimators_value_and_gradient_on_a_toy_distribution():
    # Two-outcome distribution. Expectation of k1 and of k3 equals KL(pi || ref).
    pi = torch.tensor([0.7, 0.3])
    ref = torch.tensor([0.5, 0.5])
    true_kl = torch.sum(pi * (pi.log() - ref.log()))
    logp = pi.log()
    logp_ref = ref.log()
    k1 = torch.sum(pi * kl_k1(logp, logp_ref))
    k3 = torch.sum(pi * kl_k3(logp, logp_ref))
    assert torch.allclose(k1, true_kl, atol=1e-5)
    assert torch.allclose(k3, true_kl, atol=1e-5)
    assert torch.all(kl_k2(logp, logp_ref) >= 0)
    assert torch.all(kl_k3(logp, logp_ref) >= 0)
    for name, fn in (("k1", kl_k1), ("k2", kl_k2), ("k3", kl_k3)):
        leaf = logp.detach().clone().requires_grad_(True)
        fn(leaf, logp_ref).sum().backward()
        assert torch.allclose(leaf.grad, kl_grad_wrt_logp(leaf.detach(), logp_ref, name), atol=1e-5)


def test_k3_gradient_pulls_toward_the_reference():
    # One step is enough. A large step can jump past the reference, and the
    # sign of the gradient then flips, which is the same pull.
    below = torch.tensor([-2.0], requires_grad=True)
    ref_high = torch.tensor([-0.2])
    kl_k3(below, ref_high).backward()
    assert below.grad.item() < 0
    with torch.no_grad():
        below_next = below - 0.05 * below.grad
    assert float(below_next) > float(below.detach())

    above = torch.tensor([1.0], requires_grad=True)
    ref_low = torch.tensor([-0.5])
    kl_k3(above, ref_low).backward()
    assert above.grad.item() > 0
    with torch.no_grad():
        above_next = above - 0.05 * above.grad
    assert float(above_next) < float(above.detach())


def test_k1_gradient_does_not_pull_up():
    logp = torch.tensor([-2.0], requires_grad=True)
    ref = torch.tensor([0.0])
    kl_k1(logp, ref).backward()
    # +1 everywhere. Descent lowers log pi even when it is already below the reference.
    assert logp.grad.item() == pytest.approx(1.0)


def test_adaptive_controller_moves_toward_the_target():
    high = AdaptiveKLController(coef=0.04, target=0.02)
    assert high.update(0.08) > 0.04
    low = AdaptiveKLController(coef=0.04, target=0.02)
    assert low.update(0.001) < 0.04


def test_kl_defaults_are_on():
    assert DEFAULT_KL_COEF["dpo"] > 0
    assert DEFAULT_KL_COEF["ppo"] > 0
    assert DEFAULT_KL_COEF["grpo"] > 0
    from posttrain.dpo import train_dpo
    from posttrain.grpo import train_grpo
    from posttrain.ppo import train_ppo
    assert inspect.signature(train_dpo).parameters["beta"].default > 0
    assert inspect.signature(train_ppo).parameters["kl_coef"].default > 0
    assert inspect.signature(train_grpo).parameters["kl_coef"].default > 0
    assert inspect.signature(train_ppo).parameters["kl_in_reward"].default is True
    assert inspect.signature(train_grpo).parameters["kl_in_reward"].default is False


def test_zero_kl_prints_a_warning(capsys):
    warn_if_kl_disabled(0.0, "grpo")
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "ablation" in err
    warn_if_kl_disabled(0.01, "grpo")
    assert capsys.readouterr().err == ""


def test_zero_beta_warning_comes_from_the_trainer(capsys):
    from posttrain.dpo import TOY_PAIRS, train_dpo
    from posttrain.tiny import build_tiny_lm
    model, tok = build_tiny_lm(seed=1)
    train_dpo(TOY_PAIRS, model=model, tokenizer=tok, beta=0.0, steps=1, seed=1)
    err = capsys.readouterr().err
    assert "WARNING" in err and "dpo" in err


def test_true_kl_matches_manual_sum():
    # Guard against a silent formula change. The number is computed here.
    pi = torch.tensor([0.7, 0.3])
    ref = torch.tensor([0.5, 0.5])
    manual = sum(float(pi[i]) * math.log(float(pi[i]) / float(ref[i])) for i in range(2))
    got = float(torch.sum(pi * kl_k3(pi.log(), ref.log())))
    assert got == pytest.approx(manual, abs=1e-6)
