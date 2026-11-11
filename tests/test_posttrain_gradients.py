"""Gradient and loss checks on tiny tensors. No download."""
import math

import torch
import torch.nn.functional as F

from posttrain.dpo import dpo_logit_grad_weight, dpo_loss, likelihood_displacement
from posttrain.grpo import clipped_token_objective, group_advantages, reduce_loss
from posttrain.logging_utils import clip_fraction_by_bucket, clipped_token_mask
from posttrain.ppo import clipped_policy_loss, gae, schulman_kl, value_loss
from posttrain.sft import masked_prompt_logit_grad, sft_nll
from posttrain.tiny import build_tiny


def test_sft_nll_matches_hand_count():
    # logit t predicts token t+1. Prompt target is ignored.
    logits = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]], dtype=torch.float)
    labels = torch.tensor([[-100, 0, 1]])
    loss, stats = sft_nll(logits, labels, normalize="token")
    # targets after the shift: 0 predicted from [0, 0], and 1 predicted from [2, 0]
    first = float(-F.log_softmax(logits[0, 0], dim=-1)[0])
    second = float(-F.log_softmax(logits[0, 1], dim=-1)[1])
    assert loss.item() == pytest_approx((first + second) / 2)
    assert float(stats["n_completion_tokens"]) == 2


def test_sft_sequence_norm_weights_examples_equally():
    logits = torch.zeros(2, 4, 2)
    # example 0 has 1 completion token, example 1 has 3
    labels = torch.tensor([
        [-100, -100, -100, 0],
        [-100, 1, 1, 1],
    ])
    token_loss, _ = sft_nll(logits, labels, normalize="token")
    seq_loss, _ = sft_nll(logits, labels, normalize="sequence")
    # Uniform logits: every token NLL is log(2). Token mean is log(2).
    # Sequence mean is also log(2), because each sequence's mean is log(2).
    assert token_loss.item() == pytest_approx(math.log(2))
    assert seq_loss.item() == pytest_approx(math.log(2))
    # Make the long example's logits confident and wrong, so token-norm feels it more.
    logits2 = logits.clone()
    logits2[1, :, 0] = 5.0
    labels2 = labels.clone()
    # target class 1 is now the low logit on the long example
    token_loss2, _ = sft_nll(logits2, labels2, normalize="token")
    seq_loss2, _ = sft_nll(logits2, labels2, normalize="sequence")
    assert token_loss2.item() > seq_loss2.item()


def test_prompt_token_targets_have_zero_logit_grad():
    torch.manual_seed(0)
    model, tok = build_tiny(seed=0)
    from posttrain.batching import collate_labeled

    batch = collate_labeled([("Incubate at", " 37 C.")], tok, max_len=32)
    stats = masked_prompt_logit_grad(model, batch["input_ids"], batch["labels"], batch["attention_mask"])
    assert stats["prompt_target_grad_abs_max"] == 0.0
    assert stats["completion_grad_abs_max"] > 0.0


def test_dpo_grad_matches_formula_and_finite_difference():
    torch.manual_seed(0)
    pi_c = torch.tensor([-1.2, -0.4], requires_grad=True)
    pi_r = torch.tensor([-1.0, -2.0], requires_grad=True)
    ref_c = torch.tensor([-1.3, -0.5])
    ref_r = torch.tensor([-1.1, -1.5])
    beta = 0.1
    loss, stats = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=beta)
    loss.backward()
    margin = stats["margin"]
    # Batch mean, so each pair's d(loss)/d(margin) is the formula divided by batch size.
    expected = dpo_logit_grad_weight(margin, beta) / margin.numel()
    # Recover dL/d margin from dL/d pi_c, since margin = (pi_c - pi_r) - constant.
    assert torch.allclose(pi_c.grad, expected, atol=1e-6)
    assert torch.allclose(pi_r.grad, -expected, atol=1e-6)
    # Finite difference on the first chosen log-prob.
    eps = 1e-4

    def at(delta):
        loss_d, _ = dpo_loss(pi_c.detach().clone() + torch.tensor([delta, 0.0]), pi_r.detach(), ref_c, ref_r, beta=beta)
        return loss_d.item()

    numerical = (at(eps) - at(-eps)) / (2 * eps)
    assert numerical == pytest_approx(pi_c.grad[0].item(), abs=1e-4)


def test_likelihood_displacement_flags_both_falling():
    assert likelihood_displacement(-1.0, -1.2, -1.4, -2.0) is True
    assert likelihood_displacement(-1.0, -1.2, -0.5, -2.0) is False  # chosen rose
    assert likelihood_displacement(-1.0, -1.2, -1.4, -1.3) is False  # gap did not grow


def test_clip_region_gradient_is_zero():
    eps = 0.2
    # ratio 1.5, advantage +1: above the upper clip, gradient must be 0
    logp = torch.tensor([0.0], requires_grad=True)
    old = torch.tensor([-math.log(1.5)])
    loss, ratio = clipped_policy_loss(logp, old, torch.tensor([1.0]), eps, use_clip=True)
    loss.backward()
    assert ratio.item() == pytest_approx(1.5)
    assert logp.grad.item() == 0.0
    # ratio 0.5, advantage -1: below the lower clip, gradient must be 0
    logp2 = torch.tensor([0.0], requires_grad=True)
    old2 = torch.tensor([-math.log(0.5)])
    loss2, _ = clipped_policy_loss(logp2, old2, torch.tensor([-1.0]), eps, use_clip=False)
    # clip off: the same point has a gradient
    loss2.backward()
    assert logp2.grad.item() != 0.0
    logp3 = torch.tensor([0.0], requires_grad=True)
    loss3, _ = clipped_policy_loss(logp3, old2.detach(), torch.tensor([-1.0]), eps, use_clip=True)
    loss3.backward()
    assert logp3.grad.item() == 0.0
    # advantage negative and ratio high: the unclipped term is worse, so the gradient is NOT zero
    logp4 = torch.tensor([0.0], requires_grad=True)
    loss4, _ = clipped_policy_loss(logp4, old.detach(), torch.tensor([-1.0]), eps, use_clip=True)
    loss4.backward()
    assert logp4.grad.item() != 0.0


def test_grpo_clip_matches_ppo_clip():
    logp = torch.tensor([[0.0, 0.1]])
    old = torch.tensor([[-0.2, 0.4]])
    adv = torch.tensor([0.5])
    a, ra = clipped_policy_loss(logp, old, adv, 0.2, True)
    b, rb = clipped_token_objective(logp, old, adv, 0.2, True)
    assert torch.allclose(a, b)
    assert torch.allclose(ra, rb)


def test_grpo_advantages_sum_to_zero_and_zero_variance_is_counted():
    rewards = torch.tensor([[1.0, 0.0, 0.5, 0.5], [0.2, 0.2, 0.2, 0.2]])
    adv, zero = group_advantages(rewards, std_norm=True)
    assert torch.allclose(adv[0].sum(), torch.tensor(0.0), atol=1e-5)
    assert zero.tolist() == [False, True]
    assert torch.allclose(adv[1], torch.zeros(4))
    adv_off, zero_off = group_advantages(rewards, std_norm=False)
    assert torch.allclose(adv_off[0].sum(), torch.tensor(0.0), atol=1e-5)
    assert zero_off.tolist() == [False, True]
    # A zero group dropped from the mask does not change a token-mean loss of the other group.
    per = torch.tensor([[1.0, 1.0], [3.0, 3.0], [9.0, 9.0], [9.0, 9.0]])
    mask_kept = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    # The kept rows are the first group. Their mean is 1 and 3 -> token mean (1+1+3+3)/4 = 2.
    assert reduce_loss(per, mask_kept, "token", max_len=8).item() == pytest_approx(2.0)


def test_kl_gradient_matches_formula():
    logp = torch.tensor(0.3, requires_grad=True)
    ref = torch.tensor(-0.2)
    kl = schulman_kl(logp, ref)
    kl.backward()
    expected = 1.0 - torch.exp(ref - logp.detach())
    assert logp.grad.item() == pytest_approx(expected.item(), abs=1e-6)


def test_gae_backs_up_a_terminal_reward():
    rewards = torch.tensor([[0.0, 0.0, 2.0]])
    values = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    adv, returns = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.full((1, 3), 2.0))
    assert torch.allclose(returns, adv)


def test_value_clip_blocks_gradient_outside_the_range():
    values = torch.tensor([5.0], requires_grad=True)
    old = torch.tensor([0.0])
    returns = torch.tensor([5.0])
    loss = value_loss(values, old, returns, epsilon=0.2, use_clip=True)
    loss.mean().backward()
    # Unclipped error is 0. Clipped error is large and flat in values, so the grad is 0.
    assert values.grad.item() == 0.0


def test_clip_buckets_do_not_invent_a_rate_for_an_empty_bin():
    old_prob = torch.tensor([0.99, 0.5])
    clipped = torch.tensor([False, True])
    mask = torch.tensor([True, True])
    stats = clip_fraction_by_bucket(old_prob, clipped, mask)
    assert stats["[0.00,0.01)"] is None
    assert stats["[0.90,1.01)"] == 0.0
    assert stats["all"] == pytest_approx(0.5)
    ratio = torch.tensor([1.5, 0.5])
    adv = torch.tensor([1.0, -1.0])
    mask_c = clipped_token_mask(ratio, adv, 0.2)
    assert mask_c.tolist() == [True, True]


def pytest_approx(value, abs=1e-5):
    import pytest
    return pytest.approx(value, abs=abs)
