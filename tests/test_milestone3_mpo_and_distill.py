"""Unit tests for Milestone 3: MPO (DPO + BCO EMA shift + SFT), TROPD distillation, and completion-masked KD."""
from __future__ import annotations

import pytest
import torch

from posttrain.distill_opd import train_tropd, tropd_token_advantages
from posttrain.grpo import TOY_PROMPTS
from posttrain.mpo import TOY_MPO_BATCH, mpo_loss, train_mpo, update_reward_shift
from posttrain.rewards import toy_protocol_reward
from scientist.distill import distill_step_loss
from scripts.posttrain.build_mpo_dataset import build_mpo_records


def test_mpo_ema_reward_shift_and_trainer():
    s1 = update_reward_shift(0.0, reward_sum=4.0, reward_count=2.0, momentum=0.90)
    assert s1 == pytest.approx(0.20)
    # Empty batch keeps shift unchanged
    assert update_reward_shift(s1, reward_sum=0.0, reward_count=0.0, momentum=0.90) == pytest.approx(0.20)

    logp_c = torch.tensor([-1.2, -1.5], requires_grad=True)
    logp_r = torch.tensor([-2.4, -2.1], requires_grad=True)
    ref_c = torch.tensor([-1.4, -1.6])
    ref_r = torch.tensor([-2.2, -2.0])
    logp_up = torch.tensor([-1.1], requires_grad=True)
    ref_up = torch.tensor([-1.3])
    logp_un = torch.tensor([-2.5], requires_grad=True)
    ref_un = torch.tensor([-2.1])

    loss, stats = mpo_loss(
        logp_chosen=logp_c,
        logp_rejected=logp_r,
        ref_chosen=ref_c,
        ref_rejected=ref_r,
        logp_unpaired_pos=logp_up,
        ref_unpaired_pos=ref_up,
        logp_unpaired_neg=logp_un,
        ref_unpaired_neg=ref_un,
        beta=0.1,
        reward_shift=0.05,
        preference_weight=1.0,
        bco_weight=0.8,
        sft_weight=0.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logp_c.grad is not None and logp_up.grad is not None and logp_un.grad is not None
    assert stats["bco_reward_count"] == 6.0

    out = train_mpo(TOY_MPO_BATCH, steps=2, seed=0)
    assert out["status"] == "ran"
    assert out["reference_unchanged"] is True

    mined = build_mpo_records(max_paired=5, max_unpaired=5)
    assert mined["summary"]["n_paired"] == 5
    assert mined["summary"]["n_unpaired"] == 5
    assert len(mined["records"]) == 10


def test_tropd_bounded_advantages_and_smoke():
    student_lp = torch.tensor([[-8.0, -0.5, -3.0]])
    teacher_lp = torch.tensor([[-0.2, -2.5, -1.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    adv, metrics = tropd_token_advantages(student_lp, [teacher_lp], mask, alpha=0.5, subtract_baseline=False)
    # Without baseline subtraction, log(0.5*pi_T + 0.5*pi_S) - log(pi_S) >= log(0.5) ~ -0.69315 everywhere!
    assert float(adv.min()) >= -0.69315 - 1e-5
    assert torch.isfinite(adv).all()
    assert "tropd_gap_mean" in metrics

    res = train_tropd(TOY_PROMPTS, toy_protocol_reward, proximal_alpha=0.5, steps=2, group_size=2, max_new_tokens=4, seed=0)
    assert res["status"] == "ran"
    assert res["reference_unchanged"] is True


def test_distill_completion_mask_and_vocab_guard():
    # Prompt positions (mask=0) must receive zero gradient
    s_logits = torch.randn(1, 5, 32, requires_grad=True)
    t_logits = torch.randn(1, 5, 32)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    comp_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]])  # only positions predicting idx 3 and 4 active

    loss, stats = distill_step_loss(s_logits, t_logits, ids, comp_mask, kl_type="mixed")
    loss.backward()
    assert stats["n_tokens"] == 2.0
    # s_logits[:, 0, :] and s_logits[:, 1, :] predict ids[:, 1] and ids[:, 2] which have comp_mask == 0!
    assert torch.allclose(s_logits.grad[:, :2, :], torch.zeros_like(s_logits.grad[:, :2, :]))
    assert float(s_logits.grad[:, 2:4, :].abs().sum()) > 0.0

    # Mismatched vocabulary without projection raises ValueError instead of silently slicing logits
    bad_teacher = torch.randn(1, 5, 1200)
    with pytest.raises(ValueError, match="Incompatible vocab sizes"):
        distill_step_loss(s_logits.detach(), bad_teacher, ids, comp_mask, same_tokenizer_family=False)
