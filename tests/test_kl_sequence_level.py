"""Exact multi-step categorical enumeration tests for Reverse-KL gradients (#54).

Verifies:
1. Per-token `reverse_kl_loss_term` (`k3`, matching `yaniguan/RL` `reference_policy_kl_type: k3`)
   matches the exact gradient on the final step (`max_abs_diff < 1e-12`), while dropping the
   state-transition KL coupling on earlier steps (`0.136729` relative L2 error on step-1 logits
   for the `torch.manual_seed(0)` 2-step vocab-3 policy, and non-zero gaps on steps 1 & 2 of a
   3-step branching policy; arXiv:2506.09477).
2. Both `sequence_reverse_kl_reward_to_go` (policy-gradient reward-to-go) and
   `sequence_reverse_kl_loss_term` / `policy_loss(..., kl_mode="sequence_exact")` match the
   exact sequence-level `grad_theta KL(Pi_theta(y|x) || Pi_ref(y|x))` within `1e-6` across
   every step on both 2-step (`3^2 = 9` sequences) and 3-step (`3^3 = 27` sequences) policies.
"""
from __future__ import annotations

import pytest
import torch

from posttrain.grpo import TOY_PROMPTS, train_grpo
from posttrain.kl import (
    reverse_kl_loss_term,
    sequence_reverse_kl_loss_term,
    sequence_reverse_kl_reward_to_go,
)
from posttrain.rewards import toy_protocol_reward
from posttrain.rl_common import policy_loss


def test_two_step_categorical_enumeration_records_token_gap_and_sequence_exact_match() -> None:
    """#54: Exact 2-step vocab-3 enumeration (`torch.manual_seed(0)`, clamps off)."""
    torch.manual_seed(0)
    l1_init = torch.randn(3, dtype=torch.float64)
    l2_init = torch.randn(3, 3, dtype=torch.float64)
    r1_init = torch.randn(3, dtype=torch.float64)
    r2_init = torch.randn(3, 3, dtype=torch.float64)

    rp1 = torch.log_softmax(r1_init, dim=-1)
    rp2 = torch.log_softmax(r2_init, dim=-1)

    # 1. Exact sequence-level KL(Pi_theta || Pi_ref) gradient over all 3 x 3 = 9 trajectories
    l1_exact = l1_init.clone().requires_grad_(True)
    l2_exact = l2_init.clone().requires_grad_(True)
    lp1_e = torch.log_softmax(l1_exact, dim=-1)
    lp2_e = torch.log_softmax(l2_exact, dim=-1)
    seq_lp = lp1_e[:, None] + lp2_e
    seq_rp = rp1[:, None] + rp2
    p_seq = seq_lp.exp()
    exact_kl = (p_seq * (seq_lp - seq_rp)).sum()
    exact_kl.backward()
    g1_exact = l1_exact.grad.clone()
    g2_exact = l2_exact.grad.clone()

    # Fixed measured values from Issue #54 (tolerance 1e-6)
    expected_g1_exact = torch.tensor(
        [0.4394522128700249, -0.3740751610752480, -0.06537705179477724],
        dtype=torch.float64,
    )
    torch.testing.assert_close(g1_exact, expected_g1_exact, rtol=1e-6, atol=1e-6)

    # 2. Per-token k3 surrogate (`sum_t reverse_kl_loss_term`, clamps off)
    l1_tok = l1_init.clone().requires_grad_(True)
    l2_tok = l2_init.clone().requires_grad_(True)
    lp1_t = torch.log_softmax(l1_tok, dim=-1)
    lp2_t = torch.log_softmax(l2_tok, dim=-1)
    p_det = (lp1_t[:, None] + lp2_t).detach().exp()
    term1 = reverse_kl_loss_term(
        lp1_t[:, None].expand(3, 3),
        rp1[:, None].expand(3, 3),
        estimator="k3",
        input_clamp_value=None,
        output_clamp_value=None,
    )
    term2 = reverse_kl_loss_term(
        lp2_t,
        rp2,
        estimator="k3",
        input_clamp_value=None,
        output_clamp_value=None,
    )
    tok_surr = (p_det * (term1 + term2)).sum()
    tok_surr.backward()

    # Step-2 (final step) is exact; Step-1 exhibits the 0.136729 (~0.137) sequential bias
    torch.testing.assert_close(l2_tok.grad, g2_exact, rtol=1e-12, atol=1e-12)
    expected_g1_surr = torch.tensor(
        [0.49262183051271413, -0.4327983204832442, -0.05982351002947006],
        dtype=torch.float64,
    )
    torch.testing.assert_close(l1_tok.grad, expected_g1_surr, rtol=1e-6, atol=1e-6)
    rel_l2_gap = float((l1_tok.grad - g1_exact).norm() / g1_exact.norm())
    assert rel_l2_gap == pytest.approx(0.1367293948317811, abs=1e-6)

    # 3. Sequence-level detached k1 reward-to-go (`sequence_reverse_kl_reward_to_go`) matches within 1e-6
    l1_rtg = l1_init.clone().requires_grad_(True)
    l2_rtg = l2_init.clone().requires_grad_(True)
    lp1_r = torch.log_softmax(l1_rtg, dim=-1)
    lp2_r = torch.log_softmax(l2_rtg, dim=-1)
    # Build [9, 2] trajectory logprob table across all (y1, y2)
    traj_lp = torch.stack([lp1_r[:, None].expand(3, 3).reshape(9), lp2_r.reshape(9)], dim=-1)
    traj_rp = torch.stack([rp1[:, None].expand(3, 3).reshape(9), rp2.reshape(9)], dim=-1)
    traj_mask = torch.ones_like(traj_lp)
    traj_prob_det = traj_lp.sum(dim=-1).detach().exp()

    kl_rtg = sequence_reverse_kl_reward_to_go(traj_lp, traj_rp, traj_mask)
    pg_kl_obj = (traj_prob_det * (traj_lp * kl_rtg).sum(dim=-1)).sum()
    pg_kl_obj.backward()
    torch.testing.assert_close(l1_rtg.grad, g1_exact, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(l2_rtg.grad, g2_exact, rtol=1e-6, atol=1e-6)

    # 4. `sequence_reverse_kl_loss_term` (both k3 and k1) matches exact sequence gradient within 1e-6
    for est in ("k3", "k1"):
        l1_seq = l1_init.clone().requires_grad_(True)
        l2_seq = l2_init.clone().requires_grad_(True)
        lp1_s = torch.log_softmax(l1_seq, dim=-1)
        lp2_s = torch.log_softmax(l2_seq, dim=-1)
        t_lp = torch.stack([lp1_s[:, None].expand(3, 3).reshape(9), lp2_s.reshape(9)], dim=-1)
        seq_term = sequence_reverse_kl_loss_term(
            t_lp,
            traj_rp,
            traj_mask,
            estimator=est,
            input_clamp_value=None,
            output_clamp_value=None,
        )
        # Forward value matches expected KL while backward carries exact sequence gradient
        assert float((traj_prob_det * seq_term.sum(dim=-1)).sum().detach()) == pytest.approx(
            float(exact_kl.detach()), abs=1e-10
        )
        (traj_prob_det * seq_term.sum(dim=-1)).sum().backward()
        torch.testing.assert_close(l1_seq.grad, g1_exact, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(l2_seq.grad, g2_exact, rtol=1e-6, atol=1e-6)


def test_three_step_categorical_enumeration_and_policy_loss_sequence_exact_mode() -> None:
    """#54: Exact 3-step (`3^3 = 27` trajectories) branching policy + `policy_loss(..., kl_mode='sequence_exact')`."""
    torch.manual_seed(1)
    l1_0 = torch.randn(3, dtype=torch.float64)
    l2_0 = torch.randn(3, 3, dtype=torch.float64)
    l3_0 = torch.randn(3, 3, 3, dtype=torch.float64)
    r1_0 = torch.randn(3, dtype=torch.float64)
    r2_0 = torch.randn(3, 3, dtype=torch.float64)
    r3_0 = torch.randn(3, 3, 3, dtype=torch.float64)

    rp1 = torch.log_softmax(r1_0, dim=-1)
    rp2 = torch.log_softmax(r2_0, dim=-1)
    rp3 = torch.log_softmax(r3_0, dim=-1)
    traj_rp = torch.stack(
        [
            rp1[:, None, None].expand(3, 3, 3).reshape(27),
            rp2[:, :, None].expand(3, 3, 3).reshape(27),
            rp3.reshape(27),
        ],
        dim=-1,
    )

    # Exact 3-step gradient
    l1_e = l1_0.clone().requires_grad_(True)
    l2_e = l2_0.clone().requires_grad_(True)
    l3_e = l3_0.clone().requires_grad_(True)
    lp1 = torch.log_softmax(l1_e, dim=-1)
    lp2 = torch.log_softmax(l2_e, dim=-1)
    lp3 = torch.log_softmax(l3_e, dim=-1)
    seq_lp = lp1[:, None, None] + lp2[:, :, None] + lp3
    seq_rp = rp1[:, None, None] + rp2[:, :, None] + rp3
    (seq_lp.exp() * (seq_lp - seq_rp)).sum().backward()
    g1_e, g2_e, g3_e = l1_e.grad.clone(), l2_e.grad.clone(), l3_e.grad.clone()

    # Per-token k3 surrogate: exact on step 3, biased on steps 1 and 2
    l1_t = l1_0.clone().requires_grad_(True)
    l2_t = l2_0.clone().requires_grad_(True)
    l3_t = l3_0.clone().requires_grad_(True)
    t_lp_tok = torch.stack(
        [
            torch.log_softmax(l1_t, dim=-1)[:, None, None].expand(3, 3, 3).reshape(27),
            torch.log_softmax(l2_t, dim=-1)[:, :, None].expand(3, 3, 3).reshape(27),
            torch.log_softmax(l3_t, dim=-1).reshape(27),
        ],
        dim=-1,
    )
    p_det = t_lp_tok.sum(dim=-1).detach().exp()
    tok_terms = reverse_kl_loss_term(
        t_lp_tok, traj_rp, estimator="k3", input_clamp_value=None, output_clamp_value=None
    )
    (p_det * tok_terms.sum(dim=-1)).sum().backward()
    torch.testing.assert_close(l3_t.grad, g3_e, rtol=1e-12, atol=1e-12)
    assert float((l2_t.grad - g2_e).norm()) > 1e-2
    assert float((l1_t.grad - g1_e).norm()) > 1e-2

    # `policy_loss` with `kl_mode="sequence_exact"` (zero task advantage, pure KL anchor)
    # weighted by `27 * p_det` per trajectory matches exact sequence gradient on ALL 3 steps!
    l1_s = l1_0.clone().requires_grad_(True)
    l2_s = l2_0.clone().requires_grad_(True)
    l3_s = l3_0.clone().requires_grad_(True)
    t_lp_seq = torch.stack(
        [
            torch.log_softmax(l1_s, dim=-1)[:, None, None].expand(3, 3, 3).reshape(27),
            torch.log_softmax(l2_s, dim=-1)[:, :, None].expand(3, 3, 3).reshape(27),
            torch.log_softmax(l3_s, dim=-1).reshape(27),
        ],
        dim=-1,
    )
    mask = torch.ones_like(t_lp_seq)
    zero_adv = torch.zeros_like(t_lp_seq)
    # Weight each trajectory i by 27 * p_det[i] via per-row policy_loss to compute exact expectation
    expected_loss = torch.zeros((), dtype=torch.float64)
    for idx in range(27):
        row_loss, _, _ = policy_loss(
            t_lp_seq[idx : idx + 1],
            t_lp_seq[idx : idx + 1].detach(),
            traj_rp[idx : idx + 1],
            zero_adv[idx : idx + 1],
            mask[idx : idx + 1],
            kl_coef=1.0,
            kl_estimator="k3",
            kl_in_loss=True,
            eps_low=0.2,
            eps_high=0.2,
            clip=False,
            cispo=False,
            token_level=True,
            kl_mode="sequence_exact",
        )
        expected_loss = expected_loss + p_det[idx] * row_loss
    expected_loss.backward()
    torch.testing.assert_close(l1_s.grad, g1_e, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(l2_s.grad, g2_e, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(l3_s.grad, g3_e, rtol=1e-6, atol=1e-6)

    # End-to-end trainer integration check for `kl_mode="sequence_exact"`
    res = train_grpo(
        TOY_PROMPTS,
        toy_protocol_reward,
        steps=2,
        group_size=2,
        max_new_tokens=4,
        kl_coef=0.02,
        kl_mode="sequence_exact",
        seed=0,
    )
    assert res["status"] == "ran"
    assert res["sft_anchor_unchanged"] is True
    assert all(torch.isfinite(torch.tensor(res["losses"])))
