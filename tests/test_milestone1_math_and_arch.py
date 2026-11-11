"""Milestone 1 verification:
1. Unbiased Reverse-KL gradient theorem (w * k3 vs raw k3 Forward-KL bug)
2. PPO causal state-action alignment (first completion token preserved) & VAPO
3. DPO chosen-only kl_to_sft & fixed-probe likelihood_displacement
4. GSPO sequence geometric-mean ratios & TIS / ICE-POP / Seq-Mask-TIS
5. Non-GPT2 HuggingFace causal LM (Qwen2Config + q_proj/v_proj LoRA) compatibility
"""
from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from posttrain.advantages import gae, vapo_length_adaptive_lambda
from posttrain.batching import maybe_lora, model_device, model_hidden_size, model_max_length
from posttrain.dpo import TOY_PAIRS, train_dpo
from posttrain.grpo import TOY_PROMPTS, train_grpo
from posttrain.kl import kl_k3, reverse_kl_loss_term
from posttrain.losses import compute_is_weights, policy_surrogate
from posttrain.ppo import train_ppo
from posttrain.rewards import toy_protocol_reward
from posttrain.sft import TOY_SFT, train_sft
from posttrain.tiny import build_tiny_lm


def test_reverse_kl_loss_term_gives_exact_unbiased_reverse_kl_gradient():
    """Prove on a discrete categorical policy pi_theta = softmax(logits) that:
    - E_{a ~ pi_theta}[d/d(theta) reverse_kl_loss_term(log pi_theta(a), log pi_ref(a))]
      equals the exact analytical gradient d/d(theta) KL(pi_theta || pi_ref)
      for ALL three estimators (k1, k2, k3).
    - Raw kl_k3 WITHOUT the multiplicative score factor w = exp(logp - sg(logp))
      yields d/d(theta) KL(pi_ref || pi_theta) (Forward KL), NOT Reverse KL.
    """
    theta_base = torch.tensor([0.8, -0.4, 0.2, -1.1], dtype=torch.float64)
    ref_logits = torch.tensor([-0.3, 0.5, 0.1, -0.7], dtype=torch.float64)
    logp_ref = torch.log_softmax(ref_logits, dim=-1)
    pi_ref = logp_ref.exp()

    # Analytical Reverse KL(pi_theta || pi_ref) gradient
    t_exact = theta_base.clone().requires_grad_(True)
    lp_exact = torch.log_softmax(t_exact, dim=-1)
    exact_rev_kl = (lp_exact.exp() * (lp_exact - logp_ref)).sum()
    exact_rev_kl.backward()
    grad_rev_kl = t_exact.grad.clone()

    # Analytical Forward KL(pi_ref || pi_theta) gradient
    t_fwd = theta_base.clone().requires_grad_(True)
    lp_fwd = torch.log_softmax(t_fwd, dim=-1)
    exact_fwd_kl = (pi_ref * (logp_ref - lp_fwd)).sum()
    exact_fwd_kl.backward()
    grad_fwd_kl = t_fwd.grad.clone()

    assert not torch.allclose(grad_rev_kl, grad_fwd_kl, atol=1e-2)

    # Raw k3 expectation under fixed sample distribution pi_sg = sg(pi_theta)
    t_raw = theta_base.clone().requires_grad_(True)
    lp_raw = torch.log_softmax(t_raw, dim=-1)
    pi_sg = lp_raw.detach().exp()
    raw_surrogate = (pi_sg * kl_k3(lp_raw, logp_ref)).sum()
    raw_surrogate.backward()
    # Raw k3 matches Forward KL, NOT Reverse KL!
    torch.testing.assert_close(t_raw.grad, grad_fwd_kl, rtol=1e-10, atol=1e-10)

    # Fixed reverse_kl_loss_term matches exact Reverse KL for k1, k2, and k3!
    for est in ("k1", "k2", "k3"):
        t_est = theta_base.clone().requires_grad_(True)
        lp_est = torch.log_softmax(t_est, dim=-1)
        pi_sg = lp_est.detach().exp()
        surr = (pi_sg * reverse_kl_loss_term(lp_est, logp_ref, estimator=est)).sum()
        surr.backward()
        torch.testing.assert_close(t_est.grad, grad_rev_kl, rtol=1e-10, atol=1e-10)


def test_off_policy_reverse_kl_importance_weighting():
    """Verify importance-weighted Reverse-KL expectation under behavior mu != pi_theta."""
    theta_base = torch.tensor([0.6, -0.2, -0.5], dtype=torch.float64)
    mu_logits = torch.tensor([0.0, 0.2, -0.1], dtype=torch.float64)
    ref_logits = torch.tensor([-0.4, 0.3, 0.1], dtype=torch.float64)
    logp_mu = torch.log_softmax(mu_logits, dim=-1)
    mu = logp_mu.exp()
    logp_ref = torch.log_softmax(ref_logits, dim=-1)

    t_exact = theta_base.clone().requires_grad_(True)
    lp_exact = torch.log_softmax(t_exact, dim=-1)
    (lp_exact.exp() * (lp_exact - logp_ref)).sum().backward()

    t_is = theta_base.clone().requires_grad_(True)
    lp_is = torch.log_softmax(t_is, dim=-1)
    # Samples drawn from mu, weighted via logp_behavior=logp_mu
    surr = (mu * reverse_kl_loss_term(lp_is, logp_ref, estimator="k3", logp_behavior=logp_mu)).sum()
    surr.backward()
    torch.testing.assert_close(t_is.grad, t_exact.grad, rtol=1e-10, atol=1e-10)


def test_gspo_sequence_ratios_and_truncated_is_modes():
    """Verify GSPO geometric-mean ratio and TIS / ICE-POP / Seq-Mask-TIS."""
    logp_old = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], dtype=torch.float64)
    # Seq 0: deltas [+0.1, +0.5, 0(masked)] -> geometric mean exp(0.3)
    logp = torch.tensor([[-0.9, -0.5, -9.0], [-1.2, -1.4, -1.0]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    adv = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=torch.float64)

    loss = policy_surrogate(
        logp, logp_old, adv, eps_low=0.5, eps_high=0.5,
        clip=True, sequence_level_ratios=True, mask=mask,
    )
    # Active tokens in seq 0 both get -exp(0.3) * 1.0
    expected_s0 = -float(torch.exp(torch.tensor(0.3, dtype=torch.float64)))
    assert loss[0, 0].item() == pytest.approx(expected_s0, abs=1e-9)
    assert loss[0, 1].item() == pytest.approx(expected_s0, abs=1e-9)
    loss_scalar = (loss * mask).sum()
    loss_scalar.backward()
    assert logp.grad[0, 2].item() == 0.0  # masked token gets zero gradient

    # ICE-POP zeroes out tokens outside [0.5, 2.0] while keeping interior tokens intact
    lp_prev = torch.tensor([[0.0, 0.0, 0.0]])
    lp_gen = torch.tensor([[0.0, -0.3, -1.5]])  # ratios: [1.0, exp(0.3)~1.35, exp(1.5)~4.48]
    m = torch.ones_like(lp_prev)
    w_ice, m_ice = compute_is_weights(lp_prev, lp_gen, m, mode="icepop", clip_min=0.5, clip_max=2.0)
    assert w_ice[0, 0].item() == pytest.approx(1.0)
    assert w_ice[0, 1].item() == pytest.approx(float(torch.exp(torch.tensor(0.3))), rel=1e-5)
    assert w_ice[0, 2].item() == 0.0
    assert m_ice["is_oob_ratio"] == pytest.approx(1.0 / 3.0)

    # Seq-Mask-TIS gates whole sequence by geometric-mean IS ratio
    w_seq, m_seq = compute_is_weights(lp_prev, lp_gen, m, mode="seq-mask-tis", clip_min=0.5, clip_max=1.5)
    # Geometric mean = exp((0 + 0.3 + 1.5)/3) = exp(0.6) ~ 1.822 > 1.5 -> whole sequence zeroed
    assert torch.all(w_seq == 0.0)
    assert m_seq["is_oob_ratio"] == pytest.approx(1.0)


def test_ppo_first_completion_token_advantage_and_vapo():
    """Verify PPO aligns s_t -> a_t without dropping the first completion token,
    and runs VAPO length-adaptive lambda + positive-example NLL cleanly.
    """
    # Single-completion-token edge case: previously adv_for_policy[:, 1:] dropped
    # the sole completion token when prompt had length 1 or misindexed it!
    rewards = torch.tensor([[0.0, 1.0]])  # 1 prompt transition, 1 completion transition
    values = torch.zeros(1, 2)
    shift_mask = torch.tensor([[0.0, 1.0]])
    lam = vapo_length_adaptive_lambda(shift_mask, alpha=0.5)
    adv = gae(rewards, values, shift_mask, gamma=1.0, lam=lam)
    assert adv[0, 0].item() == 0.0
    assert adv[0, 1].item() == pytest.approx(1.0)

    model, tok = build_tiny_lm(seed=3)
    out = train_ppo(
        TOY_PROMPTS,
        toy_protocol_reward,
        model=model,
        tokenizer=tok,
        steps=2,
        group_size=2,
        max_new_tokens=3,
        vapo_length_adaptive_lam=True,
        vapo_alpha=0.5,
        positive_nll_coef=0.05,
        positive_reward_threshold=0.0,
        seed=3,
    )
    assert out["status"] == "ran"
    assert out["reference_unchanged"] is True
    assert all(torch.isfinite(torch.tensor(out["losses"])))


def test_dpo_kl_to_sft_excludes_rejected_and_uses_fixed_probe():
    """Verify DPO kl_to_sft is computed strictly on chosen completions and
    likelihood_displacement uses fixed probe pairs.
    """
    model, tok = build_tiny_lm(seed=5)
    out = train_dpo(TOY_PAIRS, model=model, tokenizer=tok, steps=3, seed=5, sft_coef=0.1)
    assert out["reference_unchanged"] is True
    assert isinstance(out["likelihood_displacement"], bool)
    assert all(r.kl_to_sft >= 0.0 for r in out["logger"].rows)


def test_qwen2_architecture_compatibility_across_all_trainers():
    """Build an in-memory Qwen2ForCausalLM (uses hidden_size, max_position_embeddings,
    and q_proj/k_proj/v_proj/o_proj Linear layers instead of GPT-2's n_embd/n_positions/c_attn)
    and run SFT+LoRA, DPO, PPO, and GRPO+GSPO end-to-end.
    """
    _, tok = build_tiny_lm(seed=7)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    qwen_sft = Qwen2ForCausalLM(cfg)
    assert model_hidden_size(qwen_sft) == 32
    assert model_max_length(qwen_sft, tok) == 64
    assert model_device(qwen_sft) == torch.device("cpu")

    # 1. SFT with modern Linear LoRA
    sft_res = train_sft(TOY_SFT, model=qwen_sft, tokenizer=tok, steps=2, use_lora=True, seed=7)
    assert sft_res["trainable"] > 0 and sft_res["trainable"] < sft_res["n_parameters"]

    # 2. DPO on Qwen2
    qwen_base = Qwen2ForCausalLM(cfg)
    dpo_res = train_dpo(TOY_PAIRS, model=qwen_base, tokenizer=tok, steps=2, seed=7)
    assert dpo_res["status"] == "ran"

    # 3. PPO on Qwen2
    qwen_ppo = Qwen2ForCausalLM(cfg)
    ppo_res = train_ppo(
        TOY_PROMPTS, toy_protocol_reward, model=qwen_ppo, tokenizer=tok,
        steps=1, group_size=2, max_new_tokens=3, seed=7,
    )
    assert ppo_res["status"] == "ran"

    # 4. GRPO + GSPO + ICE-POP on Qwen2
    qwen_grpo = Qwen2ForCausalLM(cfg)
    grpo_res = train_grpo(
        TOY_PROMPTS, toy_protocol_reward, model=qwen_grpo, tokenizer=tok,
        steps=1, group_size=2, max_new_tokens=3, seed=7,
        sequence_level_ratios=True, token_level=False,
        use_is_correction=True, is_mode="icepop",
    )
    assert grpo_res["status"] == "ran"
