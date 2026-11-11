"""Smoke steps on the tiny model. Losses must be finite. No download."""
import math

import torch

from posttrain.checkpoint import load_checkpoint, save_checkpoint
from posttrain.dpo import run_smoke as dpo_smoke
from posttrain.grpo import run_smoke as grpo_smoke
from posttrain.lora import apply_lora, trainable_parameters
from posttrain.monitor import TrainingMonitor
from posttrain.ppo import run_smoke as ppo_smoke
from posttrain.rft import run_smoke as rft_smoke
from posttrain.sft import run_smoke as sft_smoke
from posttrain.tiny import build_tiny


def _finite(rows):
    assert rows
    assert all(r["finite"] and math.isfinite(r["loss"]) for r in rows)


def test_sft_smoke_and_lora_freezes_the_base():
    rows = sft_smoke(steps=1, use_lora=True)
    _finite(rows)
    assert "grad_norm" in rows[0]
    model, _tok = build_tiny(seed=1)
    wrapped = apply_lora(model)
    names = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert names
    assert all("lora" in n for n in names)
    assert trainable_parameters(wrapped)


def test_dpo_smoke_logs_margin():
    rows = dpo_smoke(steps=2)
    _finite(rows)
    assert "pi_chosen" in rows[-1] and "pi_rejected" in rows[-1]
    assert "margin" in rows[-1]
    assert "likelihood_displacement" in rows[-1]


def test_ppo_smoke():
    rows = ppo_smoke(steps=1)
    _finite(rows)
    assert "value_loss" in rows[0]
    assert "approx_kl" in rows[0]
    assert "clip_frac_by_prob" in rows[0]
    assert "entropy" in rows[0]


def test_grpo_smoke_counts_prior_only():
    rows = grpo_smoke(steps=1)
    _finite(rows)
    assert any(r.get("prior_only") == 1.0 for r in rows)
    trained = [r for r in rows if r.get("prior_only") == 0.0]
    assert trained
    assert "zero_var_frac" in trained[0]
    assert "clip_frac" in trained[0]


def test_rft_smoke():
    _finite(rft_smoke(steps=1))


def test_checkpoint_restores_weights_optimizer_and_rng(tmp_path):
    torch.manual_seed(0)
    model, _tok = build_tiny(seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(input_ids=torch.randint(0, 50, (2, 8))).logits.mean()
    loss.backward()
    opt.step()
    saved = [p.detach().clone() for p in model.parameters()]
    torch.manual_seed(123)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, step=7, extra={"note": "smoke"})
    expected = torch.rand(4).clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    torch.manual_seed(999)
    fresh, _ = build_tiny(seed=4)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=0.01)
    payload = load_checkpoint(path, fresh, fresh_opt)
    assert payload["step"] == 7
    for a, b in zip(saved, fresh.parameters()):
        assert torch.allclose(a, b)
    assert torch.allclose(torch.rand(4), expected)
    # The optimizer state loads: a second step does not crash on a missing moment buffer.
    out = fresh(input_ids=torch.randint(0, 50, (2, 8))).logits.sum()
    out.backward()
    fresh_opt.step()


def test_grpo_forwards_sample_meta_into_the_reward():
    import copy

    from posttrain.grpo import train_steps
    from posttrain.logging_utils import StepLogger

    model, tok = build_tiny(seed=2)
    ref = copy.deepcopy(model)
    seen = {}

    def reward_fn(prompt, completion, meta):
        seen["info"] = meta.get("marker")
        return 0.3, {"reward_name": "marker"}

    train_steps(
        model, ref, tok, ["Incubate at"], reward_fn, steps=1, lr=1e-3, logger=StepLogger("grpo"),
        group_size=2, max_new_tokens=2, updates_per_rollout=1, metas=[{"marker": "kept"}],
    )
    assert seen["info"] == "kept"


def test_monitor_stops_when_proxy_rises_and_gold_falls():
    mon = TrainingMonitor(min_delta=0.05, min_steps=2)
    mon.update(step=0, proxy=0.2, gold=0.6, rollouts=[{"completion": "ok", "answer": "A", "reward": 0.2}])
    assert mon.stop is False
    mon.update(step=1, proxy=0.5, gold=0.3, rollouts=[{"completion": "ok", "answer": "A", "reward": 0.5}])
    assert mon.stop is True
    assert "gold" in mon.reason
