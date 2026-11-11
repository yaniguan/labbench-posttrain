"""Contamination metrics, the eval report, KL configs, and a checkpoint round-trip."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from posttrain.contamination import (
    continuation_metrics,
    cpu_demo,
    interpret_holdout,
    letter_was_memorised,
    membership_gap,
    min_k_logprob,
    option_order,
    shuffled_control,
)
from posttrain.eval_suite import (
    average_at_k,
    binomial_se,
    bootstrap_ci,
    build_report,
    mcnemar,
    noise_note,
    pass_at_k,
    render_markdown,
)
from posttrain.kl_report import render_kl_report


def test_continuation_metrics_rank_a_copy_above_a_shuffle():
    gold = "heat the sample to 37 C and spin"
    perfect = continuation_metrics(gold, gold)
    assert perfect["exact_match"] == 1.0
    assert perfect["normalized_edit_distance"] == 0.0
    assert perfect["rouge_l"] == 1.0
    messy = continuation_metrics(shuffled_control(gold, seed=2), gold)
    assert messy["exact_match"] == 0.0
    assert messy["rouge_l"] < perfect["rouge_l"]


def test_option_order_and_letter_memory():
    shuffled = option_order(["red", "green", "blue"], "green", seed=1)
    assert shuffled["options"][ord(shuffled["new_letter"]) - ord("A")] == "green"
    if shuffled["letter_changed"]:
        assert letter_was_memorised(shuffled["old_letter"], shuffled["old_letter"], shuffled["new_letter"])
        assert not letter_was_memorised(shuffled["new_letter"], shuffled["old_letter"], shuffled["new_letter"])


def test_min_k_and_holdout_interpreter():
    scores = [0.0, -1.0, -5.0, -0.2]
    worst = min_k_logprob(scores, percent=0.5)
    assert worst == pytest.approx((-5.0 + -1.0) / 2)
    assert membership_gap(-0.2, -1.0) == pytest.approx(0.8)
    missing = interpret_holdout("2024", "AIME-2025", "2025", None, None)
    assert missing["status"] == "not run"
    supplied = interpret_holdout("2024", "AIME-2025", "2025", 0.4, 0.2)
    assert supplied["status"] == "scores supplied by caller"
    assert "after the model" in supplied["reading"]


def test_cpu_demo_runs_on_the_tiny_model():
    demo = cpu_demo()
    assert demo["status"] == "ran on tiny model"
    assert demo["original"][0]["fraction"] == 0.5
    for key in ("exact_match", "normalized_edit_distance", "rouge_l", "token_overlap"):
        value = demo["original"][0][key]
        assert 0.0 <= value <= 1.0
    assert demo["canary"]["triggered"] is False
    assert demo["mean_logprob_original"] is not None


def test_eval_report_toy_numbers_are_labeled():
    rows = [{"correct": i % 2 == 0, "task": "pqa" if i < 4 else "ord", "length": 3} for i in range(8)]
    methods = {
        "sft": [True, False, True, False],
        "grpo": [True, True, False, False],
    }
    report = build_report(rows=rows, methods=methods, status="toy demo, not a model result", heldout_gold=[True, False])
    text = render_markdown(report)
    assert "toy demo" in text
    assert report["accuracy"]["n"] == 8
    assert "per_task" in report
    assert report["mcnemar"]["status"] == "computed"
    assert "30" in noise_note(30)
    empty = build_report()
    assert empty["status"] == "not run"
    assert "not run" in render_markdown(empty)
    assert pass_at_k(1, 4, 1) == pytest.approx(0.25)
    assert pass_at_k(0, 4, 1) == 0.0
    assert average_at_k([True, False, True, True], 2) == pytest.approx(0.5)
    assert binomial_se(0.5, 30) == pytest.approx((0.25 / 30) ** 0.5)
    ci = bootstrap_ci([1.0, 0.0, 1.0, 0.0], n_boot=200, seed=0)
    assert ci["low"] <= ci["mean"] <= ci["high"]
    paired = mcnemar([True, True, False, False], [True, False, True, False])
    assert paired["n10"] == 1 and paired["n01"] == 1


def test_kl_report_without_rows_says_not_run():
    text = render_kl_report(None)
    assert "not run" in text
    assert "knee" in text
    rows = [
        {"kl_coef": 0.1, "proxy_reward": 0.4, "gold_metric": 0.4, "kl_to_sft": 0.01, "hacking_flags": []},
        {"kl_coef": 0.0, "proxy_reward": 0.8, "gold_metric": 0.1, "kl_to_sft": 0.5, "hacking_flags": ["metagaming"]},
    ]
    filled = render_kl_report(rows)
    assert "drift pattern" in filled
    assert "metagaming" in filled


def test_plot_script_writes_a_png(tmp_path):
    import importlib.util
    log = tmp_path / "log.jsonl"
    log.write_text(
        '{"kl_to_sft": 0.01, "reward_mean": 0.2, "gold_metric": 0.2}\n'
        '{"kl_to_sft": 0.2, "reward_mean": 0.5, "gold_metric": 0.1}\n',
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "plot_reward_vs_kl", Path("scripts/posttrain/plot_reward_vs_kl.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "curve.png"
    module.plot(module.load_rows(log), out)
    assert out.exists() and out.stat().st_size > 0
    assert module.main(["--log", str(tmp_path / "missing.json")]) == 0


def _walk_kl(node, found, ablation, recommended, path):
    if isinstance(node, list):
        for item in node:
            _walk_kl(item, found, ablation, recommended, path)
        return
    if not isinstance(node, dict):
        return
    if "ablation" in node:
        ablation = bool(node["ablation"])
    if "recommended" in node:
        recommended = bool(node["recommended"])
    method = node.get("method")
    coef = None
    if "kl_coef" in node and method in {"dpo", "ppo", "grpo"}:
        coef = node["kl_coef"]
    elif "reference_policy_kl_penalty" in node:
        coef = node["reference_policy_kl_penalty"]
        method = method or "nemo"
    if coef is not None:
        found.append((path, method, float(coef), ablation, recommended))
    for value in node.values():
        _walk_kl(value, found, ablation, recommended, path)


def test_every_shipped_rl_config_keeps_kl_on_unless_it_is_an_ablation():
    root = Path("configs/posttrain")
    found = []
    for path in root.rglob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        _walk_kl(cfg, found, False, True, path)
    assert found, "no KL settings found"
    for path, method, coef, ablation, recommended in found:
        if coef == 0.0:
            assert "kl_grid" in str(path), path
            assert ablation is True, path
            assert recommended is False, path
        else:
            assert coef > 0, path
            assert method


def test_checkpoint_restores_rng(tmp_path):
    from posttrain.checkpoint import load_checkpoint, save_checkpoint
    from posttrain.tiny import build_tiny_lm
    model, _ = build_tiny_lm(seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    torch.manual_seed(123)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, step=3, extra={"method": "sft"})
    torch.manual_seed(999)
    torch.randn(2)
    load_checkpoint(path, model, opt)
    restored = torch.randn(2)
    torch.manual_seed(123)
    expected = torch.randn(2)
    assert torch.equal(restored, expected)


def test_lora_reduces_trainable_parameters():
    from posttrain.batching import maybe_lora, trainable_count
    from posttrain.tiny import build_tiny_lm
    model, _ = build_tiny_lm(seed=0)
    full, total = trainable_count(model)
    assert full == total
    lora = maybe_lora(model, True)
    trainable, total_lora = trainable_count(lora)
    assert trainable < total_lora
