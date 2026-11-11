"""Contamination metrics, the toy CPU demo, and the LAB-Bench tripwire."""
import math

import pytest

from posttrain.contamination import (
    assert_not_labbench_training_path,
    assert_training_corpus_clean,
    canary_in_outputs,
    continuation_metrics,
    load_labbench_questions,
    membership_gap,
    option_order_effect,
    perturb_text,
    shuffle_text,
    toy_demo,
)
from posttrain.toy import SFT_PAIRS


def test_continuation_metrics_on_a_copy_and_a_shuffle():
    gold = "mix the buffer then warm it"
    perfect = continuation_metrics(gold, gold)
    assert perfect["exact"] == 1.0
    assert perfect["normalized_edit"] == 0.0
    assert perfect["rouge_l"] == 1.0
    shuffled = continuation_metrics(shuffle_text(gold, seed=0), gold)
    assert shuffled["exact"] == 0.0
    assert shuffled["rouge_l"] < perfect["rouge_l"]
    empty = continuation_metrics("", gold)
    assert empty["exact"] == 0.0
    assert empty["token_overlap"] == 0.0


def test_membership_gap_and_option_order_are_finite():
    gap = membership_gap([-0.1, -0.2, -3.0], [-2.0, -2.5, -4.0])
    assert gap["gap"] > 0  # original is less negative, the weak membership direction
    effect = option_order_effect([
        {"correct_original": True, "correct_shuffled": False, "letter_same": True, "content_same": False},
        {"correct_original": True, "correct_shuffled": True, "letter_same": False, "content_same": True},
    ])
    assert effect["drop"] == pytest.approx(0.5)
    assert effect["letter_stuck_rate"] == pytest.approx(0.5)
    assert perturb_text("one two three four five", seed=1) != "one two three four five"


def test_toy_demo_runs_on_cpu():
    report = toy_demo(seed=0)
    assert report["status"].startswith("toy")
    assert math.isfinite(report["overfit_loss"])
    assert report["canary_in_toy_text"] is False
    assert "fitted" in report and "unfitted" in report
    # The fitted model saw the sentence. Its original-prefix score should beat the cold model.
    fitted = report["fitted"]["original"]["rouge_l"]
    cold = report["unfitted"]["original"]["rouge_l"]
    assert fitted > cold


def test_labbench_is_eval_only_and_training_text_is_clean():
    rows = load_labbench_questions(limit=3)
    assert rows
    assert all(r["has_canary_in_source_line"] for r in rows)
    with pytest.raises(RuntimeError):
        assert_not_labbench_training_path("data/labbench/protocolqa-v1-public.jsonl")
    # Toy smoke strings are not the benchmark.
    assert_training_corpus_clean([a for pair in SFT_PAIRS for a in pair])
    assert canary_in_outputs(["a protocol step"])["hit"] is False
