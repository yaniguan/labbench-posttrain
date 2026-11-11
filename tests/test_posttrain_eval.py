"""Evaluation report math. The demo rows are synthetic. Task metrics of a real model are not run."""
import math

from posttrain.eval_suite import (
    NOT_RUN,
    average_at_k,
    binomial_se,
    bootstrap_mean_ci,
    build_report,
    demo_toy_report,
    empty_report,
    expected_calibration_error,
    mcnemar,
    pass_at_k,
    paired_bootstrap,
)


def test_pass_at_k_and_average():
    # n=4, c=2, k=2 -> 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert pass_at_k(4, 2, 2) == pytest_approx(1 - 1 / 6)
    assert pass_at_k(4, 4, 2) == 1.0
    assert math.isnan(pass_at_k(4, 1, 5))
    assert average_at_k([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0]], 2) == pytest_approx((0.5 + 1.0) / 2)


def test_bootstrap_and_paired_and_mcnemar():
    ci = bootstrap_mean_ci([0.0, 1.0, 1.0, 0.0], n_boot=500, seed=0)
    assert ci["n"] == 4
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    wide = paired_bootstrap([1, 1, 1, 1, 0], [0, 0, 0, 0, 0], n_boot=500, seed=0)
    assert wide["excludes_zero"] is True
    tied = paired_bootstrap([1, 0, 1, 0], [1, 0, 1, 0], n_boot=200, seed=0)
    assert tied["mean_diff"] == 0.0
    table = mcnemar([1, 1, 0, 0], [1, 0, 1, 0])
    assert table["only_a"] == 1 and table["only_b"] == 1
    assert table["statistic"] == 0.0
    se = binomial_se(0.5, 30)
    # sqrt(0.25/30) ≈ 0.091. A 10-point gap on 30 items is about one SE.
    assert se == pytest_approx((0.25 / 30) ** 0.5)


def test_calibration_and_empty_report():
    perfect = expected_calibration_error([1, 1, 0, 0], [1, 1, 0, 0], n_bins=2)
    assert perfect["ece"] == pytest_approx(0.0)
    report = empty_report()
    assert report["status"] == NOT_RUN
    assert report["accuracy"] == NOT_RUN
    toy = demo_toy_report(seed=0)
    assert toy["accuracy"]["n"] == 40
    assert toy["reward_ladder"]["status"] == NOT_RUN
    assert toy["held_out_gold"]["used_for_training"] is False
    assert "sft" in toy["methods"]


def test_small_n_note_is_not_a_fake_score():
    built = build_report()
    assert built["note"].lower().find("not run") >= 0 or built["status"] == NOT_RUN


def pytest_approx(value, abs=1e-6):
    import pytest
    return pytest.approx(value, abs=abs)
