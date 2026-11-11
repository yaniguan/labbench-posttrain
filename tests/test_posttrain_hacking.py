"""Hacking detectors, reward holes, and the canary check."""
from __future__ import annotations

import pytest

from posttrain.hacking_checks import (
    CANARY_NEEDLES,
    adversarial_score,
    canary_in_files,
    canary_in_texts,
    consistency,
    constant_answer,
    metagaming,
    monitor_step,
    non_answering,
    position_format_bias,
    render_json,
    template_shift,
    trivial_baselines,
)
from posttrain.reward_baselines import (
    LADDER,
    format_only_reward,
    length_reward,
    majority_label,
    no_repetition_reward,
    score_baseline,
)


def test_each_detector_fires_on_a_synthetic_batch():
    empty = [{"completion": "", "reward": 0.0, "task": "pqa"} for _ in range(4)]
    assert non_answering(empty)["triggered"]
    constant = [{"completion": "x", "answer": "A", "reward": 0.2, "label": str(i)} for i in range(6)]
    assert constant_answer(constant)["triggered"]
    positioned = [{"completion": "a", "reward": 1.0, "option_position": 0}]
    positioned += [{"completion": "b", "reward": 0.0, "option_position": 1} for _ in range(4)]
    assert position_format_bias(positioned)["triggered"]
    meta = [{"completion": "I will game the reward function", "reward": 1.0}]
    assert metagaming(meta)["triggered"]
    assert template_shift(0.9, 0.4)["triggered"]
    assert consistency([False, False, False, False])["triggered"]
    assert canary_in_texts(["hello " + CANARY_NEEDLES[0]])["triggered"]
    assert not canary_in_texts(["heat the sample to 37 C"])["triggered"]


def test_monitor_stops_when_proxy_and_gold_diverge():
    rows = [{"completion": "ok", "reward": 1.0, "answer": "A", "task": "pqa"}]
    with pytest.raises(RuntimeError, match="gold"):
        monitor_step(0, 1, rows, proxy_mean=0.9, gold_mean=0.1, stop_on_flag=True)
    quiet = monitor_step(0, 1, rows, proxy_mean=0.4, gold_mean=0.4, stop_on_flag=False)
    assert quiet["proxy_gold_divergence"] is False
    assert "detectors" in render_json(quiet)


def test_reward_ladder_modes():
    assert "random_p50" in LADDER and "prior_only" in LADDER
    assert format_only_reward(r"final \boxed{37}") == 1.0
    assert format_only_reward(r"\boxed{}") == 0.0
    assert format_only_reward("[ANSWER_START]37[ANSWER_END]") == 0.0
    assert no_repetition_reward("ok " * 12) == 0.0
    assert no_repetition_reward("heat the sample to 37 C") == 1.0
    assert 0.0 < length_reward("one two") < 1.0
    group = [r"\boxed{A}", r"\boxed{A}", r"\boxed{B}"]
    voted = score_baseline("majority_vote", prompt="q", completion=group[2], group_completions=group, seed=0)
    assert voted.value == 0.0
    voted_ok = score_baseline("majority_vote", prompt="q", completion=group[0], group_completions=group, seed=0)
    assert voted_ok.value == 1.0
    wrong = score_baseline(
        "incorrect_label", prompt="q", completion="[ANSWER_START]B[ANSWER_END]",
        meta={"incorrect_label": "B"}, seed=0,
    )
    assert wrong.value == 1.0
    rng_a = score_baseline("random_p50", prompt="q", completion="c", seed=3, index=1).value
    rng_b = score_baseline("random_p50", prompt="q", completion="c", seed=3, index=1).value
    assert rng_a == rng_b
    assert score_baseline("constant_1", prompt="q", completion="c").value == 1.0
    assert score_baseline("constant_0", prompt="q", completion="c").value == 0.0
    assert majority_label(["A", None, "A", "B"]) == "A"


def test_trivial_baselines_and_adversarial_rewards_stay_low():
    for name in ("pqa", "ord", "err", "gen"):
        floors = trivial_baselines(name, n=40)
        assert floors["empty"] == 0.0
        assert floors["always_A"] < 0.75
        adv = adversarial_score(name, n=30)
        assert adv["empty"] == 0.0
        assert adv["copy_prompt"] < 0.2
    pqa = adversarial_score("pqa", n=20)
    assert pqa["all_choices_one_tag"] == 0.0
    assert pqa["boxed_only"] == 0.0
    assert pqa["not_a_choice"] == 0.0
    ord_ = adversarial_score("ord", n=20)
    assert ord_["comment_on_gold"] == 0.0
    assert ord_["not_a_permutation"] == 0.0
    assert ord_["words_around_list"] == 0.0
    # The last answer block is the one that counts, so a correct list followed
    # by the identity list does not collect the gold reward.
    assert ord_["hedge_wrong_last"] < 0.4
    err = adversarial_score("err", n=30)
    assert err["falsehood_plus_gold"] == 0.0
    assert err["gold_outside_tags"] == 0.0
    assert err["copy_prompt"] == 0.0
    gen = adversarial_score("gen", n=20)
    assert gen["one_line_dump"] == 0.0
    assert gen["boilerplate"] < 0.2


def test_gen_repeated_first_step_is_not_rewarded():
    import statistics as st
    from bioenvs import make_env
    env = make_env("gen", split="test", seed=1)
    vals = []
    for i in range(100):
        sample = env.sample(i)
        ref = sample.info["ref"]
        text = "\n".join(f"{j + 1}. {ref[0]}" for j in range(len(ref)))
        vals.append(env.score(sample, text).value)
    assert st.mean(vals) < 0.15


def test_canary_scanner_on_files(tmp_path):
    clean = tmp_path / "train.txt"
    clean.write_text("Heat the sample.\n", encoding="utf-8")
    dirty = tmp_path / "leak.txt"
    dirty.write_text("notes " + CANARY_NEEDLES[1] + "\n", encoding="utf-8")
    assert canary_in_files([str(clean)])["triggered"] is False
    assert canary_in_files([str(dirty)])["triggered"] is True


def test_shipped_training_configs_have_no_canary():
    from pathlib import Path
    root = Path("configs/posttrain")
    paths = [str(p) for p in root.rglob("*.yaml")]
    assert paths
    report = canary_in_files(paths)
    assert report["triggered"] is False
