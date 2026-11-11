"""Reward ladder and the config grid. No download, no training run."""
from collections import Counter

from posttrain.configs import load_grid, load_yaml
from posttrain.contamination import assert_training_corpus_clean
from posttrain.gold_pairs import gold_completion, gold_dpo_pairs, gold_sft_pairs
from posttrain.monitor import holdout_ids
from posttrain.reward_baselines import NAMES, compute_baseline, has_obvious_repetition
from posttrain.rewards import get_env, reward


def test_grid_covers_families_and_does_not_download():
    runs = load_grid()
    families = {r["family"] for r in runs}
    rewards = {r["reward_name"] for r in runs}
    assert families == {"qwen", "llama", "olmo"}
    assert set(NAMES) <= rewards
    assert all(r["download_in_ci"] is False and r["status"] == "not run" for r in runs)
    assert len(runs) == 3 * len(NAMES) * 4
    holdout = load_yaml("configs/posttrain/aime_holdout.yaml")
    assert holdout["status"] == "not run"
    assert holdout["questions"] == []


def test_random_is_seeded_and_near_half():
    meta = {"seed": 3, "env": "pqa", "info": {}}
    a, _ = compute_baseline("random_p50", "prompt", "answer", meta)
    b, _ = compute_baseline("random_p50", "prompt", "answer", meta)
    assert a == b
    bits = [compute_baseline("random_p50", f"p{i}", f"c{i}", {"seed": 1})[0] for i in range(400)]
    rate = sum(bits) / len(bits)
    assert 0.4 < rate < 0.6


def test_constants_length_and_repetition():
    assert compute_baseline("constant_one", "p", "c", {})[0] == 1.0
    assert compute_baseline("constant_zero", "p", "c", {})[0] == 0.0
    short, _ = compute_baseline("length_reward", "p", "one", {"length_cap": 10})
    longer, _ = compute_baseline("length_reward", "p", "one two three four five", {"length_cap": 10})
    assert longer > short
    assert has_obvious_repetition("washwell " * 12)
    assert compute_baseline("no_repetition", "p", "wash the cells once", {})[0] == 1.0
    assert compute_baseline("no_repetition", "p", "washwell " * 12, {})[0] == 0.0


def test_format_majority_and_incorrect_on_pqa():
    env = get_env("pqa", "test", 1)
    sample = env.sample(0)
    meta = {"env": "pqa", "info": sample.info, "split": "test", "seed": 1}
    gold = gold_completion("pqa", sample)
    assert reward(sample.prompt, gold, meta)[0] > 0.8
    assert compute_baseline("format_only", sample.prompt, "", meta)[0] == 0.0
    assert compute_baseline("format_only", sample.prompt, gold, meta)[0] == 1.0
    wrong = next(c for c in sample.info["choices"] if c != sample.info["answer"])
    rollouts = [f"[ANSWER_START]{wrong} & 10[ANSWER_END]"] * 5
    maj_meta = {**meta, "rollouts": rollouts}
    hit, diag = compute_baseline("majority_vote", sample.prompt, f"[ANSWER_START]{wrong} & 20[ANSWER_END]", maj_meta)
    miss, _ = compute_baseline("majority_vote", sample.prompt, gold, maj_meta)
    assert hit == 1.0 and miss == 0.0 and diag["label"] == wrong
    none, diag0 = compute_baseline("majority_vote", sample.prompt, gold, meta)
    assert none == 0.0 and diag0["status"] == "no_rollouts"
    _probe, bad_diag = compute_baseline("incorrect_label", sample.prompt, "", meta)
    label = bad_diag["label"]
    assert label and label != sample.info["answer"]
    bad, _ = compute_baseline("incorrect_label", sample.prompt, f"[ANSWER_START]{label} & 20[ANSWER_END]", meta)
    good, _ = compute_baseline("incorrect_label", sample.prompt, gold, meta)
    assert bad == 1.0 and good == 0.0


def test_boxed_format_is_the_paper_check():
    meta = {"env": "pqa", "info": {"choices": ["a"], "answer": "a", "id": "x"}, "format_style": "boxed"}
    assert compute_baseline("format_only", "q", r"the answer is \boxed{4}", meta)[0] == 1.0
    assert compute_baseline("format_only", "q", r"\boxed{}", meta)[0] == 0.0


def test_gold_pairs_beat_rejected_and_respect_holdout():
    pairs = gold_dpo_pairs("ord", n=8, split="test", seed=1)
    env = get_env("ord", "test", 1)
    chosen_scores = []
    rejected_scores = []
    for row in pairs:
        sample_info = row["info"]
        from bioenvs.base import Sample
        sample = Sample(row["id"], row["prompt"], sample_info)
        chosen_scores.append(env.score(sample, row["chosen"]).value)
        rejected_scores.append(env.score(sample, row["rejected"]).value)
    assert sum(chosen_scores) / len(chosen_scores) == 1.0
    assert sum(rejected_scores) / len(rejected_scores) < 0.5
    sft = gold_sft_pairs("pqa", n=5, split="test", seed=1)
    assert_training_corpus_clean([p for pair in sft for p in pair])
    held = holdout_ids([row["id"] for row in gold_dpo_pairs("pqa", n=10, split="test", seed=0)], frac=0.3, seed=0)
    kept = gold_sft_pairs("pqa", n=10, split="test", seed=0, holdout_ids=held)
    # The ids are not stored on sft pairs. Check the prompts of held-out samples are absent.
    env = get_env("pqa", "test", 0)
    held_prompts = set()
    for i in range(len(env)):
        s = env.sample(i)
        if s.id in held:
            held_prompts.add(s.prompt)
    assert held_prompts
    assert not any(prompt in held_prompts for prompt, _completion in kept)


def test_type_filter_keeps_only_the_named_types():
    from posttrain.gold_pairs import filter_by_type, iter_samples

    samples = iter_samples("gen", n=200, split="test", seed=1, holdout_ids=None)
    kept = filter_by_type(samples, ["easy"])
    assert kept
    assert all(s.info["type"] == "easy" for s in kept)
    assert len(kept) < len(samples)


def test_reward_dispatch_names_ground_truth():
    env = get_env("err", "test", 0)
    sample = env.sample(0)
    value, diag = reward(sample.prompt, "", {"env": "err", "info": sample.info, "reward_name": "ground_truth"})
    assert value == 0.0
    assert diag["reward_name"] == "ground_truth"
    counts = Counter(NAMES)
    assert counts["prior_only"] == 1
