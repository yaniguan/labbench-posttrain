"""One evaluation report for a checkpoint.

Sections that were not actually computed are the string "not run". This file does not
invent accuracies. Pass the records you have; missing pieces stay missing.

Small test sets are noisy. For a proportion p on n items the usual standard error is
sqrt(p * (1 - p) / n). At p = 0.5 and n = 30 (the size of AIME) that is about 0.091,
so a 10 point gap is roughly one standard error. Shao et al. say differences under
about 2 points on AIME may not be significant (Figure 13 caption). Report the interval,
not just the point.
"""
from __future__ import annotations

import math
import random

import numpy as np

NOT_RUN = "not run"


def bootstrap_mean_ci(values, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    if arr.size == 1:
        return {"n": 1, "mean": float(arr[0]), "lo": float(arr[0]), "hi": float(arr[0])}
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"n": int(arr.size), "mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "seed": seed, "n_boot": n_boot}


def binomial_se(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    p = min(1.0, max(0.0, p))
    return math.sqrt(p * (1.0 - p) / n)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al.). n samples, c correct, look at k of them.

    1 - C(n-c, k) / C(n, k) when n-c >= k, else 1.
    """
    if k <= 0 or n <= 0 or k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def average_at_k(rewards_per_prompt: list[list[float]], k: int) -> float:
    """Mean reward using up to k samples per prompt. Extra samples are ignored."""
    if k <= 0 or not rewards_per_prompt:
        return float("nan")
    means = []
    for rewards in rewards_per_prompt:
        take = list(rewards)[:k]
        if take:
            means.append(sum(take) / len(take))
    return sum(means) / len(means) if means else float("nan")


def expected_calibration_error(confidences, correct, n_bins: int = 5) -> dict:
    """ECE: sum over bins of (count/n) * |accuracy - mean confidence|.

    confidences are in [0, 1]. If you do not have a confidence, skip this section.
    """
    conf = np.asarray(list(confidences), dtype=float)
    corr = np.asarray(list(correct), dtype=float)
    if conf.size == 0:
        return {"ece": None, "bins": []}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf >= edges[i]) & (conf < edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": 0, "acc": None, "conf": None})
            continue
        acc = float(corr[mask].mean())
        mean_conf = float(conf[mask].mean())
        ece += (count / conf.size) * abs(acc - mean_conf)
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": count, "acc": acc, "conf": mean_conf})
    return {"ece": float(ece), "bins": bins}


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """CI for the mean of (a - b) on paired items. excludes_zero is the call at 95%."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.shape != y.shape or x.size == 0:
        return {"n": int(x.size), "mean_diff": None, "lo": None, "hi": None, "excludes_zero": False}
    diff = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "n": int(diff.size),
        "mean_diff": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "seed": seed,
    }


def mcnemar(correct_a, correct_b) -> dict:
    """McNemar on paired correct/incorrect bits. b = only B right, c = only A right.

    The statistic is (b-c)^2 / (b+c). We do not attach a p-value unless scipy is present.
    With b+c small, the test is weak. Say so instead of rounding a tiny p.
    """
    a = np.asarray(list(correct_a), dtype=bool)
    b = np.asarray(list(correct_b), dtype=bool)
    only_b = int((~a & b).sum())
    only_a = int((a & ~b).sum())
    both = int((a & b).sum())
    neither = int((~a & ~b).sum())
    discordant = only_a + only_b
    stat = 0.0 if discordant == 0 else (only_b - only_a) ** 2 / discordant
    out = {"n": int(a.size), "only_a": only_a, "only_b": only_b, "both": both, "neither": neither, "statistic": stat}
    if discordant == 0:
        out["note"] = "no disagreements"
        return out
    try:
        from scipy.stats import chi2

        out["p_chi2"] = float(chi2.sf(stat, 1))
    except Exception:
        out["p_chi2"] = None
    return out


def length_stats(lengths) -> dict:
    arr = np.asarray(list(lengths), dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def per_group(rows: list[dict], key: str, score_key: str = "correct") -> dict:
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "unknown")), []).append(float(row.get(score_key, 0.0)))
    return {name: {"n": len(vals), "mean": float(sum(vals) / len(vals))} for name, vals in sorted(groups.items())}


def method_comparison(by_method: dict[str, list[float]], reference: str = "base", seed: int = 0) -> dict:
    """Paired comparison. Lists must be aligned on the same items."""
    if reference not in by_method:
        return {"status": NOT_RUN, "reason": f"no {reference} scores"}
    ref = by_method[reference]
    table = {}
    for name, scores in by_method.items():
        summary = bootstrap_mean_ci(scores, seed=seed)
        if name == reference:
            table[name] = {"summary": summary, "versus_reference": None}
        else:
            table[name] = {"summary": summary, "versus_reference": paired_bootstrap(scores, ref, seed=seed)}
    return table


def reward_ladder_table(scores_by_reward: dict | None) -> dict:
    if not scores_by_reward:
        return {"status": NOT_RUN}
    return {"status": "computed", "scores": scores_by_reward}


def empty_report() -> dict:
    return {
        "status": NOT_RUN,
        "note": "No checkpoint was scored. Task metrics are not run.",
        "accuracy": NOT_RUN,
        "pass_at_k": NOT_RUN,
        "average_at_k": NOT_RUN,
        "by_task": NOT_RUN,
        "by_subset": NOT_RUN,
        "calibration": NOT_RUN,
        "length": NOT_RUN,
        "reward_ladder": {"status": NOT_RUN},
        "hacking": NOT_RUN,
        "contamination": NOT_RUN,
        "methods": NOT_RUN,
        "paired": NOT_RUN,
        "held_out_gold": NOT_RUN,
    }


def build_report(*, rows: list[dict] | None = None, samples_by_prompt: dict | None = None,
                 ladder: dict | None = None, hacking: dict | None = None, contamination: dict | None = None,
                 methods: dict | None = None, gold_rows: list[dict] | None = None, seed: int = 0,
                 ks: tuple[int, ...] = (1, 4)) -> dict:
    """rows: one per item, with correct (0/1), optional reward, task, subset, confidence, length, id.

    samples_by_prompt: prompt id -> list of rewards or correctness bits, for pass@k and average@k.
    """
    if not rows and not methods and not ladder and not hacking and not contamination:
        return empty_report()
    report = empty_report()
    report["status"] = "partial"
    report["note"] = "Numbers below were computed from the records passed in. Anything still marked 'not run' was not computed."
    report["seed"] = seed
    if rows:
        correct = [float(r.get("correct", 0.0)) for r in rows]
        acc = bootstrap_mean_ci(correct, seed=seed)
        acc["binomial_se"] = binomial_se(acc["mean"], acc["n"]) if acc["mean"] is not None else None
        report["accuracy"] = acc
        report["by_task"] = per_group(rows, "task", "correct")
        report["by_subset"] = per_group(rows, "subset", "correct")
        if any("confidence" in r for r in rows):
            conf_rows = [r for r in rows if "confidence" in r]
            report["calibration"] = expected_calibration_error(
                [r["confidence"] for r in conf_rows], [r.get("correct", 0.0) for r in conf_rows],
            )
        if any("length" in r for r in rows):
            report["length"] = length_stats([r["length"] for r in rows if "length" in r])
        if any("reward" in r for r in rows):
            report["reward"] = bootstrap_mean_ci([r["reward"] for r in rows if "reward" in r], seed=seed)
    if samples_by_prompt:
        # A sample counts as a pass at 0.5. Binary 0/1 bits work. A partial reward
        # below 0.5 does not. Say which threshold you used; do not silently move it.
        prompt_lists = [[float(s) for s in samples] for samples in samples_by_prompt.values()]
        report["pass_at_k"] = {}
        report["average_at_k"] = {}
        report["pass_threshold"] = 0.5
        for k in ks:
            passes = []
            for samples in prompt_lists:
                bits = [1 if s >= 0.5 else 0 for s in samples]
                kk = min(k, len(bits))
                passes.append(pass_at_k(len(bits), sum(bits), kk))
            report["pass_at_k"][str(k)] = float(np.nanmean(passes)) if passes else None
            report["average_at_k"][str(k)] = average_at_k(prompt_lists, k)
    if ladder is not None:
        report["reward_ladder"] = reward_ladder_table(ladder)
    if hacking is not None:
        report["hacking"] = {"status": "computed", "flagged": hacking.get("flagged", []), "markdown": hacking.get("markdown")}
    if contamination is not None:
        report["contamination"] = contamination if contamination.get("status") else {"status": "computed", **contamination}
    if methods is not None:
        report["methods"] = method_comparison(methods, seed=seed)
    if gold_rows:
        report["held_out_gold"] = bootstrap_mean_ci([float(r.get("correct", r.get("reward", 0.0))) for r in gold_rows], seed=seed)
        report["held_out_gold"]["n_items"] = len(gold_rows)
        report["held_out_gold"]["used_for_training"] = False
    return report


def demo_toy_report(seed: int = 0) -> dict:
    """A report on made-up rows so the code path runs. The numbers are the rows, not a model."""
    rng = random.Random(seed)
    rows = []
    for i in range(40):
        correct = 1 if rng.random() < 0.5 else 0
        rows.append({
            "id": i,
            "correct": correct,
            "reward": correct,
            "task": "pqa" if i < 20 else "err",
            "subset": "a" if i % 2 == 0 else "b",
            "confidence": 0.8 if correct else 0.3,
            "length": 10 + (i % 5),
        })
    samples = {i: [rng.random() < 0.4 for _ in range(4)] for i in range(10)}
    # Convert bools to floats for average@k.
    samples = {i: [float(b) for b in bits] for i, bits in samples.items()}
    methods = {
        "base": [r["correct"] for r in rows],
        "sft": [1.0 if r["correct"] or rng.random() < 0.05 else 0.0 for r in rows],
    }
    return build_report(
        rows=rows,
        samples_by_prompt=samples,
        ladder=None,
        hacking=None,
        contamination={"status": NOT_RUN},
        methods=methods,
        gold_rows=rows[:8],
        seed=seed,
    )
