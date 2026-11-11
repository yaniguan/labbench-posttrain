"""One report per checkpoint.

Small test sets are noisy. With n questions and accuracy p, the standard
error of the mean is sqrt(p * (1 - p) / n). At p = 0.5 and n = 30 that is
about 0.091, nine points. Shao et al. (Appendix D, Figure 13 caption) say
AIME has 30 questions and gaps under about 2 points may not be significant.
The standard error says the noise is larger than 2 points. Do not rank two
lab runs on 30 items by a gap of a few points.

Nothing in this module invents a model score. Pass the rows you measured.
If you have no rows, the report says not run.
"""
from __future__ import annotations

import math
import random

from scipy.stats import binomtest


def accuracy(correct: list[bool] | list[int]) -> float | None:
    if not correct:
        return None
    return sum(float(c) for c in correct) / len(correct)


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile interval of the mean. Returns the mean of `values`, not a guess."""
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0, "status": "not run"}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_i = int((alpha / 2) * n_boot)
    hi_i = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return {
        "mean": sum(values) / n,
        "low": means[lo_i],
        "high": means[hi_i],
        "n": n,
        "n_boot": n_boot,
        "status": "computed",
    }


def _comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))


def pass_at_k(n_correct: int, n: int, k: int) -> float | None:
    """Unbiased pass@k. None when the group is smaller than k.

    1 - C(n - c, k) / C(n, k). A group with no correct sample scores 0.
    """
    if n < k or k < 1:
        return None
    if n_correct <= 0:
        return 0.0
    if n - n_correct < k:
        return 1.0
    return 1.0 - _comb(n - n_correct, k) / _comb(n, k)


def average_at_k(trials: list[bool], k: int) -> float | None:
    """Mean of per-sample correctness, using the first k trials.

    pass@k asks whether any of the k is right. average@k asks how many are
    right. On a lab bench you pay for every sample, so the average matters.
    """
    if len(trials) < k or k < 1:
        return None
    return sum(float(c) for c in trials[:k]) / k


def binomial_se(p: float, n: int) -> float | None:
    if n <= 0:
        return None
    return math.sqrt(p * (1 - p) / n)


def paired_bootstrap(a: list[bool], b: list[bool], n_boot: int = 1000, seed: int = 0) -> dict:
    """Interval for the mean of a - b on paired items.

    Same questions, two methods. The interval is on the paired difference.
    """
    if len(a) != len(b) or not a:
        return {"diff": None, "low": None, "high": None, "status": "not run"}
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    ci = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    return {"diff": ci["mean"], "low": ci["low"], "high": ci["high"], "n": ci["n"], "status": "computed"}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact McNemar test on paired correct/incorrect bits.

    n10: a right, b wrong. n01: a wrong, b right. The p-value is the
    two-sided binomial test that those discordant pairs split evenly.
    """
    if len(a) != len(b) or not a:
        return {"n10": None, "n01": None, "p_value": None, "status": "not run"}
    n10 = sum(bool(x) and not bool(y) for x, y in zip(a, b))
    n01 = sum((not bool(x)) and bool(y) for x, y in zip(a, b))
    n = n10 + n01
    if n == 0:
        p = 1.0
    else:
        p = float(binomtest(n10, n, 0.5).pvalue)
    return {"n10": n10, "n01": n01, "p_value": p, "status": "computed"}


def calibration(confidences: list[float], correct: list[bool], n_bins: int = 5) -> list[dict]:
    """Bin predicted confidence against accuracy. Empty input returns []."""
    if not confidences or len(confidences) != len(correct):
        return []
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == n_bins - 1 and c == 1)]
        if not idx:
            bins.append({"low": lo, "high": hi, "n": 0, "confidence": None, "accuracy": None})
            continue
        bins.append({
            "low": lo,
            "high": hi,
            "n": len(idx),
            "confidence": sum(confidences[i] for i in idx) / len(idx),
            "accuracy": sum(float(correct[i]) for i in idx) / len(idx),
        })
    return bins


def length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"n": 0, "mean": None, "min": None, "max": None, "status": "not run"}
    return {
        "n": len(lengths),
        "mean": sum(lengths) / len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "status": "computed",
    }


def per_group(rows: list[dict], key: str) -> dict:
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "unknown")), []).append(bool(row.get("correct")))
    return {name: accuracy(vals) for name, vals in groups.items()}


def noise_note(n: int, p: float = 0.5) -> str:
    se = binomial_se(p, n)
    if se is None:
        return "No items, so there is no accuracy to trust."
    return (
        f"With {n} questions and accuracy {p:.2f}, the standard error is about {se:.3f}. "
        "A gap smaller than that is noise. Shao et al. flag gaps under about 2 points "
        "on AIME (30 questions) as not significant; the standard error at a coin-flip "
        "accuracy is about 9 points, so 2 points is a low bar for caution."
    )


def build_report(
    *,
    rows: list[dict] | None = None,
    methods: dict[str, list[bool]] | None = None,
    ladder: dict | None = None,
    hacking: dict | None = None,
    contamination: dict | None = None,
    lengths: list[int] | None = None,
    confidences: list[float] | None = None,
    heldout_gold: list[bool] | None = None,
    status: str = "not run",
) -> dict:
    """Assemble the checkpoint report.

    `rows` items may carry correct, task, subset, confidence.
    `methods` maps a method name to paired correct bits, same order.
    Pass status='toy demo' when the bits are a unit-test fixture.
    """
    report = {"status": status}
    if not rows and not methods:
        report["note"] = "No checkpoint was scored. Accuracy, the ladder, and the comparisons are not run."
        report["noise"] = noise_note(30)
        report["ladder"] = ladder or {"status": "not run"}
        report["hacking"] = hacking or {"status": "not run"}
        report["contamination"] = contamination or {"status": "not run"}
        report["lengths"] = length_stats(lengths or [])
        report["heldout_gold"] = {"status": "not run"}
        return report
    correct = [bool(r.get("correct")) for r in (rows or [])]
    report["accuracy"] = bootstrap_ci([float(c) for c in correct]) if correct else {"status": "not run"}
    report["n"] = len(correct)
    report["noise"] = noise_note(len(correct) or 30, report["accuracy"].get("mean") or 0.5)
    if rows and any("task" in r for r in rows):
        report["per_task"] = per_group(rows, "task")
    if rows and any("subset" in r for r in rows):
        report["per_subset"] = per_group(rows, "subset")
    if rows and "k_correct" in rows[0]:
        report["pass_at_k"] = []
        for row in rows:
            for k in (1, 4, 8):
                report["pass_at_k"].append({
                    "id": row.get("id"),
                    "k": k,
                    "pass_at_k": pass_at_k(int(row["k_correct"]), int(row["k_n"]), k),
                    "average_at_k": average_at_k(row.get("k_trials", []), k) if row.get("k_trials") else None,
                })
    if confidences and correct:
        report["calibration"] = calibration(confidences, correct)
    elif rows and any("confidence" in r for r in rows):
        report["calibration"] = calibration(
            [float(r.get("confidence", 0.0)) for r in rows],
            correct,
        )
    else:
        report["calibration"] = []
    report["lengths"] = length_stats(lengths or [int(r.get("length", 0)) for r in (rows or []) if "length" in r])
    report["ladder"] = ladder or {"status": "not run"}
    report["hacking"] = hacking or {"status": "not run"}
    report["contamination"] = contamination or {"status": "not run"}
    report["methods"] = {}
    if methods:
        names = list(methods)
        for name, bits in methods.items():
            report["methods"][name] = bootstrap_ci([float(c) for c in bits])
        if len(names) >= 2:
            report["paired"] = paired_bootstrap(methods[names[0]], methods[names[1]])
            report["mcnemar"] = mcnemar(methods[names[0]], methods[names[1]])
            report["paired_methods"] = names[:2]
    if heldout_gold is not None:
        report["heldout_gold"] = bootstrap_ci([float(c) for c in heldout_gold])
    else:
        report["heldout_gold"] = {"status": "not run"}
    return report


def render_markdown(report: dict) -> str:
    lines = ["# Evaluation report", "", f"Status: {report.get('status', 'not run')}", ""]
    if report.get("note"):
        lines.append(report["note"])
        lines.append("")
    acc = report.get("accuracy") or {}
    if acc.get("mean") is None:
        lines.append("Accuracy: not run")
    else:
        lines.append(
            f"Accuracy: {acc['mean']:.3f} (bootstrap {acc['low']:.3f} to {acc['high']:.3f}, n={acc['n']})"
        )
    lines.append("")
    lines.append(report.get("noise", ""))
    lines.append("")
    lines.append("Held-out gold is the metric the monitor watches. It is not a training reward.")
    gold = report.get("heldout_gold") or {}
    if gold.get("mean") is None:
        lines.append("Held-out gold: not run")
    else:
        lines.append(f"Held-out gold: {gold['mean']:.3f}")
    lines.append("")
    if report.get("methods"):
        lines.append("## Methods")
        lines.append("")
        for name, block in report["methods"].items():
            if block.get("mean") is None:
                lines.append(f"- {name}: not run")
            else:
                lines.append(f"- {name}: {block['mean']:.3f}")
        if report.get("mcnemar", {}).get("status") == "computed":
            m = report["mcnemar"]
            lines.append(
                f"Paired test ({' vs '.join(report.get('paired_methods', []))}): "
                f"McNemar p={m['p_value']:.4f} (n10={m['n10']}, n01={m['n01']})"
            )
        lines.append("")
    lines.append("## Reward ladder")
    lines.append("")
    lines.append(str(report.get("ladder", {}).get("status", "not run")))
    lines.append("")
    lines.append("## Hacking checks")
    lines.append("")
    lines.append(str(report.get("hacking", {}).get("status", report.get("hacking"))))
    lines.append("")
    lines.append("## Contamination")
    lines.append("")
    lines.append(str(report.get("contamination", {}).get("status", report.get("contamination"))))
    lines.append("")
    return "\n".join(lines)
