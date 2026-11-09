"""Aggregate results/protocol_reasoning/labbench_*.jsonl -> labbench_summary.{json,md} and a figure.

Headline = macro-average accuracy over the three families (ProtocolQA / CloningScenarios / SeqQA), so the
600 SeqQA items don't drown out the 108 ProtocolQA items. CIs: bootstrap over questions (both choice-order
seeds of a question move together). Model-vs-model gaps: paired bootstrap on the same questions.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results" / "protocol_reasoning"
FAMS = ["ProtocolQA", "CloningScenarios", "SeqQA"]   # overridden by --families
rng = np.random.default_rng(0)


def _f1(p: float, c: float) -> float:
    """Harmonic mean of precision (correct / answered) and coverage (answered / total)."""
    return 2 * p * c / (p + c) if p + c else 0.0


def load():
    rows = {}
    for f in RESULTS.glob("labbench_*_seed?.jsonl"):   # excludes *.budget*.jsonl backups of truncated runs
        for l in f.read_text().splitlines():
            r = json.loads(l)
            rows[(r["model"], r["seed"], r["id"])] = r
    return list(rows.values())


def per_question(rows, model, field):
    """{(family, id): mean over seeds of field}"""
    acc = defaultdict(list)
    for r in rows:
        if r["model"] == model:
            acc[(r["family"], r["id"])].append(float(field(r)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def macro(vals_by_q, idx_by_fam):
    return float(np.mean([np.mean([vals_by_q[i] for i in idx]) for idx in idx_by_fam.values()]))


def boot_macro(q_vals: dict, B=2000, other: dict | None = None):
    keys = sorted(set(q_vals) & set(other)) if other else sorted(q_vals)   # paired: only questions both models answered
    by_fam = {f: [k for k in keys if k[0] == f] for f in FAMS}
    arrs = {f: np.array([q_vals[k] - (other[k] if other else 0) for k in by_fam[f]]) for f in FAMS}
    point = float(np.mean([a.mean() for a in arrs.values()]))
    draws = [np.mean([a[rng.integers(0, len(a), len(a))].mean() for a in arrs.values()]) for _ in range(B)]
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    global FAMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="restrict to these choice-order seeds")
    a = ap.parse_args()
    FAMS = a.families
    sfx = f"_{a.tag}" if a.tag else ""
    rows = [r for r in load() if r["family"] in FAMS and (a.seeds is None or r["seed"] in a.seeds)]
    models = sorted({r["model"] for r in rows})
    out = {"models": {}, "pairwise": {}, "n_rows": len(rows)}
    qv = {}
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        qv[m] = {"correct": per_question(rows, m, lambda r: r["correct"])}
        d = {"n_rows": len(mr), "seeds": sorted({r["seed"] for r in mr})}
        d["accuracy_macro"] = boot_macro(qv[m]["correct"])
        d["families"] = {}
        for fam in FAMS:
            fr = [r for r in mr if r["family"] == fam]
            if not fr:
                continue
            n_sure = sum(r["sure"] for r in fr)
            d["families"][fam] = {
                "n": len(fr), "accuracy": float(np.mean([r["correct"] for r in fr])),
                "precision": sum(r["correct"] for r in fr) / n_sure if n_sure else 0.0,   # LAB-Bench: correct / answered-not-refused
                "coverage": n_sure / len(fr),
                "f1": _f1(sum(r["correct"] for r in fr) / n_sure if n_sure else 0.0, n_sure / len(fr)),
                "chance": float(np.mean([1 / r["n_choices"] for r in fr])),
            }
        fs = d["families"].values()
        d["macro4"] = {k: float(np.mean([v[k] for v in fs])) for k in ("accuracy", "coverage", "precision", "f1")}
        # SeqQA breakdown by subset family prefix (ORF / PCR / RE / Prop)
        sq = defaultdict(list)
        for r in mr:
            if r["family"] == "SeqQA":
                sq[r["subset"].split("-")[0]].append(r["correct"])
        d["seqqa_by_type"] = {k: float(np.mean(v)) for k, v in sq.items()}
        d["parse_fail_rate"] = float(np.mean([r["parse_fail"] for r in mr]))
        d["content_filter_rate"] = float(np.mean([r["finish_reason"] == "content_filter" for r in mr]))
        d["truncated_rate"] = float(np.mean([r["finish_reason"] == "length" for r in mr]))
        ans = [r for r in mr if r["finish_reason"] != "content_filter"]
        d["accuracy_excl_content_filter"] = float(np.mean([r["correct"] for r in ans]))
        d["avg_completion_tokens"] = float(np.mean([r["usage"]["completion"] for r in mr]))
        d["avg_reasoning_tokens"] = float(np.mean([r["usage"].get("reasoning", 0) for r in mr]))
        d["cost_usd_total"] = float(sum(r["usage"].get("cost", 0) for r in mr))
        # order sensitivity: how often does the model pick a different *content* when the options are reshuffled?
        # (only meaningful with >=2 seeds) -> compare correctness agreement across seeds
        by = defaultdict(dict)
        for r in mr:
            by[r["id"]][r["seed"]] = r["correct"]
        flips = [len({v for v in s.values()}) > 1 for s in by.values() if len(s) >= 2]
        d["correctness_flip_rate_across_choice_orders"] = float(np.mean(flips)) if flips else None
        out["models"][m] = d
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            p, lo, hi = boot_macro(qv[a]["correct"], other=qv[b]["correct"])
            out["pairwise"][f"{a} - {b}"] = {"diff": p, "ci95": [lo, hi], "significant": lo > 0 or hi < 0}
    (RESULTS / f"labbench_summary{sfx}.json").write_text(json.dumps(out, indent=1))

    short = {"CloningScenarios": "Cloning", "ProtocolQA": "ProtocolQA"}
    L = ["| model | macro acc [95% CI] | " + " | ".join(short.get(f, f) for f in FAMS) +
         " | macro precision | macro coverage | filtered | trunc | tok/q | $ |",
         "|" + "---|" * (len(FAMS) + 7)]
    for m, d in out["models"].items():
        a_, lo, hi = d["accuracy_macro"]
        f = d["families"]
        L.append(f"| {m} | {a_:.3f} [{lo:.3f}, {hi:.3f}] | " + " | ".join(f"{f[x]['accuracy']:.3f}" if x in f else "-" for x in FAMS) +
                 f" | {np.mean([v['precision'] for v in f.values()]):.3f} | {np.mean([v['coverage'] for v in f.values()]):.3f}"
                 f" | {d['content_filter_rate']:.3f} | {d['truncated_rate']:.3f} | {d['avg_completion_tokens']:.0f} | {d['cost_usd_total']:.2f} |")
    L.append("")
    L += [f"- {k}: {v['diff']:+.3f} macro-acc, 95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
          f"{'  (significant)' if v['significant'] else '  (not significant)'}" for k, v in out["pairwise"].items()]
    (RESULTS / f"labbench_summary{sfx}.md").write_text("\n".join(L))
    print("\n".join(L))
    M = ["| model | " + " | ".join(FAMS) + " | **macro** |", "|---|" + "---|" * (len(FAMS) + 1)]
    for m, d in out["models"].items():
        for k, lab in (("accuracy", "acc"), ("coverage", "cov"), ("precision", "prec"), ("f1", "F1")):
            M.append(f"| {m} {lab} | " + " | ".join(f"{d['families'][f][k]:.3f}" if f in d["families"] else "-" for f in FAMS)
                     + f" | **{d['macro4'][k]:.3f}** |")
    (RESULTS / f"labbench_metrics4{sfx}.md").write_text("\n".join(M))
    print("\n".join(M))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(FAMS) + 3), 3.8))
        w = 0.25
        for j, m in enumerate(models):
            vals = [out["models"][m]["families"][f]["accuracy"] for f in FAMS] + [out["models"][m]["accuracy_macro"][0]]
            ax.bar(np.arange(len(FAMS) + 1) + (j - 1) * w, vals, w, label=m)
        chance = [out["models"][models[0]]["families"][f]["chance"] for f in FAMS]
        ax.set_xticks(range(len(FAMS) + 1), FAMS + ["macro"], rotation=20)
        for i, c in enumerate(chance):
            ax.hlines(c, i - 0.4, i + 0.4, colors="k", linestyles=":", lw=1)
        ax.set_ylabel("accuracy"); ax.set_ylim(0, 1); ax.legend(frameon=False); ax.set_title("LAB-Bench accuracy by family (dotted = chance)")
        fig.tight_layout(); fig.savefig(RESULTS / f"labbench_accuracy{sfx}.png", dpi=160)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
