"""Same questions, three output-token budgets: 4,096 and 12,000 (fixed) and as-needed (64,000 cap).

ProtocolQA + CloningScenarios, answer-order seed 0, three models.
* 12k   : the actual 12k run (GLM: results/protocol_reasoning/*.budgetorig.jsonl, the rows before its 32k re-run).
* 4k    : simulated from the 12k run. Output was generated with temperature 0 and token counts include reasoning, so a
          response that used more than 4,096 tokens would have been cut off at 4,096 and is scored as unanswered.
* as-needed : questions truncated at 12k are re-run with a 64k cap (results/protocol_reasoning/asneeded). Questions that finished under a
          smaller cap are kept as they are.
    python -m labeval.budget
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results" / "protocol_reasoning"
MODELS = ["haiku-5.5", "gpt-6-luna", "glm-5.3-flash"]
FAMS = ["ProtocolQA", "CloningScenarios"]


def rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def base_12k(m: str):
    p = R / f"labbench_{m}_seed0.budgetorig.jsonl"
    p = p if p.exists() and m.startswith("glm") else R / f"labbench_{m}_seed0.jsonl"
    return [r for r in rows(p) if r["family"] in FAMS]


def cap_rows(rs, cap):
    """Apply a token cap to rows generated under a larger one."""
    out = []
    for r in rs:
        cut = r["finish_reason"] == "length" or r["usage"]["completion"] > cap
        out.append({**r, "answered": (not cut) and r["pred"] is not None and r["pred"] != r["unsure"],
                    "correct_c": (not cut) and r["correct"], "cut": cut, "tok": min(r["usage"]["completion"], cap)})
    return out


def _one(g):
    n, c, a = len(g), sum(r["correct_c"] for r in g), sum(r["answered"] for r in g)
    prec, cov = (c / a if a else 0.0), a / n
    return {"n": n, "acc": c / n, "cov": cov, "prec": prec, "f1": 2 * prec * cov / (prec + cov) if prec + cov else 0.0,
            "trunc": float(np.mean([r["cut"] for r in g])), "tok_mean": float(np.mean([r["tok"] for r in g]))}


def metrics(rs):
    d = {f: _one([r for r in rs if r["family"] == f]) for f in FAMS}
    d["macro"] = {k: float(np.mean([d[f][k] for f in FAMS])) for k in ("acc", "cov", "prec", "f1", "trunc", "tok_mean")}
    return d


def main():
    out, tok, unfinished = {}, {}, {}
    for m in MODELS:
        b12 = base_12k(m)
        an = [r for r in rows(R / "asneeded" / f"labbench_{m}_seed0.jsonl") if r["family"] in FAMS]
        # requests still unfinished when the run was stopped (they were generating towards the 64k cap) count as cut off at 64k
        have = {r["id"] for r in an}
        missing = [{**r, "finish_reason": "length", "pred": None, "correct": False, "usage": {**r["usage"], "completion": 64000}}
                   for r in b12 if r["id"] not in have]
        an = an + missing
        unfinished[m] = len(missing)
        views = {"4k (fixed)": cap_rows(b12, 4096), "12k (fixed)": cap_rows(b12, 12000), "as needed (64k cap)": cap_rows(an, 64000)}
        out[m] = {k: {"n_rows": len(v), **metrics(v)} for k, v in views.items()}
        t = np.array([r["usage"]["completion"] for r in an])
        tok[m] = {"mean": float(t.mean()), "median": float(np.median(t)), "p90": float(np.percentile(t, 90)), "max": int(t.max()),
                  "n_rows": len(an)}
    (R / "labbench_budget.json").write_text(json.dumps({"metrics": out, "tokens_as_needed": tok, "unfinished_at_64k": unfinished}, indent=1))
    L = ["| model | budget | ProtocolQA acc | Cloning acc | macro acc | macro cov | macro prec | macro F1 | trunc (PQA / Cloning) | mean tokens/q |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for k, v in out[m].items():
            P, C, M = v["ProtocolQA"], v["CloningScenarios"], v["macro"]
            L.append(f"| {m} | {k} | {P['acc']:.2f} | {C['acc']:.2f} | {M['acc']:.2f} | {M['cov']:.2f} | {M['prec']:.2f} | {M['f1']:.2f} | "
                     f"{P['trunc']:.2f} / {C['trunc']:.2f} | {M['tok_mean']:.0f} |")
    L += ["", "Tokens actually used per question, as-needed run (completion incl. reasoning):", "",
          "| model | mean | median | p90 | max | rows |", "|---|---|---|---|---|---|"]
    L += [f"| {m} | {t['mean']:.0f} | {t['median']:.0f} | {t['p90']:.0f} | {t['max']} | {t['n_rows']} |" for m, t in tok.items()]
    L += ["", "Rows counted as cut off at 64k because the request had not finished when the run was stopped: "
          + ", ".join(f"{m} {n}" for m, n in unfinished.items())]
    (R / "labbench_budget.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
