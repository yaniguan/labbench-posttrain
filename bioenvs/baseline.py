"""Roll models out in the RL envs to check that each reward is learnable (not just well-formed).

    python -m bioenvs.baseline --models haiku-5.5 gpt-6-luna glm-5.3-flash --n 100          # greedy, per-model reward
    python -m bioenvs.baseline --models haiku-5.5 --n 60 --k 4 --temperature 1.0 --tag grpo  # group variance

Reports per env/model: mean reward, format-parse rate, reward components, per-subtype means and (k>1)
the share of prompts whose k samples have non-zero reward variance - i.e. prompts that would give GRPO a
non-zero advantage. Prompts where every sample gets the same reward carry no gradient.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from bioenvs import make_env
from common.llm import LLM

RESULTS = Path(__file__).resolve().parents[1] / "results" / "protocol_reasoning"


async def rollout(env_name: str, model: str, n: int, k: int, temperature: float, max_tokens: int, conc: int):
    env = make_env(env_name, split="test", seed=123)
    llm = LLM(model, max_tokens=max_tokens, temperature=temperature, concurrency=conc)
    samples = [env.sample(i) for i in range(n)]
    pbar = tqdm(total=n * k, desc=f"{env_name}/{model}", ncols=80)

    async def one(s, j):
        r = await llm.complete(s.prompt, tag=f"rl{j}")
        pbar.update(1)
        return env.score(s, r["text"]), r

    res = await asyncio.gather(*[asyncio.gather(*[one(s, j) for j in range(k)]) for s in samples])
    pbar.close()
    await llm.aclose()
    rows = []
    for s, group in zip(samples, res):
        rows.append({"id": s.id, "type": s.info.get("type"),
                     "rewards": [r.value for r, _ in group], "parsed": [r.parsed for r, _ in group],
                     "components": [r.components for r, _ in group],
                     "truncated": [x["finish_reason"] == "length" for _, x in group],
                     "api_error": [x["error"] is not None for _, x in group],
                     "completion_tokens": [x["usage"].get("completion", 0) for _, x in group],
                     "sample_output": group[0][1]["text"][:600]})
    return rows, llm.cost


def summarise(rows):
    flat = [(v, p) for r in rows for v, p in zip(r["rewards"], r["parsed"])]
    comps = defaultdict(list)
    for r in rows:
        for c in r["components"]:
            for key, v in c.items():
                if v >= 0:
                    comps[key].append(v)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]] += r["rewards"]
    out = {"n_prompts": len(rows), "k": len(rows[0]["rewards"]),
           "mean_reward": st.mean(v for v, _ in flat), "parse_rate": st.mean(p for _, p in flat),
           "truncated_rate": st.mean(t for r in rows for t in r["truncated"]),
           "api_error_rate": st.mean(t for r in rows for t in r["api_error"]),
           "mean_completion_tokens": st.mean(t for r in rows for t in r["completion_tokens"]),
           "components": {k_: st.mean(v) for k_, v in comps.items()},
           "by_type": {t: st.mean(v) for t, v in by_type.items()}}
    if out["k"] > 1:
        stds = [st.pstdev(r["rewards"]) for r in rows]
        out["mean_group_std"] = st.mean(stds)
        out["frac_prompts_with_variance"] = st.mean(s > 1e-9 for s in stds)
        out["mean_best_of_k"] = st.mean(max(r["rewards"]) for r in rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["pqa", "ord", "err", "gen"])
    ap.add_argument("--models", nargs="+", default=["haiku-5.5", "gpt-6-luna", "glm-5.3-flash"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--tag", default="greedy")
    a = ap.parse_args()

    async def go():
        jobs = [(e, m) for e in a.envs for m in a.models]
        outs = await asyncio.gather(*[rollout(e, m, a.n, a.k, a.temperature, a.max_tokens, a.concurrency) for e, m in jobs])
        summary = {}
        for (e, m), (rows, cost) in zip(jobs, outs):
            summary.setdefault(e, {})[m] = {**summarise(rows), "cost_usd": cost}
            (RESULTS / f"rl_{a.tag}_{e}_{m}.json").write_text(json.dumps(rows))
        (RESULTS / f"rl_summary_{a.tag}.json").write_text(json.dumps(summary, indent=1))
        for e, d in summary.items():
            for m, s in d.items():
                extra = f" var-prompts={s['frac_prompts_with_variance']:.2f}" if "frac_prompts_with_variance" in s else ""
                print(f"{e:4s} {m:14s} reward={s['mean_reward']:.3f} parse={s['parse_rate']:.2f} trunc={s['truncated_rate']:.2f}{extra}")

    asyncio.run(go())


if __name__ == "__main__":
    main()
