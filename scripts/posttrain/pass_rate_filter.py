"""Offline pass-rate prompt difficulty filter for ORD, ERR, and BioProBench train splits (#17).

Samples `k` completions per `train` prompt using either:
  - the local offline tiny GPT-2 model (`--smoke`),
  - a local HuggingFace causal LM (`--model PATH_OR_ID`), or
  - a cached completions/rewards JSONL file (`--cached PATH`),
and writes per-prompt `mean_reward`, `std_reward`, and `kept` flags to JSONL plus a summary JSON
with `kept_fraction` per environment and run metadata (`common.run_meta.build_run_meta`).

Filtering is strictly enforced on `split == "train"` only; `val` and `test` splits are rejected.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bioenvs import make_env  # noqa: E402
from common.run_meta import build_run_meta  # noqa: E402


def _prompt_keep(mean_reward: float, std_reward: float, *, low: float = 0.1, high: float = 0.9, min_std: float = 1e-6) -> bool:
    return bool(low <= float(mean_reward) <= high and float(std_reward) > float(min_std))


def load_pass_rate_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_kept_ids(
    source: str | Path | Iterable[dict[str, Any]],
    *,
    low: float = 0.1,
    high: float = 0.9,
    min_std: float = 1e-6,
    env_name: str | None = None,
) -> set[str]:
    """Return prompt IDs passing the `[low, high]` mean-reward band and `std_reward > min_std`."""
    rows = load_pass_rate_rows(source) if isinstance(source, (str, Path)) else list(source)
    kept: set[str] = set()
    for row in rows:
        if row.get("split", "train") != "train":
            raise ValueError(f"Pass-rate filter file contains non-train split={row.get('split')!r}")
        if env_name is not None and row.get("env") not in (None, env_name):
            continue
        mean_r = float(row["mean_reward"])
        std_r = float(row["std_reward"])
        if _prompt_keep(mean_r, std_r, low=low, high=high, min_std=min_std):
            kept.add(str(row["id"]))
    return kept


def make_pass_rate_filter_fn(
    source: str | Path | set[str] | Iterable[dict[str, Any]],
    *,
    split: str = "train",
    low: float = 0.1,
    high: float = 0.9,
    min_std: float = 1e-6,
    env_name: str | None = None,
) -> Callable[[dict[str, Any]], bool]:
    """Build a `BioEnv(..., filter_fn=...)` predicate that only permits `split == 'train'`."""
    if split != "train":
        raise ValueError(f"Pass-rate prompt filtering is only allowed on split='train', got {split!r}")
    if isinstance(source, set):
        kept_ids = {str(x) for x in source}
    else:
        kept_ids = load_kept_ids(source, low=low, high=high, min_std=min_std, env_name=env_name)

    def _filter(row: dict[str, Any]) -> bool:
        return str(row.get("id", "")) in kept_ids

    _filter._train_only = True  # type: ignore[attr-defined]
    return _filter


def evaluate_cached_rows(
    cached_rows: list[dict[str, Any]],
    *,
    reward_version: str = "v3",
    low: float = 0.1,
    high: float = 0.9,
    min_std: float = 1e-6,
) -> list[dict[str, Any]]:
    """Compute per-prompt mean/std rewards from cached OpenRouter/HF completions or pre-scored rewards."""
    env_cache: dict[str, Any] = {}
    sample_maps: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []

    for rec in cached_rows:
        split = str(rec.get("split", "train"))
        if split != "train":
            raise ValueError(f"Refusing to filter non-train split={split!r} in cached pass-rate input.")
        env_name = str(rec.get("env", "ord"))
        pid = str(rec["id"])
        if "rewards" in rec and rec["rewards"] is not None:
            rewards = [float(x) for x in rec["rewards"]]
        else:
            completions = [str(c) for c in rec.get("completions", [])]
            if not completions:
                raise ValueError(f"Cached row {pid!r} must supply either 'rewards' or non-empty 'completions'.")
            if env_name not in env_cache:
                env = make_env(env_name, split="train", seed=0, reward_version=reward_version)
                env_cache[env_name] = env
                sample_maps[env_name] = {str(env.sample(i).id): env.sample(i) for i in range(len(env))}
            env = env_cache[env_name]
            samp = sample_maps[env_name].get(pid)
            if samp is None:
                from bioenvs.base import Sample
                samp = Sample(id=pid, prompt=str(rec.get("prompt", "")), info=dict(rec.get("info") or {}))
            rewards = [float(env.score(samp, comp).value) for comp in completions]

        mean_r = float(st.mean(rewards)) if rewards else 0.0
        std_r = float(st.pstdev(rewards)) if len(rewards) > 1 else 0.0
        out.append(
            {
                "env": env_name,
                "split": "train",
                "id": pid,
                "k": len(rewards),
                "mean_reward": round(mean_r, 6),
                "std_reward": round(std_r, 6),
                "rewards": [round(r, 6) for r in rewards],
                "kept": _prompt_keep(mean_r, std_r, low=low, high=high, min_std=min_std),
            }
        )
    return out


def sample_with_model(
    envs: list[str],
    *,
    k: int = 4,
    max_items: int = 16,
    model_name_or_path: str | None = None,
    smoke: bool = False,
    temperature: float = 1.0,
    max_new_tokens: int = 24,
    seed: int = 0,
    reward_version: str = "v3",
    low: float = 0.1,
    high: float = 0.9,
    min_std: float = 1e-6,
) -> list[dict[str, Any]]:
    """Sample `k` completions per `train` prompt with tiny CPU LM (`--smoke`) or a local HF model."""
    torch.manual_seed(seed)
    if smoke or not model_name_or_path:
        from posttrain.tiny import build_tiny_lm

        model, tokenizer = build_tiny_lm(seed=seed)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval()
    device = next(model.parameters()).device
    max_ctx = int(getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 256)))
    prompt_budget = max(16, max_ctx - max_new_tokens - 2)

    rows: list[dict[str, Any]] = []
    for env_name in envs:
        env = make_env(env_name, split="train", seed=seed, max_items=max_items, reward_version=reward_version)
        for i in range(len(env)):
            samp = env.sample(i)
            enc = tokenizer(
                samp.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=prompt_budget,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            p_len = int(input_ids.shape[1])
            rewards: list[float] = []
            for j in range(k):
                torch.manual_seed(seed + i * 97 + j)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=max(temperature, 1e-4),
                        pad_token_id=tokenizer.pad_token_id,
                    )
                comp = tokenizer.decode(gen[0, p_len:].tolist(), skip_special_tokens=True)
                rewards.append(float(env.score(samp, comp).value))
            mean_r = float(st.mean(rewards))
            std_r = float(st.pstdev(rewards)) if len(rewards) > 1 else 0.0
            rows.append(
                {
                    "env": env_name,
                    "split": "train",
                    "id": str(samp.id),
                    "k": k,
                    "mean_reward": round(mean_r, 6),
                    "std_reward": round(std_r, 6),
                    "rewards": [round(r, 6) for r in rewards],
                    "kept": _prompt_keep(mean_r, std_r, low=low, high=high, min_std=min_std),
                }
            )
    return rows


def summarize_pass_rates(
    rows: list[dict[str, Any]],
    *,
    low: float = 0.1,
    high: float = 0.9,
    min_std: float = 1e-6,
    reward_version: str = "v3",
    sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_env: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_env.setdefault(str(r["env"]), []).append(r)

    per_env: dict[str, dict[str, Any]] = {}
    for env_name, env_rows in sorted(by_env.items()):
        n_total = len(env_rows)
        n_kept = sum(1 for r in env_rows if _prompt_keep(r["mean_reward"], r["std_reward"], low=low, high=high, min_std=min_std))
        per_env[env_name] = {
            "n_total": n_total,
            "n_kept": n_kept,
            "kept_fraction": round(n_kept / n_total, 4) if n_total else 0.0,
            "mean_reward_avg": round(float(st.mean(r["mean_reward"] for r in env_rows)), 4) if n_total else 0.0,
            "nonzero_std_fraction": round(sum(1 for r in env_rows if float(r["std_reward"]) > min_std) / n_total, 4) if n_total else 0.0,
        }
    return {
        "band": {"low": low, "high": high, "min_std": min_std},
        "per_env": per_env,
        "meta": build_run_meta(reward_version=reward_version, sampling=sampling or {}),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Pass-rate prompt difficulty filter for BioProBench train splits (#17).")
    ap.add_argument("--envs", nargs="+", default=["ord", "err"], help="Train environments to score")
    ap.add_argument("--split", default="train", help="Must be 'train' (val/test are rejected)")
    ap.add_argument("--cached", type=str, default=None, help="Cached completions/rewards JSONL path")
    ap.add_argument("--model", type=str, default=None, help="Local HF causal LM path or ID")
    ap.add_argument("--smoke", action="store_true", help="Run on CPU with the offline tiny GPT-2 model")
    ap.add_argument("--k", type=int, default=4, help="Completions per prompt")
    ap.add_argument("--max-items", type=int, default=16, help="Max train prompts per env")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--low", type=float, default=0.1, help="Minimum mean reward to keep")
    ap.add_argument("--high", type=float, default=0.9, help="Maximum mean reward to keep")
    ap.add_argument("--min-std", type=float, default=1e-6, help="Minimum reward std to keep")
    ap.add_argument("--reward-version", default="v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/pass_rate_filter.jsonl", help="Output per-prompt JSONL")
    ap.add_argument("--summary-out", type=str, default=None, help="Optional summary JSON path")
    args = ap.parse_args(argv)

    if args.split != "train":
        raise ValueError(f"Pass-rate filtering is strictly restricted to split='train', got {args.split!r}")

    if args.cached:
        cached = load_pass_rate_rows(args.cached)
        rows = evaluate_cached_rows(
            cached,
            reward_version=args.reward_version,
            low=args.low,
            high=args.high,
            min_std=args.min_std,
        )
    else:
        rows = sample_with_model(
            list(args.envs),
            k=args.k,
            max_items=args.max_items,
            model_name_or_path=args.model,
            smoke=bool(args.smoke or not args.model),
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            reward_version=args.reward_version,
            low=args.low,
            high=args.high,
            min_std=args.min_std,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")

    summary = summarize_pass_rates(
        rows,
        low=args.low,
        high=args.high,
        min_std=args.min_std,
        reward_version=args.reward_version,
        sampling={
            "k": args.k,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "smoke": bool(args.smoke or (not args.model and not args.cached)),
            "cached": args.cached,
            "model": args.model,
        },
    )
    sum_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    sum_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for env_name, stats in summary["per_env"].items():
        print(
            f"[pass_rate_filter] env={env_name} kept={stats['n_kept']}/{stats['n_total']} "
            f"({stats['kept_fraction']:.3f}) nonzero_std={stats['nonzero_std_fraction']:.3f}"
        )
    return {"rows": rows, "summary": summary, "out": str(out_path), "summary_out": str(sum_path)}


if __name__ == "__main__":
    main()
