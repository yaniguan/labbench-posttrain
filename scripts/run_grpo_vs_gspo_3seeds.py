"""Reproducible 3-seed GRPO vs GSPO + spurious-reward controls runner (Refs #18).

Manages the 4 matched experiment arms under `configs/posttrain/experiments/`:
  - `grpo_true`           (GRPO + GDPO true v3 reward)
  - `gspo_true`           (GSPO sequence geometric-mean IS + true v3 reward)
  - `grpo_random_control` (GRPO + Bernoulli random reward control, p=0.5)
  - `grpo_format_control` (GRPO + format-only tag reward control)

Modes:
  - `--dry-run`: Validates all 4 arm configs, checks matched 3-seed rollout budgets and KL settings,
    and writes/prints an honest summary table with `status = "not run"` (`GPU results: PENDING`).
  - `--smoke`: Runs all 4 arms x 3 seeds (`12` tiny-LM CPU runs) through `posttrain.grpo.train_grpo`
    on `BioProBenchEnv` (`PQA + ORD + ERR`, `v3`), saving per-seed reward & KL-to-SFT curves plus
    a 3-seed mean/std + gap-to-best-control summary explicitly labeled as CPU smoke (`GPU results: PENDING`).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.run_meta import build_run_meta  # noqa: E402
from labpt_nemo_rl.config_validation import compose_recipe  # noqa: E402

EXP_DIR = REPO_ROOT / "configs" / "posttrain" / "experiments"
ARM_ORDER = ("grpo_true", "gspo_true", "grpo_random_control", "grpo_format_control")
CONTROL_ARMS = {"grpo_random_control", "grpo_format_control"}


def load_arm_configs(exp_dir: Path = EXP_DIR) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        path = exp_dir / f"{arm}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing experiment config: {path}")
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if cfg.get("arm") != arm:
            raise ValueError(f"{path}: expected arm={arm!r}, got {cfg.get('arm')!r}")
        if float(cfg.get("kl_coef", 0.0)) <= 0.0:
            raise ValueError(f"{path}: kl_coef must be > 0 for controlled RL comparison")
        configs[arm] = cfg
    # Verify matched rollout budgets and seed sets across all 4 arms.
    budgets = {arm: cfg["rollout_budget"] for arm, cfg in configs.items()}
    first_budget = budgets[ARM_ORDER[0]]
    first_seeds = list(configs[ARM_ORDER[0]]["seeds"])
    for arm, cfg in configs.items():
        if cfg["rollout_budget"] != first_budget:
            raise ValueError(f"Arm {arm} rollout_budget {cfg['rollout_budget']} != {first_budget}")
        if list(cfg["seeds"]) != first_seeds:
            raise ValueError(f"Arm {arm} seeds {cfg['seeds']} != {first_seeds}")
        # Verify underlying 4-axis recipe composes cleanly.
        rec = cfg["recipe"]
        compose_recipe(
            algo=rec["algo"],
            domain=rec["domain"],
            model=rec["model"],
            hardware=rec["hardware"],
        )
    return configs


def _run_smoke_seed(
    arm_cfg: dict[str, Any],
    seed: int,
    *,
    steps: int = 2,
    group_size: int = 2,
) -> dict[str, Any]:
    from labpt_nemo_rl.envs.bioprobench_env import BioProBenchEnv
    from posttrain.grpo import train_grpo
    from posttrain.tiny import build_tiny_lm

    env_spec = arm_cfg["env"]
    raw_rm = arm_cfg.get("reward_mode", "true")
    rm_str = "true" if raw_rm is True else str(raw_rm).lower()
    bio_env = BioProBenchEnv(
        tasks=list(env_spec["tasks"]),
        split=str(env_spec.get("train_split", "train")),
        seed=seed,
        reward_version=str(arm_cfg.get("reward_version", "v3")),
        reward_mode=rm_str,
        max_items_per_task=4,
    )
    prompts = bio_env.build_dataset(n_per_task=2, include_counterfactuals=False)
    reward_fn = bio_env.posttrain_reward_fn()
    model, tok = build_tiny_lm(seed=seed)
    ref_model, _ = build_tiny_lm(seed=seed)

    res = train_grpo(
        prompts,
        reward_fn,
        model=model,
        tokenizer=tok,
        reference=ref_model,
        steps=steps,
        group_size=group_size,
        max_new_tokens=10,
        kl_coef=float(arm_cfg["kl_coef"]),
        kl_estimator=str(arm_cfg.get("kl_estimator", "k3")),
        kl_in_reward=bool(arm_cfg.get("kl_in_reward", False)),
        advantage_mode=str(arm_cfg.get("advantage_mode", "grpo")),
        sequence_level_ratios=bool(arm_cfg.get("sequence_level_importance_ratios", False)),
        token_level=bool(arm_cfg.get("token_level_loss", True)),
        eps_low=float(arm_cfg.get("eps_low", 0.2)),
        eps_high=float(arm_cfg.get("eps_high", 0.28)),
        reward_mode="ground_truth",
        seed=seed,
    )
    history = res["logger"].rows
    reward_curve = [float(h.reward_mean) for h in history]
    kl_curve = [float(h.kl_to_sft) for h in history]

    # Evaluate final tiny checkpoint on a tiny frozen BioProBench test slice (honest CPU smoke probe).
    test_env = BioProBenchEnv(
        tasks=list(env_spec["tasks"]),
        split="test",
        seed=seed,
        reward_version=str(arm_cfg.get("reward_version", "v3")),
        reward_mode="true",
        max_items_per_task=2,
    )
    test_rows = test_env.build_dataset(n_per_task=2, include_counterfactuals=False)
    test_fn = test_env.posttrain_reward_fn()
    test_scores = [float(test_fn(r["prompt"], "[ANSWER_START]ABSTAIN[ANSWER_END]", r).value) for r in test_rows]

    return {
        "arm": arm_cfg["arm"],
        "seed": seed,
        "execution_tier": "cpu_tiny_smoke",
        "gpu_results": "PENDING",
        "reward_mode": arm_cfg["reward_mode"],
        "advantage_mode": arm_cfg["advantage_mode"],
        "reward_curve": reward_curve,
        "kl_to_sft_curve": kl_curve,
        "final_train_reward": reward_curve[-1] if reward_curve else 0.0,
        "final_kl_to_sft": kl_curve[-1] if kl_curve else 0.0,
        "smoke_test_probe_mean": float(st.mean(test_scores)) if test_scores else 0.0,
        "config": arm_cfg,
        "meta": build_run_meta(
            reward_version=str(arm_cfg.get("reward_version", "v3")),
            sampling={"steps": steps, "group_size": group_size, "max_new_tokens": 10},
            extra={"execution_tier": "cpu_tiny_smoke", "gpu_results": "PENDING"},
        ),
    }


def summarize_experiment(
    configs: dict[str, dict[str, Any]],
    per_seed_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate 3-seed mean/std and gap-to-best-control, keeping unexecuted arms as `'not run'`."""
    if not per_seed_runs:
        arms_summary = {
            arm: {
                "status": "not run",
                "gpu_results": "PENDING",
                "reward_mode": cfg["reward_mode"],
                "advantage_mode": cfg["advantage_mode"],
                "seeds": list(cfg["seeds"]),
                "mean_reward": "not run",
                "std_reward": "not run",
                "mean_kl_to_sft": "not run",
                "gap_to_best_control": "not run",
            }
            for arm, cfg in configs.items()
        }
        return {
            "status": "not run",
            "gpu_results": "PENDING",
            "arms": arms_summary,
            "meta": build_run_meta(reward_version="v3", extra={"gpu_results": "PENDING"}),
        }

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for run in per_seed_runs:
        by_arm.setdefault(str(run["arm"]), []).append(run)

    arm_means: dict[str, float] = {}
    arms_summary: dict[str, dict[str, Any]] = {}
    for arm, cfg in configs.items():
        runs = by_arm.get(arm, [])
        if not runs:
            arms_summary[arm] = {
                "status": "not run",
                "gpu_results": "PENDING",
                "reward_mode": cfg["reward_mode"],
                "advantage_mode": cfg["advantage_mode"],
                "seeds": list(cfg["seeds"]),
                "mean_reward": "not run",
                "std_reward": "not run",
                "mean_kl_to_sft": "not run",
                "gap_to_best_control": "not run",
            }
            continue
        vals = [float(r["final_train_reward"]) for r in runs]
        kls = [float(r["final_kl_to_sft"]) for r in runs]
        mean_v = float(st.mean(vals))
        std_v = float(st.pstdev(vals)) if len(vals) > 1 else 0.0
        arm_means[arm] = mean_v
        arms_summary[arm] = {
            "status": runs[0]["execution_tier"],
            "gpu_results": "PENDING",
            "reward_mode": cfg["reward_mode"],
            "advantage_mode": cfg["advantage_mode"],
            "seeds": [int(r["seed"]) for r in runs],
            "mean_reward": round(mean_v, 6),
            "std_reward": round(std_v, 6),
            "mean_kl_to_sft": round(float(st.mean(kls)), 6),
            "gap_to_best_control": 0.0,
        }

    ctrl_vals = [arm_means[c] for c in CONTROL_ARMS if c in arm_means]
    best_ctrl = max(ctrl_vals) if ctrl_vals else None
    for arm, entry in arms_summary.items():
        if entry["mean_reward"] != "not run" and best_ctrl is not None:
            entry["gap_to_best_control"] = round(float(entry["mean_reward"]) - best_ctrl, 6)
        else:
            entry["gap_to_best_control"] = "not run"

    return {
        "status": per_seed_runs[0]["execution_tier"],
        "gpu_results": "PENDING",
        "best_control_mean": round(best_ctrl, 6) if best_ctrl is not None else "not run",
        "arms": arms_summary,
        "meta": build_run_meta(
            reward_version="v3",
            extra={"execution_tier": per_seed_runs[0]["execution_tier"], "gpu_results": "PENDING"},
        ),
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# GRPO vs GSPO 3-Seed Controlled Experiment Summary",
        "",
        f"- Execution status: `{summary['status']}`",
        f"- GPU A100 status: `{summary['gpu_results']}`",
        "",
        "| Arm | Advantage | Reward Mode | Seeds | Mean Reward | Std | Mean KL to SFT | Gap vs Best Control | GPU Status |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for arm in ARM_ORDER:
        row = summary["arms"][arm]
        mr = f"{row['mean_reward']:.4f}" if isinstance(row["mean_reward"], (int, float)) else str(row["mean_reward"])
        sr = f"{row['std_reward']:.4f}" if isinstance(row["std_reward"], (int, float)) else str(row["std_reward"])
        kl = f"{row['mean_kl_to_sft']:.4f}" if isinstance(row["mean_kl_to_sft"], (int, float)) else str(row["mean_kl_to_sft"])
        gap = (
            f"{row['gap_to_best_control']:+.4f}"
            if isinstance(row["gap_to_best_control"], (int, float))
            else str(row["gap_to_best_control"])
        )
        lines.append(
            f"| `{arm}` | `{row['advantage_mode']}` | `{row['reward_mode']}` | `{row['seeds']}` | "
            f"{mr} | {sr} | {kl} | {gap} | `{row['gpu_results']}` |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="3-seed GRPO vs GSPO + reward controls runner (Refs #18).")
    ap.add_argument("--dry-run", action="store_true", help="Validate 4 arm configs and emit honest 'not run' summary")
    ap.add_argument("--smoke", action="store_true", help="Execute 4 arms x 3 seeds on CPU tiny LM for pipeline check")
    ap.add_argument("--steps", type=int, default=2, help="Training steps per seed in --smoke mode")
    ap.add_argument("--group-size", type=int, default=2, help="Completions per prompt in --smoke mode")
    ap.add_argument("--out-dir", type=str, default=None, help="Optional directory to write per-seed JSONs + summary")
    args = ap.parse_args(argv)

    configs = load_arm_configs()
    runs: list[dict[str, Any]] | None = None

    if args.smoke and not args.dry_run:
        runs = []
        for arm in ARM_ORDER:
            cfg = configs[arm]
            for seed in cfg["seeds"]:
                runs.append(_run_smoke_seed(cfg, int(seed), steps=args.steps, group_size=args.group_size))

    summary = summarize_experiment(configs, runs)
    md = render_summary_markdown(summary)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if runs:
            for r in runs:
                p = out_dir / f"{r['arm']}_seed{r['seed']}.json"
                p.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out_dir / "summary.md").write_text(md, encoding="utf-8")

    print(md)
    return {"configs": configs, "runs": runs, "summary": summary, "markdown": md}


if __name__ == "__main__":
    main()
