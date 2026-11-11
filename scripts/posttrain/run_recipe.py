"""Unified 4-axis recipe runner (`algo` x `domain` x `model` x `hardware`).

Examples:
  # Compose, validate, and run a local/CI smoke test on posttrain/:
  python scripts/posttrain/run_recipe.py --algo gdpo_multireward --domain bioenvs_pqa_ord_err \
      --model qwen3_1.7b --hardware local_1gpu_colocated --smoke

  # Export a validated NeMo-RL launch YAML for 8xA100/H100:
  python scripts/posttrain/run_recipe.py --algo gdpo_multireward --domain bioenvs_pqa_ord_err \
      --model qwen3_8b --hardware node_8xa100_colocated --dump-yaml outputs/recipes/launch.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from labpt_nemo_rl.config_validation import compose_recipe  # noqa: E402
from labpt_nemo_rl.envs.bioprobench_env import BioProBenchEnv  # noqa: E402
from labpt_nemo_rl.interfaces import NEMO_RL_AVAILABLE  # noqa: E402
from posttrain.hacking_checks import canary_in_files  # noqa: E402


def run_posttrain_backend(cfg: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    algo = cfg["recipe_meta"]["algo"]
    domain = cfg["recipe_meta"]["domain"]
    loss_cfg = cfg.get("loss_fn", {}) or {}
    grpo_cfg = cfg.get("grpo", {}) or {}

    if algo == "mpo_offline":
        from posttrain.mpo import TOY_MPO_BATCH, train_mpo

        mpo_cfg = cfg.get("mpo", {}) or {}
        if domain == "calisol_conductivity":
            from scripts.posttrain.calisol_pipeline import build_calisol_mpo_batch
            from posttrain.tiny import build_tiny_lm

            mpo_records = build_calisol_mpo_batch(max_pairs=2 if smoke else 64, max_unpaired=2 if smoke else 64)
            tiny_m, tiny_tok = build_tiny_lm(seed=0, n_positions=1024)
            return train_mpo(
                mpo_records,
                beta=float(mpo_cfg.get("reference_policy_kl_penalty", 0.1)),
                preference_weight=float(mpo_cfg.get("preference_loss_weight", 1.0)),
                bco_weight=float(mpo_cfg.get("bco_loss_weight", 0.8)),
                sft_weight=float(mpo_cfg.get("sft_loss_weight", 0.2)),
                reward_shift_momentum=float(mpo_cfg.get("reward_shift_momentum", 0.95)),
                max_length=1024,
                model=tiny_m,
                tokenizer=tiny_tok,
                steps=2 if smoke else int(mpo_cfg.get("max_num_steps", 10)),
            )
        return train_mpo(
            TOY_MPO_BATCH,
            beta=float(mpo_cfg.get("reference_policy_kl_penalty", 0.1)),
            preference_weight=float(mpo_cfg.get("preference_loss_weight", 1.0)),
            bco_weight=float(mpo_cfg.get("bco_loss_weight", 0.8)),
            sft_weight=float(mpo_cfg.get("sft_loss_weight", 0.2)),
            reward_shift_momentum=float(mpo_cfg.get("reward_shift_momentum", 0.95)),
            steps=2 if smoke else int(mpo_cfg.get("max_num_steps", 10)),
        )

    if algo == "tropd_distill":
        from posttrain.distill_opd import train_tropd
        from posttrain.grpo import TOY_PROMPTS
        from posttrain.rewards import toy_protocol_reward

        opd_cfg = cfg.get("on_policy_distillation", {}) or {}
        return train_tropd(
            TOY_PROMPTS,
            toy_protocol_reward,
            proximal_alpha=float(opd_cfg.get("proximal_teacher_alpha", 0.5)),
            task_reward_weight=float(opd_cfg.get("task_reward_weight", 0.4)),
            distill_weight=float(opd_cfg.get("distill_weight", 0.6)),
            steps=2 if smoke else 10,
        )

    from posttrain.grpo import TOY_PROMPTS, train_grpo
    from posttrain.rewards import toy_protocol_reward

    extra_model_kw: dict[str, Any] = {}
    if domain in ("bioenvs_pqa_ord_err", "bioenvs_gen"):
        bio_cfg = (cfg.get("env") or {}).get("bioprobench", {})
        env = BioProBenchEnv(bio_cfg, max_items_per_task=4 if smoke else 32)
        rows = env.build_dataset(n_per_task=2 if smoke else 16, include_counterfactuals=False)
        prompts = rows[:2] if smoke else rows
        reward_fn = env.posttrain_reward_fn()
    elif domain == "calisol_conductivity":
        from labpt_nemo_rl.envs.calisol_env import CALiSolEnv
        cal_cfg = (cfg.get("env") or {}).get("calisol", {})
        env = CALiSolEnv(cal_cfg, max_items_per_task=4 if smoke else 32)
        rows = env.build_dataset(n_per_task=2 if smoke else 16, include_counterfactuals=False)
        prompts = rows[:2] if smoke else rows
        reward_fn = env.posttrain_reward_fn()
        if smoke:
            from posttrain.tiny import build_tiny_lm

            m_tiny, tok_tiny = build_tiny_lm(seed=0, n_positions=1024)
            extra_model_kw = {"model": m_tiny, "tokenizer": tok_tiny}
    else:
        prompts = TOY_PROMPTS
        reward_fn = toy_protocol_reward

    adv_est = (grpo_cfg.get("adv_estimator") or {}).get("name", "grpo")
    gdpo_w = (grpo_cfg.get("adv_estimator") or {}).get("gdpo_channel_weights")
    shaping = grpo_cfg.get("reward_shaping") or {}
    len_pen = grpo_cfg.get("length_penalty") or {}
    seq_ratios = bool(loss_cfg.get("sequence_level_importance_ratios", False))

    return train_grpo(
        prompts,
        reward_fn,
        kl_coef=float(loss_cfg.get("reference_policy_kl_penalty", 0.01)),
        kl_estimator=str(loss_cfg.get("reference_policy_kl_type", "k3")),
        eps_low=float(loss_cfg.get("ratio_clip_min", 0.2)),
        eps_high=float(loss_cfg.get("ratio_clip_max", 0.28)),
        dual_clip_c=loss_cfg.get("ratio_clip_c"),
        sequence_level_ratios=seq_ratios,
        token_level=bool(loss_cfg.get("token_level_loss", not seq_ratios)),
        baseline="loo" if (grpo_cfg.get("adv_estimator") or {}).get("use_leave_one_out_baseline") else "mean",
        advantage_mode="gdpo" if adv_est == "gdpo" and (domain.startswith("bioenvs") or domain == "calisol_conductivity") else "grpo",
        gdpo_weights=gdpo_w,
        dynamic_sampling=bool(grpo_cfg.get("use_dynamic_sampling", False)),
        dynamic_sampling_max_rounds=2 if smoke else int(grpo_cfg.get("dynamic_sampling_max_gen_batches", 3)),
        overlong_filtering=bool(grpo_cfg.get("overlong_filtering", False)),
        stop_properly_coef=shaping.get("stop_properly_penalty_coef"),
        reasoning_length_coef=float(len_pen.get("coef", 0.0)) if len_pen.get("enabled") else 0.0,
        overlong_buffer_length=4 if smoke else shaping.get("overlong_buffer_length"),
        overlong_buffer_penalty=float(shaping.get("overlong_buffer_penalty", 0.0)) if shaping.get("enabled") else 0.0,
        use_is_correction=bool(loss_cfg.get("use_importance_sampling_correction", False)),
        is_mode=str(loss_cfg.get("truncated_importance_sampling_type", "icepop")),
        is_clip_min=float(loss_cfg.get("truncated_importance_sampling_ratio_min", 0.5)),
        is_clip=float(loss_cfg.get("truncated_importance_sampling_ratio", 2.0)),
        group_size=4 if smoke else int(grpo_cfg.get("num_generations_per_prompt", 8)),
        max_new_tokens=8 if smoke else 256,
        steps=1 if smoke else 10,
        **extra_model_kw,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Compose, validate, and run 4-axis RL/Post-Training recipes.")
    parser.add_argument("--algo", required=True, help="Algorithm axis under configs/recipes/algo/")
    parser.add_argument("--domain", required=True, help="Domain axis under configs/recipes/domain/")
    parser.add_argument("--model", required=True, help="Model axis under configs/recipes/model/")
    parser.add_argument("--hardware", required=True, help="Hardware axis under configs/recipes/hardware/")
    parser.add_argument("--backend", choices=("auto", "posttrain", "nemo_rl"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run fast tiny-LM smoke execution")
    parser.add_argument("--dump-yaml", type=str, default=None, help="Write composed YAML config to path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Dot-path override key=value")
    args = parser.parse_args(argv)

    cfg = compose_recipe(
        algo=args.algo,
        domain=args.domain,
        model=args.model,
        hardware=args.hardware,
        overrides=args.overrides,
    )

    # Canary check on any training data file referenced by the domain axis
    train_files = []
    for sub in (cfg.get("env") or {}).values():
        if isinstance(sub, dict) and sub.get("train_file"):
            p = REPO_ROOT / str(sub["train_file"])
            if p.exists():
                train_files.append(str(p))
    if train_files:
        report = canary_in_files(train_files)
        if report["triggered"]:
            raise RuntimeError(f"Canary detected in training file: {report['detail']}")

    if args.dump_yaml:
        out_path = Path(args.dump_yaml)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_cfg = {k: v for k, v in cfg.items() if k != "recipe_meta"}
        out_path.write_text(yaml.safe_dump(export_cfg, sort_keys=False), encoding="utf-8")

    backend = args.backend
    if backend == "auto":
        backend = "posttrain" if (args.smoke or not NEMO_RL_AVAILABLE) else "nemo_rl"

    if backend == "posttrain":
        res = run_posttrain_backend(cfg, smoke=args.smoke)
        summary = {"backend": "posttrain", "status": res.get("status", "ran"), "final_loss": res.get("final_loss", 0.0)}
        print(json.dumps(summary))
        return {"config": cfg, "result": res, "backend": "posttrain"}

    summary = {
        "backend": "nemo_rl",
        "nemo_rl_available": NEMO_RL_AVAILABLE,
        "status": "validated_ready",
        "dump_yaml": args.dump_yaml,
    }
    print(json.dumps(summary))
    return {"config": cfg, "result": summary, "backend": "nemo_rl"}


if __name__ == "__main__":
    main()
