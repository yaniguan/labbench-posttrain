"""Unified `labrl` CLI entry point (`pip install -e .` -> `labrl ...`).

Subcommands:
  - `labrl list`: inspect registered corpora, verifiable environments, stages, engines, and 4-axis recipes.
  - `labrl pipeline`: run an end-to-end SFT -> DPO/MPO/TROPD/PPO/GRPO -> Eval pipeline with SplitGuard + KL anchor.
  - `labrl recipe`: compose, validate, export, or run a 4-axis (`algo x domain x model x hardware`) recipe.
  - `labrl validate`: strictly validate a NeMo-RL YAML configuration file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

import labrl  # triggers built-in corpus/env/stage registrations
from labpt_nemo_rl.config_validation import (
    RECIPES_ROOT,
    compose_recipe,
    load_yaml,
    validate_nemo_rl_yaml,
)
from labpt_nemo_rl.interfaces import NEMO_RL_AVAILABLE
from labrl.core.registry import (
    CORPUS_REGISTRY,
    ENV_REGISTRY,
    REWARD_TEMPLATE_REGISTRY,
    STAGE_REGISTRY,
)
from labrl.pipeline import PostTrainPipeline
from labrl.rewards import list_reward_templates, make_reward_template
from posttrain.hacking_checks import canary_in_files


def _discover_recipe_axes(root: Path = RECIPES_ROOT) -> dict[str, list[str]]:
    axes: dict[str, list[str]] = {}
    for axis in ("algo", "domain", "model", "hardware"):
        axis_dir = root / axis
        if axis_dir.exists():
            axes[axis] = sorted(p.stem for p in axis_dir.glob("*.yaml"))
        else:
            axes[axis] = []
    return axes


def _cmd_list(_args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "version": labrl.__version__,
        "corpora": sorted(CORPUS_REGISTRY.keys()),
        "environments": sorted(ENV_REGISTRY.keys()),
        "reward_templates": sorted(REWARD_TEMPLATE_REGISTRY.keys()),
        "stages": sorted(STAGE_REGISTRY.keys()),
        "engines": ["auto", "torch_local", "nemo"],
        "nemo_rl_available": NEMO_RL_AVAILABLE,
        "recipe_axes": _discover_recipe_axes(),
    }
    print(json.dumps(payload, indent=2))
    return payload


def _cmd_templates(args: argparse.Namespace) -> dict[str, Any]:
    if args.template_id:
        tmpl = make_reward_template(args.template_id)
        payload: dict[str, Any] = {
            **tmpl.describe(),
            "prompt_contract": tmpl.format_prompt_contract(),
        }
    else:
        payload = {"templates": list_reward_templates()}
    print(json.dumps(payload, indent=2))
    return payload


def _cmd_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    pipe = PostTrainPipeline.from_methods(
        methods,
        corpus=args.corpus,
        env=args.env,
        base_model=args.base_model,
        engine=args.engine,
        smoke=args.smoke,
        include_eval=not args.no_eval,
        seed=args.seed,
    )
    manifest = pipe.run(output_dir=args.output_dir)
    payload = manifest.to_dict()
    print(json.dumps(payload, indent=2))
    return payload


def _run_recipe_torch_local(cfg: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    algo = cfg["recipe_meta"]["algo"]
    domain = cfg["recipe_meta"]["domain"]
    loss_cfg = cfg.get("loss_fn", {}) or {}
    grpo_cfg = cfg.get("grpo", {}) or {}

    if algo == "mpo_offline":
        from posttrain.mpo import TOY_MPO_BATCH, train_mpo
        from posttrain.tiny import build_tiny_lm

        mpo_cfg = cfg.get("mpo", {}) or {}
        if domain == "calisol_conductivity":
            records = labrl.make_corpus("calisol23").to_mpo(limit=4 if smoke else 64)
            tiny_m, tiny_tok = build_tiny_lm(seed=0, n_positions=1024)
            return train_mpo(
                records,
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

    from labpt_nemo_rl.envs.bioprobench_env import BioProBenchEnv
    from labpt_nemo_rl.envs.calisol_env import CALiSolEnv
    from posttrain.grpo import TOY_PROMPTS, train_grpo
    from posttrain.rewards import toy_protocol_reward
    from posttrain.tiny import build_tiny_lm

    extra_model_kw: dict[str, Any] = {}
    if domain in ("bioenvs_pqa_ord_err", "bioenvs_gen"):
        bio_cfg = (cfg.get("env") or {}).get("bioprobench", {})
        env = BioProBenchEnv(bio_cfg, max_items_per_task=4 if smoke else 32)
        rows = env.build_dataset(n_per_task=2 if smoke else 16, include_counterfactuals=False)
        prompts = rows[:2] if smoke else rows
        reward_fn = env.posttrain_reward_fn()
    elif domain == "calisol_conductivity":
        cal_cfg = (cfg.get("env") or {}).get("calisol", {})
        env = CALiSolEnv(cal_cfg, max_items_per_task=4 if smoke else 32)
        rows = env.build_dataset(n_per_task=2 if smoke else 16, include_counterfactuals=False)
        prompts = rows[:2] if smoke else rows
        reward_fn = env.posttrain_reward_fn()
        if smoke:
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


def _cmd_recipe(args: argparse.Namespace) -> dict[str, Any]:
    cfg = compose_recipe(
        algo=args.algo,
        domain=args.domain,
        model=args.model,
        hardware=args.hardware,
        overrides=args.overrides,
    )

    repo_root = RECIPES_ROOT.parents[1]
    train_files: list[str] = []
    for sub in (cfg.get("env") or {}).values():
        if isinstance(sub, dict) and sub.get("train_file"):
            p = repo_root / str(sub["train_file"])
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
        backend = "torch_local" if (args.smoke or not NEMO_RL_AVAILABLE) else "nemo"

    if backend in ("torch_local", "posttrain"):
        res = _run_recipe_torch_local(cfg, smoke=args.smoke)
        summary = {
            "backend": "torch_local",
            "recipe": cfg["recipe_meta"],
            "status": res.get("status", "ran"),
            "final_loss": float(res.get("final_loss", 0.0)),
            "dump_yaml": args.dump_yaml,
        }
        print(json.dumps(summary, indent=2))
        return summary

    summary = {
        "backend": "nemo",
        "recipe": cfg["recipe_meta"],
        "nemo_rl_available": NEMO_RL_AVAILABLE,
        "status": "validated_ready",
        "dump_yaml": args.dump_yaml,
    }
    print(json.dumps(summary, indent=2))
    return summary


def _cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.yaml_path)
    cfg = load_yaml(path)
    validate_nemo_rl_yaml(cfg, source=str(path))
    report = canary_in_files([str(path)])
    if report["triggered"]:
        raise RuntimeError(f"Canary detected in YAML config {path}: {report['detail']}")
    payload = {
        "valid": True,
        "path": str(path),
        "top_level_keys": sorted(cfg.keys()),
    }
    print(json.dumps(payload, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labrl",
        description="labrl: End-to-End SFT-to-RLVR Post-Training SDK & CLI for Scientific Observables",
    )
    parser.add_argument("--version", action="version", version=f"labrl {labrl.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List registered corpora, environments, reward templates, stages, and recipe axes")
    p_list.set_defaults(_handler=_cmd_list)

    p_tmpl = sub.add_parser("templates", help="Inspect verified reward templates, schemas, GDPO channels, and anti-hack guarantees")
    p_tmpl.add_argument("--id", dest="template_id", default=None, help="Specific template_id to inspect")
    p_tmpl.set_defaults(_handler=_cmd_templates)

    p_pipe = sub.add_parser("pipeline", help="Run an SFT-anchored post-training pipeline")
    p_pipe.add_argument("--corpus", default="calisol23", help="Registered corpus name (default: calisol23)")
    p_pipe.add_argument("--env", default=None, help="Optional verifiable environment override")
    p_pipe.add_argument(
        "--methods",
        default="sft,dpo,mpo,ppo,grpo",
        help="Comma-separated methods to run from the SFT anchor (sft,dpo,mpo,tropd,ppo,grpo)",
    )
    p_pipe.add_argument("--base-model", default="Qwen/Qwen3-1.7B", help="Base HF model ID (or 'tiny' for smoke)")
    p_pipe.add_argument("--engine", choices=("auto", "torch_local", "nemo"), default="auto")
    p_pipe.add_argument("--smoke", action="store_true", help="Fast zero-GPU tiny-LM verification run")
    p_pipe.add_argument("--no-eval", action="store_true", help="Skip held-out test split evaluation")
    p_pipe.add_argument("--output-dir", default=None, help="Directory for checkpoints, NeMo configs, and lineage_manifest.json")
    p_pipe.add_argument("--seed", type=int, default=0)
    p_pipe.set_defaults(_handler=_cmd_pipeline)

    p_rec = sub.add_parser("recipe", help="Compose, validate, export, or run a 4-axis recipe")
    p_rec.add_argument("--algo", required=True, help="Algorithm axis under configs/recipes/algo/")
    p_rec.add_argument("--domain", required=True, help="Domain axis under configs/recipes/domain/")
    p_rec.add_argument("--model", required=True, help="Model axis under configs/recipes/model/")
    p_rec.add_argument("--hardware", required=True, help="Hardware axis under configs/recipes/hardware/")
    p_rec.add_argument("--backend", choices=("auto", "torch_local", "posttrain", "nemo"), default="auto")
    p_rec.add_argument("--smoke", action="store_true", help="Run fast tiny-LM smoke execution")
    p_rec.add_argument("--dump-yaml", default=None, help="Write composed NeMo-RL launch YAML to path")
    p_rec.add_argument("--set", dest="overrides", action="append", default=[], help="Dot-path override key=value")
    p_rec.set_defaults(_handler=_cmd_recipe)

    p_val = sub.add_parser("validate", help="Validate a NeMo-RL YAML configuration file")
    p_val.add_argument("yaml_path", help="Path to YAML config file")
    p_val.set_defaults(_handler=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._handler(args)


def cli_entry() -> None:
    """Console-script entry point returning None (exit code 0) rather than dict."""
    main()


if __name__ == "__main__":
    cli_entry()
