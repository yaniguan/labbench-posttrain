"""NeMo-RL (`yaniguan/RL`) launcher for BioProBench & CALiSol-23 GRPO / GSPO / GDPO recipes (#16).

Composes a 4-axis recipe (`algo` x `domain` x `model` x `hardware`), registers
`labbench_bioprobench` and `labbench_calisol` into `nemo_rl.environments.utils.ENV_REGISTRY`
(or verifies registration locally when `--dry-run` is passed), builds the prompt dataset and
first mini-batch, and launches `nemo_rl.algorithms.grpo.grpo_train` on Ray.

Example:
    python scripts/posttrain/run_nemo_rl.py --algo gspo_longcot --domain bioenvs_pqa_ord_err \
        --model qwen3_1.7b --hardware local_1gpu_colocated --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from labpt_nemo_rl.config_validation import compose_recipe, validate_nemo_rl_yaml  # noqa: E402
from labpt_nemo_rl.envs.bioprobench_env import BioProBenchEnv  # noqa: E402
from labpt_nemo_rl.envs.calisol_env import CALiSolEnv  # noqa: E402
from labpt_nemo_rl.interfaces import NEMO_RL_AVAILABLE  # noqa: E402

PINNED_RL_COMMIT = "a7299bc888e4b001020493d976f722513e57e68c"

ENV_ACTOR_FQNS: dict[str, str] = {
    "labbench_bioprobench": "labpt_nemo_rl.envs.bioprobench_env.BioProBenchEnv",
    "labbench_calisol": "labpt_nemo_rl.envs.calisol_env.CALiSolEnv",
}

_LOCAL_ENV_REGISTRY: dict[str, dict[str, str]] = {}


def register_labbench_envs() -> dict[str, Any]:
    """Register `labbench_bioprobench` and `labbench_calisol` in `nemo_rl` (or local fallback)."""
    if NEMO_RL_AVAILABLE:
        from nemo_rl.environments.utils import ENV_REGISTRY as NEMO_ENV_REGISTRY  # type: ignore
        from nemo_rl.environments.utils import register_env as nemo_register_env  # type: ignore

        for env_name, fqn in ENV_ACTOR_FQNS.items():
            if env_name not in NEMO_ENV_REGISTRY:
                nemo_register_env(env_name, fqn)
        return {"backend": "nemo_rl", "registered": list(ENV_ACTOR_FQNS.keys())}

    for env_name, fqn in ENV_ACTOR_FQNS.items():
        _LOCAL_ENV_REGISTRY[env_name] = {"actor_class_fqn": fqn}
    return {"backend": "local_shim", "registered": list(ENV_ACTOR_FQNS.keys())}


def build_domain_env_and_dataset(
    cfg: dict[str, Any],
    *,
    n_per_task: int = 8,
    val_n_per_task: int = 4,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    """Instantiate the domain environment actor and build train + val prompt datasets."""
    domain = (cfg.get("recipe_meta") or {}).get("domain", "bioenvs_pqa_ord_err")
    env_block = cfg.get("env") or {}
    if domain == "calisol_conductivity" or "calisol" in env_block:
        cal_cfg = dict(env_block.get("calisol") or {})
        train_env = CALiSolEnv(cal_cfg, split=str(cal_cfg.get("split", "train")), max_items_per_task=max(4, n_per_task))
        val_env = CALiSolEnv(cal_cfg, split="val", max_items_per_task=max(4, val_n_per_task))
        return (
            train_env,
            train_env.build_dataset(n_per_task=n_per_task, include_counterfactuals=False),
            val_env.build_dataset(n_per_task=val_n_per_task, include_counterfactuals=False),
        )

    bio_cfg = dict(env_block.get("bioprobench") or {})
    filter_path = bio_cfg.pop("pass_rate_filter_path", None)
    val_split = str(bio_cfg.pop("val_split", "val"))
    train_env = BioProBenchEnv(bio_cfg, split=str(bio_cfg.get("split", "train")), max_items_per_task=max(4, n_per_task))
    if filter_path:
        from scripts.posttrain.pass_rate_filter import load_kept_ids, make_pass_rate_filter_fn

        kept_ids = load_kept_ids(filter_path)
        fn = make_pass_rate_filter_fn(kept_ids)
        for task_name, sub_env in train_env.envs.items():
            sub_env.rows = [r for r in sub_env.rows if fn(r)]
            sub_env._index = {r.get("id"): idx for idx, r in enumerate(sub_env.rows)}
    val_env = BioProBenchEnv(bio_cfg, split=val_split, max_items_per_task=max(4, val_n_per_task))
    inc_cf = bool(bio_cfg.get("include_counterfactuals", False))
    return (
        train_env,
        train_env.build_dataset(n_per_task=n_per_task, include_counterfactuals=inc_cf),
        val_env.build_dataset(n_per_task=val_n_per_task, include_counterfactuals=False),
    )


def resolve_nemo_entrypoint(algo: str | None, cfg: dict[str, Any]) -> dict[str, str]:
    """Resolve stage-specific NeMo-RL (`yaniguan/RL`) driver module and train function (#49)."""
    algo_norm = (algo or "").strip().lower()
    if algo_norm == "sft" or ("sft" in cfg and "grpo" not in cfg):
        return {
            "stage": "sft",
            "module": "nemo_rl.algorithms.sft",
            "train_fn": "sft_train",
            "setup_fn": "setup",
        }
    if algo_norm == "mpo_offline" or ("mpo" in cfg and "grpo" not in cfg):
        return {
            "stage": "mpo",
            "module": "nemo_rl.algorithms.dpo",
            "train_fn": "dpo_train",
            "setup_fn": "setup",
            "loss_mode": "mpo",
        }
    if algo_norm == "dpo" or ("dpo" in cfg and "grpo" not in cfg):
        return {
            "stage": "dpo",
            "module": "nemo_rl.algorithms.dpo",
            "train_fn": "dpo_train",
            "setup_fn": "setup",
        }
    if algo_norm == "ppo" or ("ppo" in cfg and "grpo" not in cfg):
        return {
            "stage": "ppo",
            "module": "nemo_rl.algorithms.ppo",
            "train_fn": "ppo_train",
            "setup_fn": "setup",
        }
    return {
        "stage": "grpo",
        "module": "nemo_rl.algorithms.grpo",
        "train_fn": "grpo_train",
        "setup_fn": "setup",
    }


def launch_nemo_rl(
    export_cfg: dict[str, Any],
    yaml_path: Path,
    *,
    algo: str | None = None,
) -> dict[str, Any]:
    """Invoke the stage-specific `yaniguan/RL` driver (`sft`, `mpo`/`dpo`, `ppo`, or `grpo`, #49)."""
    import importlib

    from omegaconf import OmegaConf  # type: ignore
    from nemo_rl.algorithms.utils import get_tokenizer  # type: ignore
    from nemo_rl.distributed.virtual_cluster import init_ray  # type: ignore
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers  # type: ignore

    entry = resolve_nemo_entrypoint(algo, export_cfg)
    stage_mod = importlib.import_module(entry["module"])
    master_cls = getattr(stage_mod, "MasterConfig", dict)
    setup_fn = getattr(stage_mod, entry["setup_fn"])
    train_fn = getattr(stage_mod, entry["train_fn"])

    register_omegaconf_resolvers()
    raw_cfg = OmegaConf.to_container(load_config(str(yaml_path)), resolve=True)
    config = master_cls(**raw_cfg) if callable(master_cls) else raw_cfg
    init_ray()

    if entry["stage"] in ("sft", "dpo", "mpo"):
        policy_cfg = config["policy"] if isinstance(config, dict) else config.policy
        data_cfg = config["data"] if isinstance(config, dict) else config.data
        tokenizer = get_tokenizer(policy_cfg["tokenizer"])
        if entry["stage"] == "sft":
            from nemo_rl.data.utils import setup_sft_data  # type: ignore

            dataset, val_dataset = setup_sft_data(tokenizer, data_cfg)
        else:
            from nemo_rl.data.utils import setup_preference_data  # type: ignore

            dataset, val_dataset = setup_preference_data(tokenizer, data_cfg)
        setup_res = setup_fn(config, tokenizer, dataset, val_dataset)
        train_fn(*setup_res)
        return {
            "status": "ran",
            "backend": "nemo_rl",
            "stage": entry["stage"],
            "entrypoint": f"{entry['module']}.{entry['train_fn']}",
            "config_path": str(yaml_path),
        }

    from nemo_rl.data.utils import setup_response_data  # type: ignore
    from nemo_rl.data_plane.factory import make_policy_factory, maybe_configure_data_plane_env  # type: ignore
    from nemo_rl.environments.utils import shutdown_environments  # type: ignore
    from nemo_rl.models.generation import configure_generation_config  # type: ignore

    maybe_configure_data_plane_env(config.data_plane)
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(config.policy["generation"], tokenizer)
    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(tokenizer, config.data, config.env)
    try:
        (
            policy,
            policy_generation,
            _nemo_gym,
            cluster,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_state,
            master_config,
            teacher_worker_groups,
            alias_to_group_alias,
        ) = setup_fn(
            config,
            tokenizer,
            dataset,
            val_dataset,
            policy_factory=make_policy_factory(config.data_plane),
        )
        train_fn(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_state,
            master_config,
            task_to_env,
            val_task_to_env=val_task_to_env,
            teacher_worker_groups=teacher_worker_groups,
            alias_to_group_alias=alias_to_group_alias,
        )
    finally:
        shutdown_environments(task_to_env, val_task_to_env)
    return {
        "status": "ran",
        "backend": "nemo_rl",
        "stage": entry["stage"],
        "entrypoint": f"{entry['module']}.{entry['train_fn']}",
        "config_path": str(yaml_path),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Compose and launch SFT/MPO/GRPO/GSPO recipes on yaniguan/RL (#16, #49).")
    ap.add_argument("--algo", default="gspo_longcot", help="Algorithm axis under configs/recipes/algo/")
    ap.add_argument("--domain", default="bioenvs_pqa_ord_err", help="Domain axis under configs/recipes/domain/")
    ap.add_argument("--model", default="qwen3_1.7b", help="Model axis under configs/recipes/model/")
    ap.add_argument("--hardware", default="local_1gpu_colocated", help="Hardware axis under configs/recipes/hardware/")
    ap.add_argument("--init-checkpoint", default=None, help="Upstream actor checkpoint path (e.g. SFT or MPO)")
    ap.add_argument("--sft-anchor", default=None, help="Frozen Stage-1 SFT reference checkpoint path for k3 KL")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="Dot-path override key=value")
    ap.add_argument("--dump-yaml", type=str, default=None, help="Optional path to save composed NeMo-RL YAML")
    ap.add_argument("--dry-run", action="store_true", help="Register envs, build actor + first batch, and exit")
    args = ap.parse_args(argv)

    all_overrides = list(args.overrides or [])
    if args.init_checkpoint:
        all_overrides.append(f"policy.model_name={args.init_checkpoint}")
    if args.sft_anchor and args.algo != "sft":
        all_overrides.append(f"policy.reference_model_name={args.sft_anchor}")

    cfg = compose_recipe(
        algo=args.algo,
        domain=args.domain,
        model=args.model,
        hardware=args.hardware,
        overrides=all_overrides,
    )
    export_cfg = {k: v for k, v in cfg.items() if k != "recipe_meta"}
    validate_nemo_rl_yaml(export_cfg, source=f"{args.algo}:{args.domain}")
    entry = resolve_nemo_entrypoint(args.algo, export_cfg)

    reg_info = register_labbench_envs()
    env_actor, train_rows, val_rows = build_domain_env_and_dataset(
        cfg,
        n_per_task=4 if args.dry_run else 32,
        val_n_per_task=2 if args.dry_run else 16,
    )
    batch_sz = min(4, len(train_rows))
    first_batch = train_rows[:batch_sz]
    # Dry-run step verification on the first batch through the NeMo-RL EnvironmentInterface
    msg_logs = [
        [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": "[ANSWER_START]ABSTAIN[ANSWER_END]"}]
        for r in first_batch
    ]
    metas = [r["metadata"] for r in first_batch]
    step_out = env_actor.step(msg_logs, metas)

    yaml_path = Path(args.dump_yaml) if args.dump_yaml else None
    if yaml_path is not None:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml.safe_dump(export_cfg, sort_keys=False), encoding="utf-8")

    if args.dry_run or not NEMO_RL_AVAILABLE:
        status = "dry_run_ok" if args.dry_run else "skipped_nemo_rl_not_installed"
        policy_block = export_cfg.get("policy") or {}
        summary = {
            "status": status,
            "nemo_rl_available": NEMO_RL_AVAILABLE,
            "pinned_rl_commit": PINNED_RL_COMMIT,
            "registered_envs": reg_info["registered"],
            "algo": args.algo,
            "stage": entry["stage"],
            "entrypoint": f"{entry['module']}.{entry['train_fn']}",
            "domain": args.domain,
            "model": args.model,
            "hardware": args.hardware,
            "actor_init": policy_block.get("model_name"),
            "sft_anchor": policy_block.get("reference_model_name"),
            "n_train_prompts": len(train_rows),
            "n_val_prompts": len(val_rows),
            "first_batch_size": len(first_batch),
            "reward_channels": sorted(step_out.rewards.keys()) if isinstance(step_out.rewards, dict) else ["scalar"],
            "dump_yaml": str(yaml_path) if yaml_path else None,
        }
        print(json.dumps(summary))
        return {"config": export_cfg, "summary": summary}

    if yaml_path is None:
        yaml_path = REPO_ROOT / "outputs" / "nemo_rl" / f"launch_{args.algo}_{args.domain}.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml.safe_dump(export_cfg, sort_keys=False), encoding="utf-8")
    run_res = launch_nemo_rl(export_cfg, yaml_path, algo=args.algo)
    print(json.dumps(run_res))
    return {"config": export_cfg, "summary": run_res}


if __name__ == "__main__":
    main()
