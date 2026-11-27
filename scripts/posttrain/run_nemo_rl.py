"""NeMo-RL (`yaniguan/RL`) launcher for BioProBench & CALiSol-23 SFT / DPO / GRPO / GSPO / GDPO recipes (#16, #49, #52).

Composes a 4-axis recipe (`algo` x `domain` x `model` x `hardware`), registers
`labbench_bioprobench` and `labbench_calisol` into `nemo_rl.environments.utils.ENV_REGISTRY`
(or verifies registration locally when `--dry-run` is passed), builds the prompt dataset and
first mini-batch, and launches the stage-specific `yaniguan/RL` driver on Ray.

Example:
    python scripts/posttrain/run_nemo_rl.py --algo gspo_longcot --domain bioenvs_pqa_ord_err \
        --model qwen3_1.7b --hardware local_1gpu_colocated --dry-run
"""
from __future__ import annotations

import argparse
from functools import partial
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
    """Resolve stage-specific NeMo-RL (`yaniguan/RL`) driver module and train function (#49, #52)."""
    algo_norm = (algo or "").strip().lower()
    if algo_norm in ("mpo", "mpo_offline") or ("mpo" in cfg and "grpo" not in cfg):
        raise NotImplementedError(
            f"Algorithm {algo_norm or 'mpo'!r} (hybrid DPO + BCO + EMA reward-shift) is not implemented in "
            f"pinned yaniguan/RL@{PINNED_RL_COMMIT}; silently falling back to dpo_train would drop BCO. "
            "Run MPO via `posttrain.mpo.train_mpo` / `TorchLocalEngine`, or use `--algo dpo` on yaniguan/RL."
        )
    if algo_norm == "sft" or ("sft" in cfg and "grpo" not in cfg):
        return {
            "stage": "sft",
            "module": "nemo_rl.algorithms.sft",
            "train_fn": "sft_train",
            "setup_fn": "setup",
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


def _setup_sft_data(tokenizer: Any, data_config: dict[str, Any], policy_config: dict[str, Any]) -> tuple[Any, Any]:
    """Build SFT train and validation `AllTaskProcessedDataset` matching `examples/run_sft.py` (#52)."""
    from nemo_rl.data.datasets import (  # type: ignore
        AllTaskProcessedDataset,
        load_response_dataset,
        merge_datasets,
        update_single_dataset_config,
    )

    def _load_one(ds_cfg: dict[str, Any]) -> Any:
        if ds_cfg.get("dataset_name") != "megatron_sft_packed":
            return load_response_dataset(ds_cfg)
        megatron_cfg = policy_config.get("megatron_cfg")
        if megatron_cfg is None or not megatron_cfg.get("enabled"):
            raise ValueError("Direct Megatron-LM prepacked SFT requires the Megatron backend")
        return load_response_dataset(
            ds_cfg,
            context_parallel_size=int(megatron_cfg["context_parallel_size"]),
        )

    train_entries = data_config["train"]
    if isinstance(train_entries, dict):
        train_entries = [train_entries]

    task_data_processors: dict[str, Any] = {}
    task_data_preprocessors: dict[str, Any] = {}
    data_list: list[Any] = []
    for ds_cfg in train_entries:
        if "default" in data_config and data_config["default"] is not None:
            update_single_dataset_config(ds_cfg, data_config["default"])
        loaded = _load_one(ds_cfg)
        data_list.append(loaded)
        proc = partial(
            loaded.processor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
        )
        task_data_processors[loaded.task_name] = (loaded.task_spec, proc)
        if getattr(loaded, "preprocessor", None) is not None:
            task_data_preprocessors[loaded.task_name] = loaded.preprocessor

    dataset = AllTaskProcessedDataset(
        merge_datasets([d.dataset for d in data_list]),
        tokenizer,
        None,
        task_data_processors,
        task_data_preprocessors=task_data_preprocessors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_task_processors: dict[str, Any] = {}
    val_task_preprocessors: dict[str, Any] = {}
    val_data_list: list[Any] = []
    for loaded in data_list:
        if getattr(loaded, "val_dataset", None) is not None:
            val_data_list.append(loaded.val_dataset)
            val_task_processors[loaded.task_name] = task_data_processors[loaded.task_name]
            if loaded.task_name in task_data_preprocessors:
                val_task_preprocessors[loaded.task_name] = task_data_preprocessors[loaded.task_name]

    val_entries = data_config.get("validation")
    if val_entries is not None:
        if isinstance(val_entries, dict):
            val_entries = [val_entries]
        for ds_cfg in val_entries:
            if "default" in data_config and data_config["default"] is not None:
                update_single_dataset_config(ds_cfg, data_config["default"])
            val_loaded = _load_one(ds_cfg)
            val_data_list.append(val_loaded.dataset)
            val_proc = partial(
                val_loaded.processor,
                add_bos=data_config["add_bos"],
                add_eos=data_config["add_eos"],
                add_generation_prompt=data_config["add_generation_prompt"],
            )
            val_task_processors[val_loaded.task_name] = (val_loaded.task_spec, val_proc)
            if getattr(val_loaded, "preprocessor", None) is not None:
                val_task_preprocessors[val_loaded.task_name] = val_loaded.preprocessor

    val_dataset = None
    if val_data_list:
        val_dataset = AllTaskProcessedDataset(
            merge_datasets(val_data_list),
            tokenizer,
            None,
            val_task_processors,
            task_data_preprocessors=val_task_preprocessors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    return dataset, val_dataset


def launch_nemo_rl(
    export_cfg: dict[str, Any],
    yaml_path: Path,
    *,
    algo: str | None = None,
) -> dict[str, Any]:
    """Invoke the stage-specific `yaniguan/RL` driver (`sft`, `dpo`, or `grpo`, #49, #52)."""
    import importlib

    from omegaconf import OmegaConf  # type: ignore
    from nemo_rl.algorithms.utils import get_tokenizer  # type: ignore
    from nemo_rl.distributed.virtual_cluster import init_ray  # type: ignore
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers  # type: ignore
    from nemo_rl.utils.outdated_config_checks import check_outdated_config  # type: ignore

    entry = resolve_nemo_entrypoint(algo, export_cfg)
    stage_mod = importlib.import_module(entry["module"])
    master_cls = getattr(stage_mod, "MasterConfig", dict)
    setup_fn = getattr(stage_mod, entry["setup_fn"])
    train_fn = getattr(stage_mod, entry["train_fn"])

    register_omegaconf_resolvers()
    raw_cfg = OmegaConf.to_container(load_config(str(yaml_path)), resolve=True)
    check_outdated_config(raw_cfg)
    config = master_cls(**raw_cfg) if callable(master_cls) else raw_cfg
    init_ray()

    if entry["stage"] == "sft":
        policy_cfg = config["policy"] if isinstance(config, dict) else config.policy
        data_cfg = config["data"] if isinstance(config, dict) else config.data
        tokenizer = get_tokenizer(policy_cfg["tokenizer"])
        dataset, val_dataset = _setup_sft_data(tokenizer, data_cfg, policy_cfg)
        (
            policy,
            cluster,
            train_dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            sft_save_state,
            master_config,
        ) = setup_fn(config, tokenizer, dataset, val_dataset)
        with checkpointer:
            train_fn(
                policy=policy,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                master_config=master_config,
                logger=logger,
                checkpointer=checkpointer,
                sft_save_state=sft_save_state,
            )
        return {
            "status": "ran",
            "backend": "nemo_rl",
            "stage": entry["stage"],
            "entrypoint": f"{entry['module']}.{entry['train_fn']}",
            "config_path": str(yaml_path),
            "cluster": str(type(cluster).__name__),
        }

    if entry["stage"] == "dpo":
        from nemo_rl.data.utils import setup_preference_data  # type: ignore

        policy_cfg = config["policy"] if isinstance(config, dict) else config.policy
        data_cfg = config["data"] if isinstance(config, dict) else config.data
        tokenizer = get_tokenizer(policy_cfg["tokenizer"])
        dataset, val_dataset = setup_preference_data(tokenizer, data_cfg)
        (
            policy,
            cluster,
            train_dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            dpo_save_state,
            master_config,
        ) = setup_fn(config, tokenizer, dataset, val_dataset)
        with checkpointer:
            train_fn(
                policy=policy,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                master_config=master_config,
                logger=logger,
                checkpointer=checkpointer,
                dpo_save_state=dpo_save_state,
            )
        return {
            "status": "ran",
            "backend": "nemo_rl",
            "stage": entry["stage"],
            "entrypoint": f"{entry['module']}.{entry['train_fn']}",
            "config_path": str(yaml_path),
            "cluster": str(type(cluster).__name__),
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
        _ = (cluster, teacher_worker_groups, alias_to_group_alias)
        with checkpointer:
            train_fn(
                policy=policy,
                policy_generation=policy_generation,
                wrapped_dataloader=dataloader,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                task_to_env=task_to_env,
                val_task_to_env=val_task_to_env,
                logger=logger,
                checkpointer=checkpointer,
                grpo_save_state=grpo_state,
                master_config=master_config,
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
    ap = argparse.ArgumentParser(description="Compose and launch SFT/DPO/GRPO/GSPO/GDPO recipes on yaniguan/RL (#16, #49, #52).")
    ap.add_argument("--algo", default="gspo_longcot", help="Algorithm axis under configs/recipes/algo/")
    ap.add_argument("--domain", default="bioenvs_pqa_ord_err", help="Domain axis under configs/recipes/domain/")
    ap.add_argument("--model", default="qwen3_1.7b", help="Model axis under configs/recipes/model/")
    ap.add_argument("--hardware", default="local_1gpu_colocated", help="Hardware axis under configs/recipes/hardware/")
    ap.add_argument("--init-checkpoint", default=None, help="Upstream actor checkpoint path (e.g. SFT or DPO)")
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
