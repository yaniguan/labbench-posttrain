"""4-axis recipe composition (`algo` x `domain` x `model` x `hardware`) and pre-flight validators."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

FORBIDDEN_NEMO_RL_TOP_KEYS = frozenset({"status", "note"})
RECIPES_ROOT = Path(__file__).resolve().parents[2] / "configs" / "recipes"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def apply_dot_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    out = deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Expected key=value override, got {item!r}")
        key_path, raw_val = item.split("=", 1)
        val = yaml.safe_load(raw_val)
        cur = out
        parts = key_path.strip().split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root in {path} must be a mapping")
    return data


def gdpo_channel_weights_to_reward_weights(channel_weights: dict[str, Any]) -> list[float]:
    """Convert `gdpo_channel_weights` mapping into ordered `reward_weights` matching sorted `reward/*` keys (#52)."""
    if not isinstance(channel_weights, dict) or not channel_weights:
        return []
    return [float(channel_weights[k]) for k in sorted(channel_weights.keys())]


def _finalize_nemo_rl_schema(cfg: dict[str, Any]) -> None:
    """Populate required `yaniguan/RL` `MasterConfig` fields from concise 4-axis recipe defaults (#52)."""
    meta = cfg.get("recipe_meta") or {}
    algo = str(meta.get("algo") or "").strip().lower()
    domain = str(meta.get("domain") or "labbench").strip()

    # Migrate top-level `async_grpo` into `grpo.async_grpo` if present on GRPO recipes
    top_async = cfg.pop("async_grpo", None)
    if isinstance(top_async, dict) and isinstance(cfg.get("grpo"), dict):
        grpo_async = cfg["grpo"].setdefault("async_grpo", {})
        if isinstance(grpo_async, dict):
            for k, v in top_async.items():
                grpo_async.setdefault(k, v)

    policy = cfg.setdefault("policy", {})
    if "dtensor_cfg" in policy and "automodel_cfg" not in policy:
        policy["automodel_cfg"] = policy.pop("dtensor_cfg")
    elif "dtensor_cfg" in policy:
        policy.pop("dtensor_cfg", None)

    model_name = str(policy.get("model_name") or "Qwen/Qwen3-1.7B")
    max_seq = int(policy.get("max_total_sequence_length") or 4096)
    micro_bs = int(policy.get("train_micro_batch_size") or 1)
    policy.setdefault("model_name", model_name)
    policy.setdefault("precision", "bfloat16")
    policy.setdefault("max_total_sequence_length", max_seq)
    policy.setdefault("train_global_batch_size", 16)
    policy.setdefault("train_micro_batch_size", micro_bs)
    policy.setdefault("max_grad_norm", 1.0)
    policy.setdefault("offload_optimizer_for_logprob", False)
    policy.setdefault("offload_policy_before_refit", False)
    policy.setdefault("offload_optimizer_for_refit", True)

    tok = policy.setdefault("tokenizer", {})
    if isinstance(tok, dict):
        tok["name"] = str(tok.get("name") or model_name)
        tok.setdefault("chat_template_kwargs", None)

    megatron_cfg = policy.get("megatron_cfg")
    if isinstance(megatron_cfg, dict) and megatron_cfg.get("enabled"):
        tp_size = int(megatron_cfg.get("tensor_model_parallel_size") or 1)
    else:
        auto_cfg = policy.setdefault("automodel_cfg", {"enabled": True, "tensor_parallel_size": 1})
        if isinstance(auto_cfg, dict) and auto_cfg.get("enabled", True):
            auto_cfg.setdefault("enabled", True)
            auto_cfg.setdefault("tensor_parallel_size", 1)
            auto_cfg.setdefault("context_parallel_size", 1)
            auto_cfg.setdefault("sequence_parallel", False)
            auto_cfg.setdefault("activation_checkpointing", False)
            auto_cfg.setdefault("cpu_offload", False)
        tp_size = int(auto_cfg.get("tensor_parallel_size") or 1) if isinstance(auto_cfg, dict) else 1
    policy.setdefault("make_sequence_length_divisible_by", max(1, tp_size))

    dyn_batch = policy.setdefault("dynamic_batching", {})
    if isinstance(dyn_batch, dict):
        dyn_batch.setdefault("enabled", False)
        dyn_batch.setdefault("train_mb_tokens", max_seq * micro_bs)
        dyn_batch.setdefault("sequence_length_round", 64)

    seq_pack = policy.get("sequence_packing")
    if isinstance(seq_pack, dict):
        seq_pack.setdefault("enabled", False)
        seq_pack.setdefault("train_mb_tokens", max_seq * micro_bs)
        seq_pack.setdefault("algorithm", "modified_first_fit_decreasing")
        seq_pack.setdefault("sequence_length_round", 64)

    opt = policy.get("optimizer")
    if isinstance(opt, dict) and "name" not in opt and "kwargs" not in opt:
        lr = float(opt.get("lr", 1.0e-6))
        wd = float(opt.get("weight_decay", 0.01))
        policy["optimizer"] = {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": lr,
                "weight_decay": wd,
                "betas": [0.9, 0.98],
                "eps": 1.0e-5,
                "foreach": False,
                "fused": False,
            },
        }
    elif opt is None:
        policy["optimizer"] = {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 1.0e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.98],
                "eps": 1.0e-5,
                "foreach": False,
                "fused": False,
            },
        }

    gen = policy.get("generation")
    if isinstance(gen, dict):
        temp = float(gen.get("temperature", 1.0))
        top_p = float(gen.get("top_p", 1.0))
        top_k = gen.get("top_k", None)
        gen.setdefault("backend", "hf")
        gen.setdefault("max_new_tokens", min(2048, max(64, max_seq // 2)))
        gen.setdefault("temperature", temp)
        gen.setdefault("top_p", top_p)
        gen.setdefault("top_k", top_k)
        gen.setdefault("val_temperature", temp)
        gen.setdefault("val_top_p", top_p)
        gen.setdefault("val_top_k", top_k)
        gen.setdefault("stop_token_ids", None)
        gen.setdefault("stop_strings", None)
        coloc = gen.setdefault("colocated", {"enabled": True})
        if isinstance(coloc, dict):
            coloc.setdefault("enabled", True)
            res = coloc.setdefault("resources", {})
            if isinstance(res, dict):
                res.setdefault("gpus_per_node", None)
                res.setdefault("num_nodes", None)
        vllm_cfg = gen.get("vllm_cfg")
        if isinstance(vllm_cfg, dict):
            vllm_cfg.setdefault("async_engine", False)
            vllm_cfg.setdefault("precision", str(policy.get("precision", "bfloat16")))
            vllm_cfg.setdefault("tensor_parallel_size", 1)
            vllm_cfg.setdefault("pipeline_parallel_size", 1)
            vllm_cfg.setdefault("expert_parallel_size", 1)
            vllm_cfg.setdefault("gpu_memory_utilization", 0.65)
            vllm_cfg.setdefault("max_model_len", max_seq)
            vllm_cfg.setdefault("enforce_eager", False)

    data = cfg.setdefault("data", {})
    if isinstance(data, dict):
        ds_name = str(data.get("dataset_name") or domain)
        data.setdefault("max_input_seq_length", max_seq)
        data.setdefault("shuffle", True)
        data.setdefault("num_workers", 1)
        data.setdefault("add_bos", True)
        data.setdefault("add_eos", True)
        data.setdefault("add_generation_prompt", algo not in ("sft", "dpo", "mpo_offline"))
        if "train" not in data:
            data["train"] = {"dataset_name": ds_name, "split": "train"}
        if "validation" not in data:
            data["validation"] = (
                {"dataset_name": ds_name, "split": "validation"}
                if algo in ("sft", "dpo", "mpo_offline")
                else None
            )

    logger_cfg = cfg.setdefault("logger", {})
    if isinstance(logger_cfg, dict):
        run_tag = f"{algo or 'rl'}-{domain}"
        logger_cfg.setdefault("log_dir", "logs")
        logger_cfg.setdefault("num_val_samples_to_print", 0)
        logger_cfg.setdefault("extra_backends", [])
        logger_cfg.setdefault("wandb_enabled", False)
        logger_cfg.setdefault("tensorboard_enabled", False)
        logger_cfg.setdefault("mlflow_enabled", False)
        logger_cfg.setdefault("swanlab_enabled", False)
        logger_cfg.setdefault("monitor_gpus", False)
        logger_cfg.setdefault("wandb", {"project": "labbench-posttrain", "name": run_tag})
        logger_cfg.setdefault("swanlab", {"project": "labbench-posttrain", "name": run_tag})
        logger_cfg.setdefault("tensorboard", {"log_dir": f"tb_logs-{run_tag}"})
        logger_cfg.setdefault(
            "mlflow",
            {"experiment_name": "labbench-posttrain", "run_name": run_tag, "tracking_uri": "http://localhost:5000"},
        )
        logger_cfg.setdefault("gpu_monitoring", {"collection_interval": 10, "flush_interval": 10})

    ckpt = cfg.setdefault("checkpointing", {})
    if isinstance(ckpt, dict):
        ckpt.setdefault("enabled", True)
        ckpt.setdefault("checkpoint_dir", f"outputs/recipes/{algo or 'rl'}")
        if algo == "sft" or ("sft" in cfg and "grpo" not in cfg):
            default_metric, default_hib = "val:val_loss", False
        elif algo in ("dpo", "mpo_offline") or ("dpo" in cfg and "grpo" not in cfg):
            default_metric, default_hib = "val:validation-default_loss", False
        else:
            default_metric, default_hib = "val:accuracy", True
        ckpt.setdefault("metric_name", default_metric)
        ckpt.setdefault("higher_is_better", default_hib)
        ckpt.setdefault("keep_top_k", 3)
        ckpt.setdefault("save_period", 50)
        ckpt.setdefault("checkpoint_must_save_by", None)
        ckpt.setdefault("save_optimizer", True)

    grpo = cfg.get("grpo")
    if isinstance(grpo, dict):
        adv = grpo.setdefault("adv_estimator", {})
        if isinstance(adv, dict):
            w_map = adv.get("gdpo_channel_weights")
            if isinstance(w_map, dict) and w_map and adv.get("reward_weights") is None:
                adv["reward_weights"] = gdpo_channel_weights_to_reward_weights(w_map)
        shaping = grpo.get("reward_shaping")
        if isinstance(shaping, dict) and shaping.get("enabled"):
            shaping.setdefault("max_response_length", max_seq)


def compose_recipe(
    *,
    algo: str,
    domain: str,
    model: str,
    hardware: str,
    recipes_root: str | Path = RECIPES_ROOT,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Compose a complete runnable recipe from the 4 orthogonal YAML axes."""
    root = Path(recipes_root)
    axes = [
        ("hardware", root / "hardware" / f"{hardware}.yaml"),
        ("model", root / "model" / f"{model}.yaml"),
        ("domain", root / "domain" / f"{domain}.yaml"),
        ("algo", root / "algo" / f"{algo}.yaml"),
    ]
    merged: dict[str, Any] = {
        "recipe_meta": {
            "algo": algo,
            "domain": domain,
            "model": model,
            "hardware": hardware,
        }
    }
    algo_cfg: dict[str, Any] = {}
    domain_cfg: dict[str, Any] = {}
    for axis_name, axis_path in axes:
        if not axis_path.exists():
            raise FileNotFoundError(f"Missing {axis_name} recipe axis file: {axis_path}")
        loaded = load_yaml(axis_path)
        if axis_name == "domain":
            domain_cfg = loaded
        elif axis_name == "algo":
            algo_cfg = loaded
        merged = deep_merge(merged, loaded)
    if "grpo" not in algo_cfg:
        merged.pop("grpo", None)
    else:
        # Preserve domain-explicit `gdpo_channel_weights` without cross-domain key pollution
        # from `algo/gdpo_multireward.yaml` defaults (#49, #52).
        dom_weights = (
            ((domain_cfg.get("grpo") or {}).get("adv_estimator") or {}).get("gdpo_channel_weights")
        )
        algo_adv = (algo_cfg.get("grpo") or {}).get("adv_estimator") or {}
        if isinstance(dom_weights, dict) and dom_weights and (
            "gdpo_channel_weights" in algo_adv or algo_adv.get("name") == "gdpo"
        ):
            merged.setdefault("grpo", {}).setdefault("adv_estimator", {})[
                "gdpo_channel_weights"
            ] = deepcopy(dom_weights)
    merged = apply_dot_overrides(merged, overrides)
    _finalize_nemo_rl_schema(merged)
    validate_composed_recipe(merged)
    return merged


def validate_nemo_rl_yaml(cfg: dict[str, Any], *, source: str = "<config>") -> None:
    """Validate that a NeMo-RL YAML dictionary has no stray keys or mutually-exclusive loss flags."""
    bad_keys = sorted(FORBIDDEN_NEMO_RL_TOP_KEYS & set(cfg.keys()))
    if bad_keys:
        raise ValueError(
            f"{source} contains non-schema top-level keys {bad_keys} that fail NeMo-RL strict config validation; "
            "use YAML '#' comments instead."
        )
    loss_fn = cfg.get("loss_fn", {}) or {}
    if loss_fn.get("sequence_level_importance_sampling_geometric_mean") and not loss_fn.get(
        "sequence_level_importance_ratios", False
    ):
        raise ValueError(
            f"{source}: `sequence_level_importance_sampling_geometric_mean=true` requires `sequence_level_importance_ratios=true`."
        )
    if loss_fn.get("sequence_level_importance_ratios") and loss_fn.get("token_level_loss", False):
        raise ValueError(f"{source}: GSPO `sequence_level_importance_ratios=true` requires `token_level_loss=false`.")
    if loss_fn.get("use_cispo") and not loss_fn.get("token_level_loss", True):
        raise ValueError(f"{source}: CISPO `use_cispo=true` requires `token_level_loss=true`.")


def validate_composed_recipe(cfg: dict[str, Any]) -> None:
    """Pre-flight validation across algorithm, model, domain, and hardware axes."""
    validate_nemo_rl_yaml(cfg, source=str(cfg.get("recipe_meta", "composed_recipe")))
    loss_fn = cfg.get("loss_fn", {}) or {}
    policy = cfg.get("policy", {}) or {}
    gen = policy.get("generation", {}) or {}
    colocated = (gen.get("colocated") or {}).get("enabled", False)
    backend = gen.get("backend", "hf")
    max_seq = int(policy.get("max_total_sequence_length", 1024))
    algo_name = (cfg.get("recipe_meta") or {}).get("algo", "")

    # Online RL with vLLM sampler + HF/Megatron trainer on long sequences needs IS mismatch gating
    if algo_name in ("dapo", "gdpo_multireward", "gspo_longcot", "tropd_distill") and backend == "vllm" and max_seq >= 4096:
        if not loss_fn.get("use_importance_sampling_correction", False):
            raise ValueError(
                f"Recipe {algo_name!r} uses vLLM generation at max_total_sequence_length={max_seq} "
                "without `loss_fn.use_importance_sampling_correction=true` (recommended: `icepop` [0.5, 2.0])."
            )
    if colocated and cfg.get("cluster", {}).get("gpus_per_node", 1) < 1:
        raise ValueError("Colocated generation requires at least 1 GPU per node.")


def validate_sft_rl_pair(sft_cfg: dict[str, Any], rl_cfg: dict[str, Any]) -> None:
    """Verify SFT and downstream RL configs share compatible base models before merging LoRA adapters."""
    sft_model = sft_cfg.get("model") or (sft_cfg.get("policy") or {}).get("model_name")
    rl_model = rl_cfg.get("model") or (rl_cfg.get("policy") or {}).get("model_name")
    if not sft_model or not rl_model:
        raise ValueError("Both SFT and RL configs must declare a model / policy.model_name")
    if str(sft_model).strip() != str(rl_model).strip():
        raise ValueError(
            f"Base-model mismatch between SFT ({sft_model!r}) and RL ({rl_model!r}): "
            "LoRA adapter weights cannot be merged across different hidden dimensions."
        )
