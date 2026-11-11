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
    for axis_name, axis_path in axes:
        if not axis_path.exists():
            raise FileNotFoundError(f"Missing {axis_name} recipe axis file: {axis_path}")
        merged = deep_merge(merged, load_yaml(axis_path))
    merged = apply_dot_overrides(merged, overrides)
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
