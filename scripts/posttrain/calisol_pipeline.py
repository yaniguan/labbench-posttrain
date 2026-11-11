"""End-to-end CALiSol-23 Conductivity RLVR & Post-Training Pipeline (SFT -> DPO / MPO / PPO / GRPO-GDPO-DAPO).

Uses real experimental ionic conductivity (`k`, in mS/cm) from CALiSol-23 (`13,228` clean measurements
across `27` publications, zero-DOI-leakage `train` / `val` / `test` splits) as the verifiable signal:

  Stage 1 — Cold-Start SFT (`posttrain.sft.train_sft`):
    Teaches `<think>...</think>[ANSWER_START]{...}[ANSWER_END]` physical reasoning + valid electrolyte JSON
    on `train`-DOI top-conductivity formulations (`calisol_design`) and forward observations (`calisol_predict`).
    Produces the frozen `sft_reference` / `sft_anchor` policy.

  Stage 2A — Offline DPO (`posttrain.dpo.train_dpo`) & Hybrid MPO (`posttrain.mpo.train_mpo`):
    Mines matched-temperature (`|dT| <= 8 K`) real experimental preference pairs `(chosen_high_k > rejected_low_k)`
    where both completions follow valid JSON formatting so DPO learns formulation physics rather than syntax.

  Stage 2B — Online PPO (`posttrain.ppo.train_ppo`):
    Critic-bootstrapped token-level GAE + VAPO length-adaptive lambda + Reverse-KL reward shaping against
    the frozen SFT checkpoint on scalar `calisol` verifiable conductivity rewards.

  Stage 2C — Online GRPO with GDPO + DAPO (`posttrain.grpo.train_grpo`):
    Critic-free group rollouts (`G=8`), GDPO independent channel normalization across
    `reward/conductivity`, `reward/support`, and `reward/validity`, DAPO asymmetric clip-higher `[0.20, 0.28]`,
    multi-round dynamic resampling of zero-variance groups, `<think>` Z-MAD reasoning length control,
    and `k3` Reverse-KL anchor (`0.01`) to the frozen SFT checkpoint.

CPU Smoke (no GPU needed):
  python scripts/posttrain/calisol_pipeline.py --smoke --methods sft,dpo,ppo,grpo

GPU Launch (once 1x / 8x A100/H100 arrives):
  python scripts/posttrain/calisol_pipeline.py --model-name Qwen/Qwen3-1.7B --lora \
      --methods sft,dpo,ppo,grpo --output-dir outputs/calisol_qwen3_1.7b
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bioenvs import make_env  # noqa: E402
from bioenvs.base import Sample  # noqa: E402
from labpt_nemo_rl.envs.calisol_env import CALiSolEnv  # noqa: E402
from posttrain.batching import clone_frozen  # noqa: E402
from posttrain.dpo import train_dpo  # noqa: E402
from posttrain.grpo import train_grpo  # noqa: E402
from posttrain.mpo import train_mpo  # noqa: E402
from posttrain.ppo import train_ppo  # noqa: E402
from posttrain.rl_common import generate_rollouts  # noqa: E402
from posttrain.sft import train_sft  # noqa: E402
from posttrain.tiny import build_tiny_lm  # noqa: E402


def _format_design_completion(formulation: dict[str, Any], t_k: float, k_mS_cm: float, *, high: bool = True) -> str:
    solv = formulation["solvents"]
    solv_names = "+".join(sorted(solv.keys()))
    regime = "sub-zero" if t_k < 268.0 else ("elevated-T" if t_k > 315.0 else "ambient")
    qual = (
        f"balances high-permittivity Li+ dissociation and low viscosity (~{k_mS_cm:.2f} mS/cm)"
        if high
        else f"exhibits higher viscosity or ion pairing (~{k_mS_cm:.2f} mS/cm)"
    )
    payload = json.dumps(
        {
            "salt": formulation["salt"],
            "c": round(float(formulation["c"]), 4),
            "c_units": formulation["c_units"],
            "solvent_ratio_type": formulation["solvent_ratio_type"],
            "solvents": {s: round(float(w), 4) for s, w in solv.items()},
        },
        separators=(", ", ": "),
    )
    return (
        f" <think>At {t_k:.1f} K ({regime}), {formulation['salt']} at {formulation['c']:.2f} "
        f"{formulation['c_units']} in {solv_names} {qual}.</think>"
        f"[ANSWER_START]{payload}[ANSWER_END]"
    )


def _format_predict_completion(row_info: dict[str, Any]) -> str:
    k_val = float(row_info["k_mS_cm"])
    sig_val = round(max(0.15, 0.10 * k_val), 3)
    solv_names = "+".join(sorted(row_info["solvents"].keys()))
    payload = json.dumps({"k_mS_cm": round(k_val, 4), "sigma_mS_cm": sig_val}, separators=(", ", ": "))
    return (
        f" <think>For {row_info['salt']} ({row_info['c']:.2f} {row_info['c_units']}) in {solv_names} "
        f"at {row_info['T_K']:.1f} K, VTF transport and ion pairing yield k ~ {k_val:.2f} mS/cm.</think>"
        f"[ANSWER_START]{payload}[ANSWER_END]"
    )


def build_calisol_sft_dataset(
    *,
    split: str = "train",
    task_mode: str = "both",
    max_design: int = 64,
    max_predict: int = 128,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build cold-start SFT examples strictly from `split` DOIs."""
    examples: list[dict[str, Any]] = []
    if task_mode in ("both", "design"):
        design_env = make_env("calisol_design", split=split, seed=seed)
        for i in range(min(max_design, len(design_env))):
            s = design_env.sample(i)
            comp = _format_design_completion(
                s.info["oracle_formulation"],
                float(s.info["target_T_K"]),
                float(s.info["k_max"]),
                high=True,
            )
            examples.append({
                "prompt": s.prompt,
                "completion": comp,
                "task": "calisol_design",
                "id": s.id,
                "dois": s.info["dois"],
            })

    if task_mode in ("both", "predict"):
        pred_env = make_env("calisol_predict", split=split, seed=seed)
        for i in range(min(max_predict, len(pred_env))):
            s = pred_env.sample(i)
            examples.append({
                "prompt": s.prompt,
                "completion": _format_predict_completion(s.info),
                "task": "calisol_predict",
                "id": s.id,
                "doi": s.info["doi"],
            })
    return examples


def build_calisol_dpo_pairs(
    *,
    split: str = "train",
    max_pairs: int = 256,
    min_k_ratio: float = 1.8,
    min_k_gap_mS_cm: float = 2.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Mine matched-temperature experimental conductivity preference pairs `(k_chosen >> k_rejected)`."""
    design_env = make_env("calisol_design", split=split, seed=seed)
    pairs: list[dict[str, Any]] = []

    for i in range(len(design_env)):
        s = design_env.sample(i)
        t_k = float(s.info["target_T_K"])
        c_units = s.info["c_units"]
        ratio_type = s.info["solvent_ratio_type"]

        # Deduplicate compositions inside the regime's local experimental neighborhood
        by_sig: dict[str, dict[str, Any]] = {}
        for rec in s.info["atlas_records"]:
            sig = f"{rec['salt']}|{round(float(rec['c']), 2)}|" + ",".join(
                f"{k}:{w:.2f}" for k, w in sorted(rec["solvents"].items())
            )
            prev = by_sig.get(sig)
            if prev is None or float(rec["k_mS_cm"]) > float(prev["k_mS_cm"]):
                by_sig[sig] = rec

        ordered = sorted(by_sig.values(), key=lambda r: float(r["k_mS_cm"]), reverse=True)
        if len(ordered) < 2:
            continue

        n_top = min(3, max(1, len(ordered) // 4))
        n_bot = min(3, max(1, len(ordered) // 4))
        for hi in ordered[:n_top]:
            for lo in reversed(ordered[-n_bot:]):
                k_hi = float(hi["k_mS_cm"])
                k_lo = float(lo["k_mS_cm"])
                if k_hi < min_k_ratio * max(k_lo, 1e-4) or (k_hi - k_lo) < min_k_gap_mS_cm:
                    continue
                chosen_form = {
                    "salt": hi["salt"],
                    "c": hi["c"],
                    "c_units": c_units,
                    "solvent_ratio_type": ratio_type,
                    "solvents": hi["solvents"],
                }
                rejected_form = {
                    "salt": lo["salt"],
                    "c": lo["c"],
                    "c_units": c_units,
                    "solvent_ratio_type": ratio_type,
                    "solvents": lo["solvents"],
                }
                pairs.append({
                    "prompt": s.prompt,
                    "chosen": _format_design_completion(chosen_form, t_k, k_hi, high=True),
                    "rejected": _format_design_completion(rejected_form, t_k, k_lo, high=False),
                    "record_type": "paired",
                    "target_T_K": t_k,
                    "k_chosen_mS_cm": k_hi,
                    "k_rejected_mS_cm": k_lo,
                    "doi_chosen": hi["doi"],
                    "doi_rejected": lo["doi"],
                })
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def build_calisol_mpo_batch(
    *,
    split: str = "train",
    max_pairs: int = 64,
    max_unpaired: int = 64,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Combine matched-T DPO pairs with unpaired BCO high-k (`label=1`) / low-k (`label=0`) records."""
    paired = build_calisol_dpo_pairs(split=split, max_pairs=max_pairs, seed=seed)
    design_env = make_env("calisol_design", split=split, seed=seed)
    unpaired: list[dict[str, Any]] = []
    for i in range(len(design_env)):
        s = design_env.sample(i)
        t_k = float(s.info["target_T_K"])
        unpaired.append({
            "prompt": s.prompt,
            "completion": _format_design_completion(s.info["oracle_formulation"], t_k, float(s.info["k_max"]), high=True),
            "label": 1,
            "record_type": "unpaired",
            "k_mS_cm": float(s.info["k_max"]),
        })
        if len(unpaired) >= max_unpaired:
            break
        unpaired.append({
            "prompt": s.prompt,
            "completion": _format_design_completion(s.info["worst_formulation"], t_k, float(s.info["k_min"]), high=False),
            "label": 0,
            "record_type": "unpaired",
            "k_mS_cm": float(s.info["k_min"]),
        })
        if len(unpaired) >= max_unpaired:
            break

    # Interleave paired and unpaired records so every mini-batch exercises both DPO and BCO terms
    mixed: list[dict[str, Any]] = []
    for idx in range(max(len(paired), len(unpaired))):
        if idx < len(paired):
            mixed.append(paired[idx])
        if idx < len(unpaired):
            mixed.append(unpaired[idx])
    return mixed


@torch.no_grad()
def evaluate_calisol_policy(
    model,
    tokenizer,
    *,
    split: str = "test",
    n_design: int = 8,
    n_predict: int = 8,
    max_new_tokens: int = 96,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate a checkpoint on the zero-DOI-leakage `test` (or `val`) split."""
    metrics: dict[str, float] = {}
    if n_design > 0:
        d_env = make_env("calisol_design", split=split, seed=seed)
        samples = [d_env.sample(i) for i in range(min(n_design, len(d_env)))]
        rollouts = generate_rollouts(
            model,
            tokenizer,
            [s.prompt for s in samples],
            [{"info": s.info} for s in samples],
            group_size=1,
            max_new=max_new_tokens,
            temperature=0.0,
            seed=seed,
        )
        rewards = [d_env.score(s, r.completion) for s, r in zip(samples, rollouts)]
        metrics[f"{split}/design_reward_mean"] = sum(r.value for r in rewards) / max(len(rewards), 1)
        metrics[f"{split}/design_conductivity_channel"] = sum(
            float(r.components.get("reward/conductivity", 0.0)) for r in rewards
        ) / max(len(rewards), 1)
        metrics[f"{split}/design_support_channel"] = sum(
            float(r.components.get("reward/support", 0.0)) for r in rewards
        ) / max(len(rewards), 1)
        metrics[f"{split}/design_validity_rate"] = sum(
            float(r.components.get("reward/validity", 0.0)) for r in rewards
        ) / max(len(rewards), 1)

    if n_predict > 0:
        p_env = make_env("calisol_predict", split=split, seed=seed)
        samples = [p_env.sample(i) for i in range(min(n_predict, len(p_env)))]
        rollouts = generate_rollouts(
            model,
            tokenizer,
            [s.prompt for s in samples],
            [{"info": s.info} for s in samples],
            group_size=1,
            max_new=max_new_tokens,
            temperature=0.0,
            seed=seed,
        )
        rewards = [p_env.score(s, r.completion) for s, r in zip(samples, rollouts)]
        metrics[f"{split}/predict_reward_mean"] = sum(r.value for r in rewards) / max(len(rewards), 1)
        metrics[f"{split}/predict_accuracy_channel"] = sum(
            float(r.components.get("reward/accuracy", 0.0)) for r in rewards
        ) / max(len(rewards), 1)
        metrics[f"{split}/predict_calibration_channel"] = sum(
            float(r.components.get("reward/calibration", 0.0)) for r in rewards
        ) / max(len(rewards), 1)
    return metrics


def _load_hf_or_tiny_model(model_name: str | None, *, smoke: bool, seed: int):
    if smoke or not model_name or model_name == "tiny":
        return build_tiny_lm(seed=seed, n_positions=1024)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def run_calisol_suite(
    *,
    methods: tuple[str, ...] = ("sft", "dpo", "ppo", "grpo"),
    task_mode: str = "both",
    model_name: str | None = None,
    use_lora: bool = False,
    smoke: bool = False,
    seed: int = 0,
    output_dir: str | Path | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run SFT cold-start followed by head-to-head DPO / MPO / PPO / GRPO on CALiSol-23."""
    cfg = dict(cfg_overrides or {})
    out_root = Path(output_dir) if output_dir else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    base_model, tokenizer = _load_hf_or_tiny_model(model_name, smoke=smoke, seed=seed)
    max_seq_len = 1024 if (smoke or not model_name) else int(cfg.get("max_length", 1536))

    sft_data = build_calisol_sft_dataset(
        split="train",
        task_mode=task_mode,
        max_design=6 if smoke else int(cfg.get("sft_max_design", 64)),
        max_predict=6 if smoke else int(cfg.get("sft_max_predict", 192)),
        seed=seed,
    )

    tasks_list = (
        ["calisol_design", "calisol_predict"]
        if task_mode == "both"
        else ([f"calisol_{task_mode}"] if not task_mode.startswith("calisol_") else [task_mode])
    )
    rl_env = CALiSolEnv(
        tasks=tasks_list,
        split="train",
        seed=seed,
        reward_version="v2",
        multi_reward=True,
        channel_schema="unified",
        max_items_per_task=4 if smoke else int(cfg.get("rl_max_items_per_task", 64)),
    )
    rl_prompts = rl_env.build_dataset(
        n_per_task=2 if smoke else int(cfg.get("rl_prompts_per_task", 32)),
        include_counterfactuals=False,
    )
    if smoke:
        rl_prompts = rl_prompts[:2]
    reward_fn = rl_env.posttrain_reward_fn()

    results: dict[str, Any] = {
        "dataset": {
            "sft_examples": len(sft_data),
            "rl_prompts": len(rl_prompts),
            "tasks": tasks_list,
        },
        "runs": {},
    }

    # Stage 1: SFT Cold-Start (always establishes the frozen SFT reference anchor for RL/DPO)
    sft_steps = 2 if smoke else int(cfg.get("sft_steps", 200))
    sft_lr = 1e-3 if smoke else float(cfg.get("sft_lr", 2e-5))
    sft_ckpt = str(out_root / "sft.pt") if out_root else None
    sft_out = train_sft(
        sft_data,
        steps=sft_steps,
        lr=sft_lr,
        use_lora=use_lora,
        seed=seed,
        max_length=max_seq_len,
        model=base_model,
        tokenizer=tokenizer,
        checkpoint_path=sft_ckpt,
    )
    sft_model = sft_out["model"]
    sft_reference = clone_frozen(sft_model)

    if "sft" in methods:
        results["runs"]["sft"] = {
            "status": sft_out["status"],
            "final_loss": sft_out["final_loss"],
            "losses": sft_out["losses"],
        }

    # Stage 2A: Offline DPO on matched-temperature experimental conductivity pairs
    if "dpo" in methods:
        dpo_pairs = build_calisol_dpo_pairs(
            split="train",
            max_pairs=8 if smoke else int(cfg.get("dpo_max_pairs", 256)),
            seed=seed,
        )
        dpo_model = copy.deepcopy(sft_model)
        dpo_ckpt = str(out_root / "dpo.pt") if out_root else None
        dpo_out = train_dpo(
            dpo_pairs,
            beta=float(cfg.get("dpo_beta", 0.1)),
            sft_coef=float(cfg.get("dpo_sft_coef", 0.1)),
            steps=2 if smoke else int(cfg.get("dpo_steps", 150)),
            lr=1e-3 if smoke else float(cfg.get("dpo_lr", 5e-6)),
            max_length=max_seq_len,
            seed=seed,
            model=dpo_model,
            tokenizer=tokenizer,
            reference=sft_reference,
            checkpoint_path=dpo_ckpt,
        )
        results["runs"]["dpo"] = {
            "status": dpo_out["status"],
            "final_loss": dpo_out["final_loss"],
            "num_pairs": len(dpo_pairs),
            "reference_unchanged": dpo_out["reference_unchanged"],
            "likelihood_displacement": dpo_out["likelihood_displacement"],
        }

    # Stage 2A+: Hybrid MPO (DPO + BCO EMA reward_shift + SFT anchor)
    if "mpo" in methods:
        mpo_records = build_calisol_mpo_batch(
            split="train",
            max_pairs=4 if smoke else int(cfg.get("mpo_max_pairs", 128)),
            max_unpaired=4 if smoke else int(cfg.get("mpo_max_unpaired", 128)),
            seed=seed,
        )
        mpo_model = copy.deepcopy(sft_model)
        mpo_ckpt = str(out_root / "mpo.pt") if out_root else None
        mpo_out = train_mpo(
            mpo_records,
            beta=float(cfg.get("mpo_beta", 0.1)),
            preference_weight=1.0,
            bco_weight=0.8,
            sft_weight=0.2,
            reward_shift_momentum=0.95,
            steps=2 if smoke else int(cfg.get("mpo_steps", 150)),
            lr=1e-3 if smoke else float(cfg.get("mpo_lr", 5e-6)),
            max_length=max_seq_len,
            seed=seed,
            model=mpo_model,
            tokenizer=tokenizer,
            reference=sft_reference,
            checkpoint_path=mpo_ckpt,
        )
        results["runs"]["mpo"] = {
            "status": mpo_out["status"],
            "final_loss": mpo_out["final_loss"],
            "num_records": len(mpo_records),
            "reference_unchanged": mpo_out["reference_unchanged"],
            "final_reward_shift": mpo_out["reward_shift"],
        }

    # Stage 2B: Online PPO (Critic GAE + VAPO length-adaptive lambda + KL-in-reward)
    if "ppo" in methods:
        ppo_model = copy.deepcopy(sft_model)
        ppo_ckpt = str(out_root / "ppo.pt") if out_root else None
        ppo_out = train_ppo(
            rl_prompts,
            reward_fn,
            kl_coef=float(cfg.get("ppo_kl_coef", 0.02)),
            kl_estimator="k3",
            kl_in_reward=True,
            eps_low=0.20,
            eps_high=0.20,
            vapo_length_adaptive_lam=True,
            vapo_alpha=0.05,
            positive_nll_coef=float(cfg.get("ppo_positive_nll_coef", 0.05)),
            positive_reward_threshold=0.70,
            group_size=2 if smoke else int(cfg.get("ppo_group_size", 4)),
            max_new_tokens=8 if smoke else int(cfg.get("max_new_tokens", 256)),
            steps=2 if smoke else int(cfg.get("ppo_steps", 100)),
            lr=1e-3 if smoke else float(cfg.get("ppo_lr", 1e-6)),
            seed=seed,
            model=ppo_model,
            tokenizer=tokenizer,
            reference=sft_reference,
            sft_anchor=clone_frozen(sft_reference),
            checkpoint_path=ppo_ckpt,
        )
        results["runs"]["ppo"] = {
            "status": ppo_out["status"],
            "final_loss": ppo_out["final_loss"],
            "reference_unchanged": ppo_out["reference_unchanged"],
            "sft_anchor_unchanged": ppo_out["sft_anchor_unchanged"],
        }

    # Stage 2C: Online GRPO with GDPO multi-channel conductivity/support/validity + DAPO
    if "grpo" in methods:
        grpo_model = copy.deepcopy(sft_model)
        grpo_ckpt = str(out_root / "grpo_gdpo.pt") if out_root else None
        grpo_out = train_grpo(
            rl_prompts,
            reward_fn,
            kl_coef=float(cfg.get("grpo_kl_coef", 0.01)),
            kl_estimator="k3",
            kl_in_reward=False,
            eps_low=float(cfg.get("grpo_eps_low", 0.20)),
            eps_high=float(cfg.get("grpo_eps_high", 0.28)),
            dual_clip_c=3.0,
            advantage_mode=str(cfg.get("advantage_mode", "gdpo")),
            gdpo_weights={
                "reward/conductivity": 1.2,
                "reward/support": 0.8,
                "reward/validity": 0.5,
            },
            dynamic_sampling=bool(cfg.get("dynamic_sampling", not smoke)),
            dynamic_sampling_max_rounds=2 if smoke else int(cfg.get("dynamic_sampling_max_rounds", 4)),
            overlong_filtering=not smoke,
            stop_properly_coef=0.2,
            reasoning_length_coef=0.05,
            reasoning_length_mode="zmad",
            overlong_buffer_length=6 if smoke else int(cfg.get("overlong_buffer_length", 192)),
            overlong_buffer_penalty=0.3,
            use_is_correction=bool(cfg.get("use_is_correction", False)),
            is_mode="icepop",
            is_clip_min=0.5,
            is_clip=2.0,
            group_size=4 if smoke else int(cfg.get("grpo_group_size", 8)),
            max_new_tokens=8 if smoke else int(cfg.get("max_new_tokens", 256)),
            steps=2 if smoke else int(cfg.get("grpo_steps", 120)),
            lr=1e-3 if smoke else float(cfg.get("grpo_lr", 1e-6)),
            seed=seed,
            model=grpo_model,
            tokenizer=tokenizer,
            reference=sft_reference,
            sft_anchor=clone_frozen(sft_reference),
            checkpoint_path=grpo_ckpt,
        )
        results["runs"]["grpo"] = {
            "status": grpo_out["status"],
            "final_loss": grpo_out["final_loss"],
            "reference_unchanged": grpo_out["reference_unchanged"],
            "sft_anchor_unchanged": grpo_out["sft_anchor_unchanged"],
        }
        results["eval_test"] = evaluate_calisol_policy(
            grpo_model,
            tokenizer,
            split="test",
            n_design=2 if smoke else 16,
            n_predict=2 if smoke else 16,
            max_new_tokens=8 if smoke else int(cfg.get("max_new_tokens", 256)),
            seed=seed,
        )

    if out_root:
        (out_root / "calisol_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="CALiSol-23 Conductivity RLVR Suite (SFT / DPO / MPO / PPO / GRPO).")
    parser.add_argument("--methods", type=str, default="sft,dpo,ppo,grpo", help="Comma-separated: sft,dpo,mpo,ppo,grpo or all")
    parser.add_argument("--task-mode", choices=("both", "design", "predict"), default="both")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config overriding hyperparameters")
    parser.add_argument("--model-name", type=str, default=None, help="HuggingFace model ID (e.g. Qwen/Qwen3-1.7B)")
    parser.add_argument("--lora", action="store_true", help="Enable Qwen/Llama/GPT-2 LoRA adapters")
    parser.add_argument("--smoke", action="store_true", help="Fast CPU verification with in-memory tiny causal LM")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    model_name = args.model_name or cfg.get("model_name")
    use_lora = bool(args.lora or cfg.get("use_lora", False))
    task_mode = str(cfg.get("task_mode", args.task_mode))
    if args.methods.strip().lower() == "all":
        methods = ("sft", "dpo", "mpo", "ppo", "grpo")
    else:
        methods = tuple(m.strip().lower() for m in args.methods.split(",") if m.strip())

    res = run_calisol_suite(
        methods=methods,
        task_mode=task_mode,
        model_name=model_name,
        use_lora=use_lora,
        smoke=args.smoke,
        seed=args.seed,
        output_dir=args.output_dir or cfg.get("output_dir"),
        cfg_overrides=cfg,
    )
    print(json.dumps({"status": "ran", "methods": list(res["runs"].keys()), "dataset": res["dataset"]}))
    return res


if __name__ == "__main__":
    main()
