"""Colab runner for the post-training ladder. Not executed in CI. Results are not run.

Order, cheapest first (also in docs/posttrain/reward_baselines.md):

    1. smoke          tiny model, CPU, toy strings
    2. sft            0.5B LoRA on gold completions from one env (ERR first)
    3. dpo            same model, checker-ranked pairs
    4. grpo           ground-truth reward, small group
    5. ladder         format_only and random_p50 on that model, then a second family

Do not start at 7B. Do not run a full ladder on one family and call it done.

Precision
    T4 (compute capability 7.5) has no bf16. Use fp16 there.
    A100 and later can use bf16. pick_dtype() checks torch.cuda.is_bf16_supported().
    CPU fallback is fp32 and only runs the smoke trainers.

Resume
    Checkpoints store the model, the optimizer, the step, and the Python/NumPy/torch
    RNG. Pass --resume. Mount Drive with --drive so a dead runtime does not lose them.

This script does not download anything unless you pass --stage sft (or later) on a
machine with a GPU. --smoke and --cpu-fallback stay local.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = ("smoke", "sft", "dpo", "grpo", "ladder")


def pick_dtype(torch_module=None):
    import torch

    torch = torch_module or torch
    if not torch.cuda.is_available():
        return torch.float32, "cpu-fp32"
    # T4 returns False. Do not ask a T4 for bf16.
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "cuda-bf16"
    return torch.float16, "cuda-fp16"


def maybe_drive(enabled: bool) -> str | None:
    if not enabled:
        return None
    try:
        from google.colab import drive
    except ImportError:
        print("not on Colab; checkpoints stay on the local disk", flush=True)
        return None
    drive.mount("/content/drive")
    out = "/content/drive/MyDrive/labbench-posttrain"
    Path(out).mkdir(parents=True, exist_ok=True)
    return out


def run_smoke_stage() -> int:
    from posttrain.dpo import run_smoke as dpo_smoke
    from posttrain.grpo import run_smoke as grpo_smoke
    from posttrain.ppo import run_smoke as ppo_smoke
    from posttrain.sft import run_smoke as sft_smoke

    for name, fn in (("sft", sft_smoke), ("dpo", dpo_smoke), ("ppo", ppo_smoke), ("grpo", grpo_smoke)):
        rows = fn(steps=1)
        if not rows or not all(r["finite"] for r in rows):
            print(f"{name} smoke failed", flush=True)
            return 1
    print("colab smoke stage done. task metrics: not run", flush=True)
    return 0


def refuse_full_training_on_cpu() -> int:
    print(
        "Full SFT/DPO/GRPO needs a GPU. Nothing was trained. "
        "Task metrics: not run. Re-run with a GPU, or pass --cpu-fallback to stay on the smoke stage.",
        flush=True,
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Colab post-training runner. Not run in CI.")
    parser.add_argument("--stage", choices=STAGES, default="smoke")
    parser.add_argument("--config", default=str(ROOT / "configs" / "posttrain" / "qwen_0.5b.yaml"))
    parser.add_argument("--drive", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--cpu-fallback", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    import torch

    dtype, dtype_name = pick_dtype(torch)
    print(f"dtype={dtype_name} cuda={torch.cuda.is_available()}", flush=True)
    drive_dir = maybe_drive(args.drive)
    if args.resume:
        print(f"resume requested: {args.resume}", flush=True)
    if drive_dir:
        print(f"checkpoints under {drive_dir}", flush=True)

    stage = "smoke" if args.smoke or args.cpu_fallback else args.stage
    if stage == "smoke" or args.cpu_fallback or not torch.cuda.is_available():
        if stage != "smoke" and not args.cpu_fallback and not torch.cuda.is_available():
            return refuse_full_training_on_cpu()
        return run_smoke_stage()

    # GPU path. Reached only when CUDA is available. Not taken in CI.
    import copy

    from posttrain.checkpoint import load_checkpoint, save_checkpoint
    from posttrain.configs import load_yaml
    from posttrain.dpo import train_steps as dpo_steps
    from posttrain.gold_pairs import gold_dpo_pairs, gold_sft_pairs
    from posttrain.grpo import train_steps as grpo_steps
    from posttrain.logging_utils import StepLogger
    from posttrain.lora import apply_lora, trainable_parameters
    from posttrain.rewards import reward as env_reward
    from posttrain.sft import train_steps as sft_steps
    from bioenvs import make_env

    cfg = load_yaml(args.config)
    print(f"config model={cfg['model_name']} status={cfg.get('status')} stage={stage}", flush=True)
    print("task metrics before this process: not run", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=dtype)
    lora_cfg = cfg.get("lora") or {}
    model = apply_lora(
        model,
        r=lora_cfg.get("r", 16),
        alpha=lora_cfg.get("alpha", 32),
        target_modules=lora_cfg.get("target_modules"),
    )
    opt = torch.optim.AdamW(trainable_parameters(model), lr=float(cfg.get("lr", 1e-4)))
    out_dir = Path(drive_dir or "outputs/posttrain")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model, opt)
        start = int(payload["step"])
        print(f"resumed at step {start}", flush=True)
    env_name = cfg.get("env_first", "err")
    steps = int(cfg.get("max_steps", 200))
    logger = StepLogger(stage)
    if stage == "sft":
        pairs = gold_sft_pairs(env_name, n=256, holdout_ids=set())
        model, opt = sft_steps(
            model, tok, pairs, steps=steps, lr=float(cfg.get("lr", 1e-4)), normalize="token",
            logger=logger, use_lora=False, optimizer=opt, max_len=1024,
        )
    elif stage == "dpo":
        rows = gold_dpo_pairs(env_name, n=256)
        ref_path = out_dir / "dpo_ref.pt"
        if not ref_path.exists():
            save_checkpoint(ref_path, model, None, step=start, extra={"role": "dpo reference, frozen"})
        ref = copy.deepcopy(model)
        ref.eval()
        model, opt = dpo_steps(
            model, ref, tok, rows, steps=steps, lr=float(cfg.get("lr", 1e-5)), beta=0.1, sft_coef=0.0,
            length_normalize=False, logger=logger, use_lora=False, optimizer=opt, max_len=1024,
        )
    elif stage in ("grpo", "ladder"):
        reward_names = ["ground_truth"] if stage == "grpo" else list(cfg.get("rewards_first", ["ground_truth", "format_only", "random_p50"]))
        env = make_env(env_name, split="train", seed=0)
        prompts, metas = [], []
        for i in range(min(128, len(env))):
            sample = env.sample(i)
            prompts.append(sample.prompt)
            metas.append({"env": env_name, "info": sample.info, "split": "train", "seed": 0})
        ref_path = out_dir / "grpo_ref.pt"
        if not ref_path.exists():
            save_checkpoint(ref_path, model, None, step=start, extra={"role": "grpo reference, frozen"})
        # Each reward starts from the same weights. Chaining them would mix the signals.
        start_state = copy.deepcopy(model.state_dict())
        ref = copy.deepcopy(model)
        ref.eval()
        for reward_name in reward_names:
            model.load_state_dict(start_state)
            opt = torch.optim.AdamW(trainable_parameters(model), lr=float(cfg.get("lr", 1e-6)))
            for meta in metas:
                meta["reward_name"] = reward_name
            print(f"grpo reward={reward_name} prompts={len(prompts)}", flush=True)
            model, opt = grpo_steps(
                model, ref, tok, prompts, env_reward, steps=steps, lr=float(cfg.get("lr", 1e-6)),
                logger=StepLogger(f"grpo-{reward_name}"), group_size=int(cfg.get("group_size", 4)),
                epsilon=float(cfg.get("clip_eps", 0.2)), kl_coef=float(cfg.get("kl_coef", 0.02)),
                use_clip=bool(cfg.get("use_clip", True)), std_norm=bool(cfg.get("std_norm", True)),
                loss_norm=cfg.get("loss_norm", "token"), max_new_tokens=int(cfg.get("max_new_tokens", 256)),
                use_lora=False, optimizer=opt, metas=metas, max_len=1024,
            )
            save_checkpoint(
                out_dir / f"grpo_{reward_name}.pt", model, opt, step=steps,
                extra={"reward_name": reward_name, "status": "trained on GPU in this process"},
            )
    else:
        print(f"unknown stage {stage}", flush=True)
        return 2
    save_checkpoint(out_dir / f"{stage}_step{start + steps}.pt", model, opt, step=start + steps, extra={"stage": stage, "env": env_name})
    print(f"saved checkpoint. task metrics vs a held-out gold set: not run yet. do that with posttrain.eval_suite.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
