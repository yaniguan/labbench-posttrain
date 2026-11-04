import yaml


def load(path, overrides=None):
    cfg = yaml.safe_load(open(path))
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def check_gpu(cfg):
    """A100-only policy: refuse to run a GPU config on anything that is not an A100."""
    import torch
    if cfg.get("device") == "a100":
        assert torch.cuda.is_available(), "This config needs a CUDA A100 (Colab A100)."
        name = torch.cuda.get_device_name(0)
        assert "A100" in name, f"A100-only config, got {name}"


def training_config(default, argv=None):
    """All trainers accept [config.yaml] [key=value ...] --smoke [--output-dir]."""
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=default)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir")
    args, overrides = parser.parse_known_args(argv)
    cfg = load(args.config, overrides)
    cfg["smoke"] = args.smoke
    if args.smoke:
        from .tiny import ensure_tiny_model
        torch.set_num_threads(1)
        cfg.update(device="cpu", bf16=False, qlora=False, grad_ckpt=False,
                   steps=4, max_steps=4, batch_size=2, grad_accum=1,
                   max_train=8, max_eval=4, max_len=512, max_completion=8,
                   num_generations=2, use_vllm=False, hidden=32, d=32)
        if "model" in cfg:
            cfg["model"] = ensure_tiny_model()
        if "teacher" in cfg:
            cfg["teacher"], cfg["student"] = ensure_tiny_model(True), ensure_tiny_model()
        cfg["output_dir"] = cfg["output_dir"].replace("/a100/", "/smoke/")
        # Never load a full-size encoder into the tiny projector.
        if "encoder_ckpt" in cfg:
            cfg["encoder_ckpt"] = "outputs/smoke/scientist_encoders/mol_encoder.pt"
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    return cfg


def training_rows(rows, cfg, limit="max_train"):
    """Smoke uses the shortest complete real examples, with answers preserved."""
    if cfg.get("smoke"):
        def size(row):
            prompt = sum(len(m["content"].encode()) + 16 for m in row["prompt"])
            answer = max(len(row.get(k, "").encode()) for k in ("answer", "chosen", "rejected"))
            return prompt + answer + 16
        rows = sorted((row for row in rows if size(row) <= cfg["max_len"]), key=size)
    rows = rows[:cfg.get(limit, 10**9)]
    if not rows:
        raise ValueError("no training rows fit this config")
    return rows
