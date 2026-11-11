"""Colab entry point. Do not run it here. Results are not run yet.

Order, cheapest first:
  1. CPU smoke (python -m posttrain.sft --smoke, then dpo, ppo, grpo).
  2. Qwen/Qwen2.5-0.5B LoRA SFT.
  3. DPO from that SFT checkpoint. beta 0.1. Reference is the SFT weights.
  4. GRPO from that SFT checkpoint. KL coefficient 0.01, estimator k3, loss term.
  5. The reward ladder and the KL grid, one family at a time.

Precision: a T4 does not run bf16. Use fp16 on a T4. bf16 is for A100, L4, and
newer. The reference model is a second copy of the SFT weights. On a 16 GB T4,
0.5B LoRA plus a frozen fp16 reference fits. A 7B model does not.

Checkpoints include the optimiser and the RNG. See posttrain.checkpoint.
Mount Drive and set OUTPUT_DIR under /content/drive so a dead runtime can resume.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Colab post-training runner. Not launched in CI.")
    p.add_argument("--stage", default="smoke", choices=["smoke", "sft", "dpo", "grpo", "ladder"])
    p.add_argument("--output-dir", default="/content/drive/MyDrive/labbench-posttrain/posttrain")
    p.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    args = p.parse_args(argv)
    print("not run yet")
    print(f"stage {args.stage} model {args.model} precision {args.precision}")
    print(f"output {args.output_dir}")
    print("KL stays on. The reference is the SFT checkpoint, not the raw pretrained model.")
    if args.precision == "bf16":
        print("bf16 needs a GPU that supports it. A T4 does not. Use fp16 on a T4.")
    out = Path(args.output_dir)
    print(f"Resume by passing the checkpoint in {out} to the trainer. The file holds weights, optimiser state, and RNG.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
