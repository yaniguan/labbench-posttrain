"""Wiring example: GRPO on one BioProBench env with TRL.   (NOT executed in this project - no GPU/time;
`env.reward_fn` itself is covered by tests/test_envs.py::test_gym_and_trl_interface.)

    pip install trl datasets
    python train/grpo_example.py --env err --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from bioenvs import make_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="err", choices=["pqa", "ord", "err", "gen"])
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--n", type=int, default=3000)
a = ap.parse_args()

env = make_env(a.env, split="train")
ds = Dataset.from_list([{"prompt": r["prompt"], "info": r["info"]} for r in env.dataset(a.n)])
trainer = GRPOTrainer(
    model=a.model,
    reward_funcs=env.reward_fn,           # extra dataset column `info` is forwarded to the reward fn as a kwarg
    args=GRPOConfig(output_dir=f"runs/{a.env}", num_generations=8, max_completion_length=1024,
                    per_device_train_batch_size=8, learning_rate=1e-6, logging_steps=5),
    train_dataset=ds,
)
trainer.train()
