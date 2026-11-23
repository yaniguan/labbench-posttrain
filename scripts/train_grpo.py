"""GRPO with verifiable rewards (schema-valid actions + ordered action reward, provenance + unit consistency).
Usage: python scripts/train_grpo.py <config>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if "src" not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labpt import rewards  # noqa: E402


def _text(c):
    if isinstance(c, list):
        if not c:
            return ""
        last = c[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return c or ""


# separate reward functions so each one is logged; they counter-balance each other
def r_schema(completions, task, **kw):
    return [rewards.schema_valid_actions(_text(c)) if t == "protocol2actions" else 0.0 for c, t in zip(completions, task)]


def r_overlap(completions, task, answer, **kw):
    return [rewards.ordered_action_reward(_text(c), a)["total"] if t == "protocol2actions" else 0.0 for c, t, a in zip(completions, task, answer)]


def r_provenance(completions, task, gold_prov, **kw):
    return [rewards.provenance_correct(_text(c), g) if t == "provenance" else 0.0 for c, t, g in zip(completions, task, gold_prov)]


def r_units(completions, task, prop, gold_units, **kw):
    return [0.5 * rewards.unit_consistent(_text(c), p or None, u or None) if t == "provenance" else 0.0
            for c, t, p, u in zip(completions, task, prop, gold_units)]


def main(argv: list[str] | None = None) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from labpt.config import check_gpu, training_config, training_rows
    from labpt.profiling import ProfileCallback

    cfg = training_config("configs/scientist/grpo_a100.yaml", argv=argv)
    check_gpu(cfg)
    with open(cfg["train_file"], encoding="utf-8") as _f:
        rows = [json.loads(line) for line in _f]
    rows = [r for r in rows if r["task"] in cfg["tasks"]]
    rows = training_rows(rows, cfg)
    ds = Dataset.from_list([
        {
            "prompt": r["prompt"],
            "task": r["task"],
            "answer": r["answer"],
            "gold_prov": r["meta"]["provenance"],
            "prop": r["meta"]["property"] or "",
            "gold_units": r["meta"]["units"] or "",
        }
        for r in rows
    ])

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bf16 = cfg.get("bf16", False) and cfg.get("device") != "cpu" and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16 if bf16 else torch.float32)
    args = GRPOConfig(
        output_dir=cfg["output_dir"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["batch_size"],
        num_generations=cfg.get("num_generations", 4),
        max_completion_length=cfg.get("max_completion", 256),
        learning_rate=cfg.get("lr", 1e-5),
        beta=cfg.get("kl_beta", 0.04),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=bf16,
        use_vllm=cfg.get("use_vllm", False),
        use_cpu=cfg.get("device") == "cpu" or not torch.cuda.is_available(),
        temperature=cfg.get("temperature", 1.0),
    )
    tr = GRPOTrainer(
        model=model,
        reward_funcs=[r_schema, r_overlap, r_provenance, r_units],
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
        callbacks=[ProfileCallback(cfg["output_dir"])],
    )
    res = tr.train()
    tr.save_model(cfg["output_dir"] + "/adapter")
    with open(cfg["output_dir"] + "/train_metrics.json", "w", encoding="utf-8") as _mf:
        json.dump({"train_loss": res.training_loss, "log": tr.state.log_history}, _mf, indent=1)
    print("done", res.training_loss)


if __name__ == "__main__":
    main()
