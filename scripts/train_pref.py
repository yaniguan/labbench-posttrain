"""Preference training: DPO or ORPO (LoRA). Usage: python scripts/train_pref.py <config.yaml> [loss=orpo]"""
import json
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from labpt.config import check_gpu, training_config, training_rows
from labpt.profiling import ProfileCallback

cfg = training_config('configs/scientist/dpo_a100.yaml')
check_gpu(cfg)
rows = [json.loads(l) for l in open(cfg["train_file"])]
rows = [r for r in rows if r["kind"] in cfg["kinds"]]
rows = training_rows(rows, cfg)
ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": [{"role": "assistant", "content": r["chosen"]}],
                         "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows])
print("pairs", len(ds))
tok = AutoTokenizer.from_pretrained(cfg["model"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bf16 = cfg.get("bf16", False) and cfg.get("device") != "cpu" and torch.cuda.is_available()
model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16 if bf16 else torch.float32)
lora = LoraConfig(r=cfg.get("lora_r", 4 if cfg.get("smoke") else 16), lora_alpha=cfg.get("lora_alpha", 32), target_modules="all-linear", task_type="CAUSAL_LM")
common = dict(output_dir=cfg["output_dir"], max_steps=cfg["max_steps"], per_device_train_batch_size=cfg["batch_size"],
              gradient_accumulation_steps=cfg.get("grad_accum", 1), learning_rate=cfg.get("lr", 5e-6), logging_steps=1,
              save_strategy="no", report_to=[], bf16=bf16, max_length=cfg.get("max_len", 1024), beta=cfg.get("beta", 0.1),
              gradient_checkpointing=cfg.get("grad_ckpt", False), use_cpu=cfg.get("device") == "cpu" or not torch.cuda.is_available())
if cfg.get("loss", "dpo") == "orpo":
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer
    args, Trainer = ORPOConfig(**common), ORPOTrainer
else:
    from trl import DPOConfig, DPOTrainer
    args, Trainer = DPOConfig(**common), DPOTrainer
tr = Trainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora,
             callbacks=[ProfileCallback(cfg["output_dir"])])
res = tr.train()
tr.save_model(cfg["output_dir"] + "/adapter")
json.dump({"train_loss": res.training_loss, "log": tr.state.log_history}, open(cfg["output_dir"] + "/train_metrics.json", "w"), indent=1)
print("final train loss", res.training_loss)
