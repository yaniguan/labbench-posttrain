"""Reward model (sequence classifier + LoRA) on preference pairs. Usage: python scripts/train_rm.py <config>"""
import json
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

sys.path.insert(0, "src")
from labpt.config import check_gpu, training_config, training_rows
from labpt.profiling import ProfileCallback

cfg = training_config('configs/scientist/rm_a100.yaml')
check_gpu(cfg)


def load_pairs(path, limit="max_train"):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["kind"] in cfg["kinds"]]
    rows = training_rows(rows, cfg, limit)
    return Dataset.from_list([{"chosen": r["prompt"] + [{"role": "assistant", "content": r["chosen"]}],
                               "rejected": r["prompt"] + [{"role": "assistant", "content": r["rejected"]}]} for r in rows])


ds = load_pairs(cfg["train_file"])
ds = ds.select(range(min(len(ds), cfg.get("max_train", 10**9))))
ev = load_pairs(cfg["eval_file"], "max_eval")
ev = ev.select(range(min(len(ev), cfg.get("max_eval", 10**9))))
tok = AutoTokenizer.from_pretrained(cfg["model"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bf16 = cfg.get("bf16", False) and cfg.get("device") != "cpu" and torch.cuda.is_available()
model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], num_labels=1, dtype=torch.bfloat16 if bf16 else torch.float32)
model.config.pad_token_id = tok.pad_token_id
args = RewardConfig(output_dir=cfg["output_dir"], max_steps=cfg["max_steps"], per_device_train_batch_size=cfg["batch_size"],
                    per_device_eval_batch_size=cfg["batch_size"], learning_rate=cfg.get("lr", 1e-4), logging_steps=1,
                    save_strategy="no", report_to=[], bf16=bf16, max_length=cfg.get("max_len", 1024), use_cpu=cfg.get("device") == "cpu" or not torch.cuda.is_available())
tr = RewardTrainer(model=model, args=args, train_dataset=ds, eval_dataset=ev, processing_class=tok,
                   peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="SEQ_CLS"),
                   callbacks=[ProfileCallback(cfg["output_dir"])])
res = tr.train()
m = tr.evaluate()
tr.save_model(cfg["output_dir"] + "/adapter")
json.dump({"train_loss": res.training_loss, "eval": m}, open(cfg["output_dir"] + "/train_metrics.json", "w"), indent=1)
print("eval", m)
