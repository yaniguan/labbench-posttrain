"""SFT with LoRA/QLoRA (TRL). Usage: python scripts/train_sft.py configs/operator/sft_smoke.yaml [key=value ...]"""
import json
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, "src")
from labpt.config import check_gpu, training_config, training_rows
from labpt.profiling import ProfileCallback

cfg = training_config('configs/scientist/sft_a100.yaml')
check_gpu(cfg)
rows = [json.loads(l) for l in open(cfg["train_file"])]
rows = [r for r in rows if r["task"] in cfg["tasks"]]
rows = training_rows(rows, cfg)
ds = Dataset.from_list([{"messages": r["prompt"] + [{"role": "assistant", "content": r["answer"]}]} for r in rows])
print("train examples", len(ds))

tok = AutoTokenizer.from_pretrained(cfg["model"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bf16 = cfg.get("bf16", False) and cfg.get("device") != "cpu" and torch.cuda.is_available()
quant = None
if cfg.get("qlora"):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16 if bf16 else torch.float32,
                                             quantization_config=quant, attn_implementation=cfg.get("attn", "sdpa"))
lora = LoraConfig(r=cfg.get("lora_r", 4 if cfg.get("smoke") else 16), lora_alpha=cfg.get("lora_alpha", 32), lora_dropout=0.05,
                  target_modules=cfg.get("lora_targets", "all-linear"), task_type="CAUSAL_LM")
args = SFTConfig(
    output_dir=cfg["output_dir"], max_steps=cfg["max_steps"], per_device_train_batch_size=cfg["batch_size"],
    gradient_accumulation_steps=cfg.get("grad_accum", 1), learning_rate=cfg.get("lr", 2e-4),
    logging_steps=1, save_strategy="no", report_to=[], bf16=bf16, max_length=cfg.get("max_len", 1024),
    gradient_checkpointing=cfg.get("grad_ckpt", False), include_num_input_tokens_seen=True,
    assistant_only_loss=False, completion_only_loss=None, packing=cfg.get("packing", False), use_cpu=cfg.get("device") == "cpu" or not torch.cuda.is_available(),
)
tr = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora,
                callbacks=[ProfileCallback(cfg["output_dir"])])
res = tr.train()
tr.save_model(cfg["output_dir"] + "/adapter")
json.dump({"train_loss": res.training_loss, "log": tr.state.log_history}, open(cfg["output_dir"] + "/train_metrics.json", "w"), indent=1)
print("final train loss", res.training_loss)
