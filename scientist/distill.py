"""Distillation: big teacher LLM -> small student LLM on our task texts (forward KL on logits + CE).
Teacher and student must share a tokenizer family (e.g. Qwen2.5-7B -> Qwen2.5-1.5B). Vocab sizes can differ
slightly (padding rows); we compare the shared first V logits.
Usage: python scientist/distill.py configs/scientist/distill_smoke.yaml"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from labpt.config import check_gpu, training_config
from labpt.profiling import StepTimer

cfg = training_config('configs/scientist/distill_a100.yaml')
check_gpu(cfg)
torch.manual_seed(0)
dev = "cpu" if cfg.get("device") == "cpu" else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (cfg.get("bf16") and dev == "cuda") else torch.float32
out = Path(cfg["output_dir"])
out.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(cfg["student"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
teacher = AutoModelForCausalLM.from_pretrained(cfg["teacher"], dtype=dtype).to(dev).eval().requires_grad_(False)
student = AutoModelForCausalLM.from_pretrained(cfg["student"], dtype=dtype).to(dev)
opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"])
rows = [json.loads(l) for l in open(cfg["train_file"])]
rows = [r for r in rows if r["task"] in cfg["tasks"]]
texts = [tok.apply_chat_template(r["prompt"] + [{"role": "assistant", "content": r["answer"]}], tokenize=False) for r in rows]
T, alpha = cfg.get("temperature", 2.0), cfg.get("alpha", 0.5)
timer = StepTimer(out, "distill")
for step in range(cfg["steps"]):
    timer.start()
    b = [texts[(step * cfg["batch_size"] + j) % len(texts)] for j in range(cfg["batch_size"])]
    enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=cfg.get("max_len", 512)).to(dev)
    timer.mark("data")
    with torch.no_grad():
        tl = teacher(**enc).logits
    timer.mark("teacher_fwd")
    so = student(**enc, labels=enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100))
    V = min(tl.shape[-1], so.logits.shape[-1])
    m = enc["attention_mask"].bool()
    kl = F.kl_div(F.log_softmax(so.logits[..., :V][m] / T, -1), F.log_softmax(tl[..., :V][m] / T, -1),
                  log_target=True, reduction="batchmean") * T * T
    loss = alpha * kl + (1 - alpha) * so.loss
    opt.zero_grad(); loss.backward(); opt.step()
    timer.mark("student_fwd_bwd_opt")
    timer.end(int(m.sum()), loss=round(loss.item(), 4), kl=round(kl.item(), 4), ce=round(so.loss.item(), 4))
prof = timer.close()
student.save_pretrained(out / "student")
res = {"teacher": cfg["teacher"], "student": cfg["student"], "steps": cfg["steps"],
       "first": {k: timer.rows[0][k] for k in ("loss", "kl", "ce")}, "last": {k: timer.rows[-1][k] for k in ("loss", "kl", "ce")},
       "profile": prof}
json.dump(res, open(out / "metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "profile"}))
