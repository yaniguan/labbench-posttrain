"""Held-out eval on data/v0/tasks/sft_test.jsonl. Usage:
python scripts/eval_llm.py --model Qwen/Qwen2.5-0.5B-Instruct [--adapter outputs/.../adapter] --n 20 --out outputs/eval.json"""
import argparse
import collections
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from labpt import rewards

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter")
ap.add_argument("--file", default="data/v0/tasks/sft_test.jsonl")
ap.add_argument("--n", type=int, default=50, help="examples per task")
ap.add_argument("--max_new", type=int, default=192)
ap.add_argument("--out", default="outputs/eval.json")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval()

rows = [json.loads(l) for l in open(a.file)]
per = collections.defaultdict(list)
for r in rows:
    if len(per[r["task"]]) < a.n:
        per[r["task"]].append(r)

results, t0, ntok = {}, time.time(), 0
for task, exs in per.items():
    m = collections.defaultdict(float)
    for i in range(0, len(exs), 8):
        batch = exs[i:i + 8]
        texts = [tok.apply_chat_template(e["prompt"], tokenize=False, add_generation_prompt=True) for e in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ntok += int((out[:, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum())
        for e, g in zip(batch, gen):
            obj = rewards.parse_json(g)
            m["json_valid"] += obj is not None
            m["exact_match"] += obj == json.loads(e["answer"])
            m["reward"] += rewards.total_reward(g, task, e["answer"], e["meta"])
            if task == "protocol2actions":
                m["schema_valid"] += rewards.schema_valid_actions(g)
                m["action_f1"] += rewards.action_overlap(g, e["answer"])
            if task == "provenance":
                m["provenance_acc"] += rewards.provenance_correct(g, e["meta"]["provenance"])
                m["unit_consistent"] += rewards.unit_consistent(g, e["meta"]["property"], e["meta"]["units"])
            if task in ("outcome", "safety_failure", "property_qa") and isinstance(obj, dict):
                gold = json.loads(e["answer"])
                key = {"outcome": "success", "safety_failure": "safety_level", "property_qa": "provenance"}[task]
                m[f"{key}_acc"] += obj.get(key) == gold[key]
    results[task] = {k: round(v / len(exs), 4) for k, v in m.items()} | {"n": len(exs)}
dt = time.time() - t0
report = {"model": a.model, "adapter": a.adapter, "device": dev, "results": results,
          "gen_tokens_per_s": round(ntok / dt, 1), "seconds": round(dt, 1)}
import os
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(report, open(a.out, "w"), indent=1)
print(json.dumps(report, indent=1))
