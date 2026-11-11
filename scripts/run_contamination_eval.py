"""Contamination eval. Eval only. Do not point this at a training run.

    python scripts/run_contamination_eval.py --demo

--demo fits a tiny random model on one toy sentence and prints the metrics.
That is a CPU check of the code. It is not a lab-model result.

    python scripts/run_contamination_eval.py --model NAME --labbench --limit 20

loads a Hugging Face model and scores LAB-Bench text. LAB-Bench is CC-BY-SA and is
eval only. This script has no training path. It has not been run on a real model here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Contamination checks. Eval only.")
    parser.add_argument("--demo", action="store_true", help="Tiny model, toy sentence, CPU.")
    parser.add_argument("--model", default="", help="Hugging Face model id. Not used by --demo. Not run in CI.")
    parser.add_argument("--labbench", action="store_true", help="Score public LAB-Bench text. Eval only.")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.demo or not args.model:
        from posttrain.contamination import toy_demo

        report = toy_demo()
        report["labbench"] = "not run" if not args.labbench else "refused without --model"
        print(json.dumps(report, indent=2, default=str), flush=True)
        return 0
    if not args.labbench:
        print("Pass --labbench to score questions, or --demo for the CPU check. Not run.", flush=True)
        return 0
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from posttrain.contamination import load_labbench_questions, prefix_report

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32 if not torch.cuda.is_available() else None)
    model.eval()
    questions = load_labbench_questions(limit=args.limit)
    texts = [q["text"][:1500] for q in questions]
    report = prefix_report(model, tok, texts, max_new_tokens=64)
    report["status"] = "computed on LAB-Bench text for eval only"
    report["n_questions"] = len(texts)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
