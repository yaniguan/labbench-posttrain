"""Run the actual trainers on tiny CPU configs; preserve failures and real metrics."""
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run_all():
    jobs = [
        ("scientist_encoders", "scientist/train_encoders.py", "encoders"),
        ("scientist_align", "scientist/train_align.py", "align"),
        ("scientist_projector", "scientist/train_projector.py", "projector"),
        ("scientist_distill", "scientist/distill.py", "distill"),
    ]
    for track in ("scientist", "operator"):
        jobs += [(f"{track}_{method}", f"scripts/{script}.py", method)
                 for method, script in [("sft", "train_sft"), ("dpo", "train_pref"), ("rm", "train_rm"), ("grpo", "train_grpo")]]
    jobs += [("operator_orpo", "scripts/train_pref.py", "orpo"),
             ("operator_bc", "operator/train_bc.py", "bc"),
             ("operator_wm", "operator/world_model.py", "wm")]
    report = {"device": "cpu", "model": "offline random GPT-2, 32 hidden / 1 layer / byte tokenizer",
              "note": "Pipeline smoke checks, not scientific or GPU performance results.", "runs": []}
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    for name, script, method in jobs:
        track = name.split("_")[0]
        run_dir = out / "smoke" / name
        run_dir.mkdir(parents=True, exist_ok=True)
        # Do not accept metrics left by a previous successful run.
        for old in ("metrics.json", "train_metrics.json"):
            (run_dir / old).unlink(missing_ok=True)
        for old_profile in run_dir.glob("**/profile"):
            shutil.rmtree(old_profile)
        cmd = [str(ROOT / ".venv/bin/python"), script, f"configs/{track}/{method}_smoke.yaml", "--smoke", "--output-dir", str(run_dir)]
        start = time.perf_counter()
        with open(run_dir / "run.log", "w") as log:
            try:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180, env=os.environ.copy())
                code, error = result.returncode, None
            except subprocess.TimeoutExpired:
                code, error = -1, "180 second timeout"
        row = {"name": name, "command": cmd[1:], "seconds": time.perf_counter() - start,
               "returncode": code, "error": error}
        metrics_file = next((run_dir / f for f in ("train_metrics.json", "metrics.json") if (run_dir / f).exists()), None)
        row["metrics"] = json.loads(metrics_file.read_text()) if metrics_file else None
        profiles = sorted(run_dir.glob("**/profile/summary.json"))
        row["profiles"] = {str(p.relative_to(run_dir)): json.loads(p.read_text()) for p in profiles}
        losses = [json.loads(line)["loss"] for p in sorted(run_dir.glob("**/profile/steps.jsonl"))
                  for line in p.read_text().splitlines() if "loss" in json.loads(line)]
        if row["metrics"] and "train_loss" in row["metrics"]:
            losses.append(row["metrics"]["train_loss"])
        row["losses"] = losses
        row["passed"] = code == 0 and bool(profiles) and bool(losses) and all(math.isfinite(v) for v in losses)
        report["runs"].append(row)
        report["passed"] = all(r["passed"] for r in report["runs"])
        (out / "smoke_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{name}: {'PASS' if row['passed'] else 'FAIL'} ({row['seconds']:.2f}s)", flush=True)
        if not row["passed"]:
            print((run_dir / "run.log").read_text()[-5000:], flush=True)
    return report


if __name__ == "__main__":
    os.chdir(ROOT)
    result = run_all()
    sys.exit(0 if result["passed"] else 1)
