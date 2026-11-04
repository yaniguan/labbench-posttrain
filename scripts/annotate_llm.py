"""Optional LLM annotations through the codex CLI (if installed). Output: data/interim/llm_annotations.jsonl
Each line: {"id", "safety_level", "failure_reason", "step_types"}; merged by build_dataset.py with
annotator="llm:codex/<model>", auto=True. Usage: python scripts/annotate_llm.py --n 40 --model gpt-6.1-sol"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
from labpt.io import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--model", default="gpt-6.1-sol")
ap.add_argument("--effort", default="high")
a = ap.parse_args()
if not shutil.which("codex"):
    sys.exit("codex CLI not found; skipping LLM annotations")

recs = read_jsonl(Path("data/interim/ord.jsonl")) + read_jsonl(Path("data/interim/messy.jsonl"))
recs = [r for r in recs if r.protocol_text and len(r.protocol_text) > 80]
random.seed(0)
batch = random.sample(recs, min(a.n, len(recs)))
items = [{"id": r.id, "text": r.protocol_text[:1200], "reported_outcome": r.outcome.model_dump()} for r in batch]
prompt = ("You annotate lab protocols / lab log lines for a dataset. For EACH item return an object "
          '{"id", "safety_level": "low|medium|high", "failure_reason": null or short snake_case tag '
          '(only if the text or reported outcome shows a failure), "step_types": list of short tags}. '
          "Return ONLY a JSON array, nothing else.\n\nITEMS:\n" + json.dumps(items, ensure_ascii=False))
with tempfile.TemporaryDirectory() as tmp:
    outf = Path(tmp) / "out.txt"
    subprocess.run(["codex", "exec", "--skip-git-repo-check", "-m", a.model, "-c", f"model_reasoning_effort={a.effort}",
                    "-s", "read-only", "-o", str(outf), prompt], stdin=subprocess.DEVNULL, cwd=tmp, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    txt = outf.read_text()
arr = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
ok_ids = {r.id for r in batch}
with open("data/interim/llm_annotations.jsonl", "w") as f:
    for x in arr:
        if x.get("id") in ok_ids:
            x["annotator"] = f"llm:codex/{a.model}"
            f.write(json.dumps(x) + "\n")
print("llm annotations:", len(arr))
