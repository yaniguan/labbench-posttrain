"""Simple profiling hooks. Saves profile/summary.json, profile/steps.jsonl and a torch profiler trace.

Use ProfileCallback with any HF/TRL Trainer, or StepTimer in a plain PyTorch loop.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


def gpu_mem():
    if torch.cuda.is_available():
        return {"gpu_mem_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                "gpu_name": torch.cuda.get_device_name(0)}
    return {"gpu_mem_alloc_gb": None, "gpu_mem_peak_gb": None, "gpu_name": None}


def make_torch_profiler(out_dir: Path, wait=1, warmup=1, active=2):
    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(p):
        p.export_chrome_trace(str(out_dir / "trace.json"))
        (out_dir / "top_ops.txt").write_text(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))

    return torch.profiler.profile(activities=acts, schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
                                  on_trace_ready=_save, record_shapes=False, profile_memory=True)


class StepTimer:
    """For plain loops: t = StepTimer(dir); t.start(); ...; t.mark('data'); ...; t.mark('fwd_bwd'); t.end(n_items)"""

    def __init__(self, out_dir, name="train"):
        self.dir = Path(out_dir) / "profile"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.f = open(self.dir / "steps.jsonl", "w")
        self.rows, self.name = [], name
        self.prof = make_torch_profiler(self.dir)
        self.prof.__enter__()

    def start(self):
        self.t0 = self.tl = time.perf_counter()
        self.parts = {}

    def mark(self, part):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.parts[part] = round(now - self.tl, 5)
        self.tl = now

    def end(self, n_items, **extra):
        dt = time.perf_counter() - self.t0
        row = {"step": len(self.rows), "step_time_s": round(dt, 5), "items_per_s": round(n_items / dt, 2),
               "breakdown_s": self.parts, **gpu_mem(), **extra}
        self.rows.append(row)
        self.f.write(json.dumps(row) + "\n")
        self.prof.step()

    def close(self):
        self.prof.__exit__(None, None, None)
        self.f.close()
        rows = self.rows[1:] or self.rows  # skip warmup step
        s = {"name": self.name, "steps": len(self.rows),
             "mean_step_time_s": round(sum(r["step_time_s"] for r in rows) / len(rows), 5),
             "mean_items_per_s": round(sum(r["items_per_s"] for r in rows) / len(rows), 2), **gpu_mem()}
        json.dump(s, open(self.dir / "summary.json", "w"), indent=1)
        return s


class ProfileCallback(TrainerCallback):
    """Records per-step wall time, tokens/s, GPU memory; torch profiler trace for a few steps."""

    def __init__(self, out_dir, trace_steps=(2, 4)):
        self.dir = Path(out_dir) / "profile"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trace_steps = trace_steps
        self.rows, self.prof, self.tokens = [], None, 0

    def on_step_begin(self, args, state, control, **kw):
        if state.global_step == self.trace_steps[0] and self.prof is None:
            self.prof = make_torch_profiler(self.dir, wait=0, warmup=0, active=self.trace_steps[1] - self.trace_steps[0])
            self.prof.__enter__()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self.t0
        ntok = getattr(state, "num_input_tokens_seen", 0) or 0
        row = {"step": state.global_step, "step_time_s": round(dt, 4),
               "tokens_per_s": round((ntok - self.tokens) / dt, 1) if ntok else None, **gpu_mem()}
        self.tokens = ntok
        self.rows.append(row)
        if self.prof is not None:
            self.prof.step()
            if state.global_step >= self.trace_steps[1]:
                self.prof.__exit__(None, None, None)
                self.prof = None
                self.trace_steps = (10**9, 10**9)

    def on_train_end(self, args, state, control, **kw):
        if self.prof is not None:
            self.prof.__exit__(None, None, None)
        with open(self.dir / "steps.jsonl", "w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
        rows = self.rows[1:] or self.rows
        tps = [r["tokens_per_s"] for r in rows if r["tokens_per_s"]]
        s = {"steps": len(self.rows), "mean_step_time_s": round(sum(r["step_time_s"] for r in rows) / max(len(rows), 1), 4),
             "mean_tokens_per_s": round(sum(tps) / len(tps), 1) if tps else None, **gpu_mem(),
             "torch": torch.__version__, "cuda": torch.version.cuda, "host": os.uname().nodename}
        json.dump(s, open(self.dir / "summary.json", "w"), indent=1)
