"""Single-turn, verifiable-reward RL environments built from BioProBench.

Every env exposes three views of the same thing:
  * gym-style    : obs = env.reset(); reward, done, info = env.step(completion)
  * functional   : env.score(sample, completion) -> Reward       (the true reward; pure, used by tests / eval)
                   env.reward(sample, completion) -> Reward       (honours reward_mode: true / random / format / incorrect)
  * trainer glue : env.dataset(n) -> rows with `prompt` + `info`;  env.reward_fn(completions, info=...)
                   is directly usable as a TRL `GRPOTrainer(reward_funcs=...)` function.

Reward is always in [0, 1]; `Reward.components` carries the individual terms so you can log or
re-weight them without touching the env.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "bioprobench"
ANSWER_RE = re.compile(r"\[ANSWER_START\](.*?)\[ANSWER_END\]", re.S)

REGISTRY: dict[str, type["BioEnv"]] = {}

# Reward modes. "true" is the real reward. The others are control arms (cf. Shao et al. 2026, "Spurious Rewards"):
# they carry little or no information about correctness, so any gain a policy shows with the true reward should be
# read as the gap to these. They are for analysis only, never for building a model.
MODES = ("true", "random", "format", "incorrect")
REWARD_VERSIONS = ("v1", "v2")
_DIAGNOSTIC_COMPONENT_KEYS = frozenset({"confidence", "true_reward", "mode_reward", "abstained"})


@dataclass
class Sample:
    id: str
    prompt: str
    info: dict  # ground truth + metadata (type / difficulty); never shown to the policy


@dataclass
class Reward:
    value: float
    parsed: bool                      # did the completion follow the required output format?
    components: dict[str, float] = field(default_factory=dict)

    def gdpo_channels(self) -> dict[str, float]:
        """Return multi-channel `reward/<name>` floats in [0, 1] for GDPO advantage estimation."""
        prefixed = {k: float(v) for k, v in self.components.items() if k.startswith("reward/")}
        if prefixed:
            return prefixed
        channels: dict[str, float] = {}
        for k, v in self.components.items():
            if k in _DIAGNOSTIC_COMPONENT_KEYS:
                continue
            # Sentinel -1.0 (e.g. ERR repair on uncorrupted items, GEN numeric when ref has no quantities)
            # maps to 0.0 so GDPO group normalization never sees out-of-range sentinel penalties.
            fv = float(v)
            channels[f"reward/{k}"] = 0.0 if fv < 0.0 else fv
        if len(channels) < 2:
            channels["reward/score"] = float(self.value)
            channels["reward/parsed"] = float(self.parsed)
        return channels


def strip_thinking(text: str) -> str:
    return text.split("</think>")[-1] if "</think>" in text else text


def extract_answer(text: str) -> str | None:
    """Last [ANSWER_START]...[ANSWER_END] block after any </think>; None if absent."""
    m = ANSWER_RE.findall(strip_thinking(text or ""))
    return m[-1].strip() if m else None


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def load_json(task: str, split: str):
    path = DATA_DIR / f"{task}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `bash scripts/get_bioprobench.sh` "
                                f"(train splits are on HuggingFace: BioProBench/BioProBench)")
    return json.loads(path.read_text())


class BioEnv:
    name: str = ""
    task: str = ""           # BioProBench file prefix

    def __init__(self, split: str = "train", seed: int = 0, filter_fn=None, max_items: int | None = None,
                 reward_mode: str = "true", random_p: float = 0.5, reward_version: str = "v2"):
        assert reward_mode in MODES, f"reward_mode must be one of {MODES}"
        assert reward_version in REWARD_VERSIONS, f"reward_version must be one of {REWARD_VERSIONS}"
        self.mode, self.random_p = reward_mode, random_p
        self.reward_version = reward_version
        self._coin = random.Random(f"coin:{seed}")
        rows = self.load_rows(split)
        rows = [r for r in rows if self.keep(r) and (filter_fn is None or filter_fn(r))]
        self.rng = random.Random(seed)
        self.rng.shuffle(rows)
        self.rows = rows[:max_items] if max_items else rows
        self._index = {r.get("id"): i for i, r in enumerate(self.rows)}
        self._cur: Sample | None = None
        self._i = 0

    # ----- subclass API -------------------------------------------------------------
    def load_rows(self, split: str) -> list[dict]:
        return load_json(self.task, split)

    def keep(self, row: dict) -> bool:
        return True

    def make_sample(self, row: dict) -> Sample:
        raise NotImplementedError

    def score(self, sample: Sample, completion: str) -> Reward:
        raise NotImplementedError

    def corrupt(self, info: dict, other: dict) -> dict:
        """Return `info` with a plausible but *wrong* label (used by reward_mode="incorrect").
        `other` is the info of a neighbouring prompt, for tasks where a wrong label has to come from elsewhere."""
        raise NotImplementedError

    def make_counterfactual(self, sample: Sample, other_row: dict) -> Sample:
        """Build a perturbed counterfactual Sample whose ground-truth shifts predictably."""
        other_sample = self.make_sample(other_row)
        return Sample(f"{sample.id}:cf", other_sample.prompt, other_sample.info)

    # ----- shared plumbing ----------------------------------------------------------
    def counterfactual_pair(self, i: int) -> tuple[Sample, Sample]:
        """Return (original_sample, counterfactual_sample) for counterfactual consistency evaluation."""
        idx = i % len(self.rows)
        orig = self.make_sample(self.rows[idx])
        other_row = self.rows[(idx + 1) % len(self.rows)]
        return orig, self.make_counterfactual(orig, other_row)
    def reward(self, sample: Sample, completion: str) -> Reward:
        """Reward under the configured mode. `score` is always the true reward."""
        true = self.score(sample, completion)
        if self.mode == "true":
            return true
        if self.mode == "format":
            v = float(true.parsed)
        elif self.mode == "random":
            v = float(self._coin.random() < self.random_p)
        else:  # incorrect: same verifier, wrong label
            idx = self._index.get(sample.info.get("id"))
            other = self.make_sample(self.rows[(idx + 1) % len(self.rows)]).info if idx is not None else sample.info
            wrong = Sample(sample.id, sample.prompt, self.corrupt(sample.info, other))
            r = self.score(wrong, completion)
            return Reward(r.value, r.parsed, {**r.components, "true_reward": true.value})
        return Reward(v, true.parsed, {"mode_reward": v, "true_reward": true.value})

    def __len__(self):
        return len(self.rows)

    def sample(self, i: int) -> Sample:
        return self.make_sample(self.rows[i % len(self.rows)])

    def reset(self) -> str:
        self._cur = self.sample(self._i)
        self._i += 1
        return self._cur.prompt

    def step(self, completion: str):
        assert self._cur is not None, "call reset() first"
        r = self.reward(self._cur, completion)
        info = {"parsed": r.parsed, **r.components, **{k: v for k, v in self._cur.info.items() if k in ("type", "id")}}
        self._cur = None
        return r.value, True, info

    def dataset(self, n: int | None = None) -> list[dict]:
        n = min(n or len(self), len(self))
        out = []
        for i in range(n):
            s = self.sample(i)
            out.append({"id": s.id, "prompt": s.prompt, "info": json.dumps(s.info)})
        return out

    @property
    def reward_fn(self):
        """TRL-style: f(completions, info=[json,...], **kw) -> list[float]."""
        def fn(completions, info, **kw):
            outs = []
            for c, inf in zip(completions, info):
                if isinstance(c, list):  # chat-format completion
                    c = c[-1]["content"]
                s = Sample(id="", prompt="", info=json.loads(inf) if isinstance(inf, str) else inf)
                outs.append(self.reward(s, c).value)
            return outs
        fn.__name__ = f"{self.name}_reward"
        return fn

    @property
    def multi_reward_fn(self):
        """GDPO-style: f(completions, info=[json,...], **kw) -> dict[str, list[float]]."""
        def fn(completions, info, **kw):
            rows_ch = []
            for c, inf in zip(completions, info):
                if isinstance(c, list):
                    c = c[-1]["content"]
                s = Sample(id="", prompt="", info=json.loads(inf) if isinstance(inf, str) else inf)
                rows_ch.append(self.reward(s, c).gdpo_channels())
            keys = sorted({k for ch in rows_ch for k in ch})
            return {k: [float(ch.get(k, 0.0)) for ch in rows_ch] for k in keys}
        fn.__name__ = f"{self.name}_multi_reward"
        return fn


def make_env(name: str, **kw) -> BioEnv:
    return REGISTRY[name](**kw)
