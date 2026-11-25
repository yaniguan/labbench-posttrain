"""Zero-leakage `SplitGuard`, `ScientificCorpus` protocol, and built-in corpora (`calisol23`, `bioprobench`, `lab_v0`)."""
from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from bioenvs import make_env
from bioenvs.calisol import TEST_DOIS, VAL_DOIS, load_clean_calisol_records
from labrl.core.registry import CORPUS_REGISTRY, register_corpus
from labrl.core.types import SplitLeakageError
from posttrain.hacking_checks import CANARY_NEEDLES, normalize_and_scan_canary


@dataclass(frozen=True)
class SplitGuard:
    """Cryptographically fingerprinted holdout & canary guard shared across pipeline stages."""

    group_key: str
    val_groups: frozenset[str] = frozenset()
    test_groups: frozenset[str] = frozenset()

    @property
    def signature(self) -> str:
        payload = json.dumps(
            {
                "group_key": self.group_key,
                "val": sorted(self.val_groups),
                "test": sorted(self.test_groups),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def assert_disjoint(self, train_groups: Iterable[str]) -> None:
        t_set = set(train_groups)
        overlap_val = t_set & self.val_groups
        overlap_test = t_set & self.test_groups
        val_test = self.val_groups & self.test_groups
        if overlap_val or overlap_test or val_test:
            raise SplitLeakageError(
                f"Holdout leakage detected on `{self.group_key}`: "
                f"train&val={sorted(overlap_val)}, train&test={sorted(overlap_test)}, val&test={sorted(val_test)}"
            )

    @staticmethod
    def assert_no_canary(texts: Iterable[str]) -> None:
        for txt in texts:
            reason = normalize_and_scan_canary(txt or "")
            if reason is not None:
                raise SplitLeakageError(
                    f"LAB-Bench post-cutoff canary string detected in training data ({reason})!"
                )


class ScientificCorpus(ABC):
    """Base protocol deriving leak-free SFT, DPO, MPO, and online RL views from one scientific dataset."""

    name: str = "base"

    @property
    @abstractmethod
    def split_guard(self) -> SplitGuard:
        raise NotImplementedError

    @abstractmethod
    def to_sft(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def to_preference(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def to_mpo(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def to_rl_prompts(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        raise NotImplementedError

    def default_env_name(self) -> str:
        return self.name


def _fmt_calisol_design(form: dict[str, Any], t_k: float, k_val: float, *, high: bool) -> str:
    solv = form["solvents"]
    solv_str = "+".join(sorted(solv.keys()))
    note = (
        f"yields high ionic dissociation and fast transport (~{k_val:.2f} mS/cm)"
        if high
        else f"suffers from higher viscosity or ion pairing (~{k_val:.2f} mS/cm)"
    )
    body = json.dumps(
        {
            "salt": form["salt"],
            "c": round(float(form["c"]), 4),
            "c_units": form["c_units"],
            "solvent_ratio_type": form["solvent_ratio_type"],
            "solvents": {s: round(float(w), 4) for s, w in solv.items()},
        },
        separators=(", ", ": "),
    )
    return (
        f" <think>At {t_k:.1f} K, {form['salt']} ({form['c']:.2f} {form['c_units']}) in {solv_str} {note}.</think>"
        f"[ANSWER_START]{body}[ANSWER_END]"
    )


@register_corpus("calisol23")
class CALiSolCorpus(ScientificCorpus):
    """CALiSol-23 experimental ionic conductivity corpus (`13,228` clean rows across `27` DOIs)."""

    def __init__(self, *, task_mode: str = "both") -> None:
        self.task_mode = task_mode
        records = load_clean_calisol_records()
        train_dois = {r["doi"] for r in records if r["split"] == "train"}
        self._guard = SplitGuard(group_key="doi", val_groups=VAL_DOIS, test_groups=TEST_DOIS)
        self._guard.assert_disjoint(train_dois)

    @property
    def split_guard(self) -> SplitGuard:
        return self._guard

    def default_env_name(self) -> str:
        return "calisol_conductivity"

    def _check_train_split(self, split: str) -> None:
        if split != "train":
            records = load_clean_calisol_records()
            sp_dois = {r["doi"] for r in records if r["split"] == split}
            self._guard.assert_disjoint(sp_dois or [split])

    def to_sft(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        self._check_train_split(split)
        cap = limit or 128
        rows: list[dict[str, Any]] = []
        if self.task_mode in ("both", "design"):
            d_env = make_env("calisol_design", split=split, seed=seed)
            n_d = cap // 2 if self.task_mode == "both" else cap
            for i in range(min(max(1, n_d), len(d_env))):
                s = d_env.sample(i)
                rows.append({
                    "prompt": s.prompt,
                    "completion": _fmt_calisol_design(
                        s.info["oracle_formulation"], float(s.info["target_T_K"]), float(s.info["k_max"]), high=True
                    ),
                    "task": "calisol_design",
                })
        if self.task_mode in ("both", "predict") and len(rows) < cap:
            p_env = make_env("calisol_predict", split=split, seed=seed)
            rem = cap - len(rows)
            for i in range(min(rem, len(p_env))):
                s = p_env.sample(i)
                k_v = float(s.info["k_mS_cm"])
                sig_v = round(max(0.15, 0.10 * k_v), 3)
                ans = json.dumps({"k_mS_cm": round(k_v, 4), "sigma_mS_cm": sig_v}, separators=(", ", ": "))
                comp = f" <think>VTF transport at {s.info['T_K']:.1f} K gives k ~ {k_v:.2f} mS/cm.</think>[ANSWER_START]{ans}[ANSWER_END]"
                rows.append({"prompt": s.prompt, "completion": comp, "task": "calisol_predict"})
        self._guard.assert_no_canary(r["prompt"] + r["completion"] for r in rows)
        return rows[:cap]

    def to_preference(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        self._check_train_split(split)
        cap = limit or 128
        d_env = make_env("calisol_design", split=split, seed=seed)
        pairs: list[dict[str, Any]] = []
        for i in range(len(d_env)):
            s = d_env.sample(i)
            t_k = float(s.info["target_T_K"])
            ordered = sorted(s.info["atlas_records"], key=lambda r: float(r["k_mS_cm"]), reverse=True)
            if len(ordered) < 2:
                continue
            hi, lo = ordered[0], ordered[-1]
            k_hi, k_lo = float(hi["k_mS_cm"]), float(lo["k_mS_cm"])
            if k_hi < 1.5 * max(k_lo, 1e-4) or (k_hi - k_lo) < 1.5:
                continue
            c_form = {
                "salt": hi["salt"], "c": hi["c"], "c_units": s.info["c_units"],
                "solvent_ratio_type": s.info["solvent_ratio_type"], "solvents": hi["solvents"],
            }
            r_form = {
                "salt": lo["salt"], "c": lo["c"], "c_units": s.info["c_units"],
                "solvent_ratio_type": s.info["solvent_ratio_type"], "solvents": lo["solvents"],
            }
            pairs.append({
                "prompt": s.prompt,
                "chosen": _fmt_calisol_design(c_form, t_k, k_hi, high=True),
                "rejected": _fmt_calisol_design(r_form, t_k, k_lo, high=False),
                "record_type": "paired",
                "k_chosen_mS_cm": k_hi,
                "k_rejected_mS_cm": k_lo,
            })
            if len(pairs) >= cap:
                break
        self._guard.assert_no_canary(p["prompt"] + p["chosen"] + p["rejected"] for p in pairs)
        return pairs

    def to_mpo(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        self._check_train_split(split)
        cap = limit or 64
        half = max(2, cap // 2)
        paired = self.to_preference(split=split, limit=half, seed=seed)
        d_env = make_env("calisol_design", split=split, seed=seed)
        unpaired: list[dict[str, Any]] = []
        for i in range(len(d_env)):
            s = d_env.sample(i)
            t_k = float(s.info["target_T_K"])
            unpaired.append({
                "prompt": s.prompt,
                "completion": _fmt_calisol_design(s.info["oracle_formulation"], t_k, float(s.info["k_max"]), high=True),
                "label": 1,
                "record_type": "unpaired",
            })
            unpaired.append({
                "prompt": s.prompt,
                "completion": _fmt_calisol_design(s.info["worst_formulation"], t_k, float(s.info["k_min"]), high=False),
                "label": 0,
                "record_type": "unpaired",
            })
            if len(unpaired) >= half:
                break
        mixed: list[dict[str, Any]] = []
        for idx in range(max(len(paired), len(unpaired))):
            if idx < len(paired):
                mixed.append(paired[idx])
            if idx < len(unpaired):
                mixed.append(unpaired[idx])
        return mixed[:cap]

    def to_rl_prompts(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        from labpt_nemo_rl.envs.calisol_env import CALiSolEnv

        cap = limit or 32
        tasks = ["calisol_design", "calisol_predict"] if self.task_mode == "both" else [f"calisol_{self.task_mode}"]
        n_per = max(1, (cap + len(tasks) - 1) // len(tasks))
        env = CALiSolEnv(tasks=tasks, split=split, seed=seed, max_items_per_task=max(4, n_per))
        rows = env.build_dataset(n_per_task=n_per, include_counterfactuals=False)[:cap]
        self._guard.assert_no_canary(r["prompt"] for r in rows)
        return rows


@register_corpus("bioprobench")
class BioProBenchCorpus(ScientificCorpus):
    """BioProBench wet-lab protocol reasoning corpus (`PQA`, `ORD`, `ERR`, `GEN`)."""

    def __init__(self, *, tasks: tuple[str, ...] = ("pqa", "ord", "err")) -> None:
        self.tasks = tuple(tasks)
        self._guard = SplitGuard(
            group_key="bioprobench_split",
            val_groups=frozenset({"val"}),
            test_groups=frozenset({"test"}),
        )

    @property
    def split_guard(self) -> SplitGuard:
        return self._guard

    def default_env_name(self) -> str:
        return "bioprobench"

    @staticmethod
    def _gold_completion(task: str, info: dict[str, Any]) -> tuple[str, str]:
        if task == "pqa":
            ans = str(info["answer"])
            wrong = next((c for c in info["choices"] if c != ans), "unknown")
            return (
                f" <think>Protocol parameter check.</think>[ANSWER_START]{ans} & 95[ANSWER_END]",
                f" <think>Guessing without verification.</think>[ANSWER_START]{wrong} & 99[ANSWER_END]",
            )
        if task == "ord":
            gold = list(info["gold"])
            rev = list(reversed(gold))
            return (
                f" <think>Verify prerequisite step order.</think>[ANSWER_START]{gold}[ANSWER_END]",
                f" <think>Arbitrary order.</think>[ANSWER_START]{rev}[ANSWER_END]",
            )
        if task == "err":
            ok = bool(info["is_correct"])
            good = "True" if ok else f"False || {info['gold']}"
            bad = "False || unchanged" if ok else "True"
            return (
                f" <think>Audit step parameters.</think>[ANSWER_START]{good}[ANSWER_END]",
                f" <think>Skip audit.</think>[ANSWER_START]{bad}[ANSWER_END]",
            )
        steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(info["ref"]))
        return steps, "1. Prepare samples.\n2. Incubate."

    def to_sft(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        if split != "train":
            self._guard.assert_disjoint([split])
        cap = limit or 64
        per = max(1, (cap + len(self.tasks) - 1) // len(self.tasks))
        rows: list[dict[str, Any]] = []
        for t in self.tasks:
            env = make_env(t, split=split, seed=seed, max_items=per)
            for i in range(len(env)):
                s = env.sample(i)
                good, _ = self._gold_completion(t, s.info)
                rows.append({"prompt": s.prompt, "completion": good, "task": t})
        self._guard.assert_no_canary(r["prompt"] + r["completion"] for r in rows)
        return rows[:cap]

    def to_preference(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        if split != "train":
            self._guard.assert_disjoint([split])
        cap = limit or 64
        per = max(1, (cap + len(self.tasks) - 1) // len(self.tasks))
        pairs: list[dict[str, Any]] = []
        for t in self.tasks:
            env = make_env(t, split=split, seed=seed, max_items=per)
            for i in range(len(env)):
                s = env.sample(i)
                good, bad = self._gold_completion(t, s.info)
                pairs.append({"prompt": s.prompt, "chosen": good, "rejected": bad, "record_type": "paired"})
        self._guard.assert_no_canary(p["prompt"] + p["chosen"] + p["rejected"] for p in pairs)
        return pairs[:cap]

    def to_mpo(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        if split != "train":
            self._guard.assert_disjoint([split])
        cap = limit or 32
        half = max(2, cap // 2)
        paired = self.to_preference(split=split, limit=half, seed=seed)
        unpaired: list[dict[str, Any]] = []
        for p in paired[: max(1, half // 2)]:
            unpaired.append({"prompt": p["prompt"], "completion": p["chosen"], "label": 1, "record_type": "unpaired"})
            unpaired.append({"prompt": p["prompt"], "completion": p["rejected"], "label": 0, "record_type": "unpaired"})
        return (paired + unpaired)[:cap]

    def to_rl_prompts(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        from labpt_nemo_rl.envs.bioprobench_env import BioProBenchEnv

        cap = limit or 32
        per = max(1, (cap + len(self.tasks) - 1) // len(self.tasks))
        env = BioProBenchEnv(tasks=list(self.tasks), split=split, seed=seed, max_items_per_task=max(4, per))
        rows = env.build_dataset(n_per_task=per, include_counterfactuals=False)[:cap]
        self._guard.assert_no_canary(r["prompt"] for r in rows)
        return rows


@register_corpus("lab_v0")
class LabV0Corpus(ScientificCorpus):
    """Multimodal & robotic operator trajectories from `data/v0`."""

    def __init__(self, *, data_dir: str | Path | None = None) -> None:
        self.root = Path(data_dir) if data_dir else (Path(__file__).resolve().parents[3] / "data" / "v0")
        self._guard = SplitGuard(group_key="entity_split", val_groups=frozenset({"val"}), test_groups=frozenset({"test"}))

    @property
    def split_guard(self) -> SplitGuard:
        return self._guard

    def default_env_name(self) -> str:
        return "lab_v0"

    @staticmethod
    def _msg_to_str(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt)

    def _load_sft_pool(self, split: str, *, pool_cap: int = 512) -> list[dict[str, Any]]:
        path = self.root / "tasks" / f"sft_{split}.jsonl"
        rows: list[dict[str, Any]] = []
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    msgs = obj.get("messages", [])
                    task = str(obj.get("task", "protocol2actions"))
                    meta = dict(obj.get("meta") or {})
                    if len(msgs) >= 2:
                        ans = str(msgs[-1].get("content", ""))
                        rows.append({
                            "prompt": self._msg_to_str(msgs[:-1]),
                            "completion": " " + ans,
                            "task": task,
                            "info": {"task": task, "answer": ans, "meta": meta},
                        })
                    elif "prompt" in obj and "answer" in obj:
                        ans = str(obj["answer"])
                        rows.append({
                            "prompt": self._msg_to_str(obj["prompt"]),
                            "completion": " " + ans,
                            "task": task,
                            "info": {"task": task, "answer": ans, "meta": meta},
                        })
                    if len(rows) >= pool_cap:
                        break
        return rows

    def to_sft(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        if split != "train":
            self._guard.assert_disjoint([split])
        cap = limit or 64
        rows = self._load_sft_pool(split, pool_cap=max(cap * 4, 256))
        rng = random.Random(f"lab_v0:sft:{split}:{seed}")
        rng.shuffle(rows)
        rows = rows[:cap]
        self._guard.assert_no_canary(r["prompt"] + r["completion"] for r in rows)
        return rows

    def to_preference(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        if split != "train":
            self._guard.assert_disjoint([split])
        cap = limit or 64
        path = self.root / "tasks" / f"pairs_{split}.jsonl"
        pairs: list[dict[str, Any]] = []
        pool_cap = max(cap * 4, 256)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    pairs.append({
                        "prompt": self._msg_to_str(obj["prompt"]),
                        "chosen": " " + str(obj["chosen"]),
                        "rejected": " " + str(obj["rejected"]),
                        "record_type": "paired",
                    })
                    if len(pairs) >= pool_cap:
                        break
        rng = random.Random(f"lab_v0:pref:{split}:{seed}")
        rng.shuffle(pairs)
        pairs = pairs[:cap]
        self._guard.assert_no_canary(p["prompt"] + p["chosen"] + p["rejected"] for p in pairs)
        return pairs

    def to_mpo(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        if split != "train":
            self._guard.assert_disjoint([split])
        cap = limit or 64
        half = max(2, cap // 2)
        pairs = self.to_preference(split=split, limit=half, seed=seed)
        sft_rows = self.to_sft(split=split, limit=max(1, half // 2), seed=seed)
        neg_pairs = self.to_preference(split=split, limit=max(2, half), seed=seed + 17)
        unpaired: list[dict[str, Any]] = []
        for idx in range(max(len(sft_rows), len(neg_pairs))):
            if idx < len(sft_rows):
                r = sft_rows[idx]
                unpaired.append({
                    "prompt": r["prompt"],
                    "completion": r["completion"],
                    "label": 1,
                    "record_type": "unpaired",
                })
            if idx < len(neg_pairs):
                np_row = neg_pairs[idx]
                unpaired.append({
                    "prompt": np_row["prompt"],
                    "completion": np_row["rejected"],
                    "label": 0,
                    "record_type": "unpaired",
                })
            if len(unpaired) >= half:
                break
        mixed: list[dict[str, Any]] = []
        for idx in range(max(len(pairs), len(unpaired))):
            if idx < len(pairs):
                mixed.append(pairs[idx])
            if idx < len(unpaired):
                mixed.append(unpaired[idx])
        out = mixed[:cap]
        self._guard.assert_no_canary(
            r["prompt"] + r.get("completion", r.get("chosen", "") + r.get("rejected", ""))
            for r in out
        )
        return out

    def to_rl_prompts(self, *, split: str = "train", limit: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
        cap = limit or 16
        rows = self._load_sft_pool(split, pool_cap=max(cap * 4, 128))
        rng = random.Random(f"lab_v0:rl:{split}:{seed}")
        rng.shuffle(rows)
        rows = rows[:cap]
        self._guard.assert_no_canary(r["prompt"] for r in rows)
        return [
            {
                "prompt": r["prompt"],
                "task": r["task"],
                "info": r["info"],
                "metadata": {"task": r["task"], "info": r["info"]},
            }
            for r in rows
        ]


def make_corpus(name_or_obj: str | ScientificCorpus, **kwargs: Any) -> ScientificCorpus:
    if isinstance(name_or_obj, ScientificCorpus):
        return name_or_obj
    if name_or_obj not in CORPUS_REGISTRY:
        raise KeyError(f"Unknown corpus {name_or_obj!r}; registered: {sorted(CORPUS_REGISTRY)}")
    return CORPUS_REGISTRY[name_or_obj](**kwargs)


from labrl.data.open_datasets_2026 import (  # noqa: E402
    EvalOnlyBenchmarkItem,
    OPEN_SCIENCE_DATASETS_2026,
    OpenScience2026Corpus,
    OpenScience2026Env,
    OpenScienceDatasetSpec,
    PartitionedOpenDataset,
    assert_trainable_dataset,
    compute_open_science_group_key,
    get_open_dataset_spec,
    list_open_datasets_2026,
    normalize_eval_only_item,
    normalize_open_science_record,
    partition_open_science_records,
)

__all__ = [
    "BioProBenchCorpus",
    "CALiSolCorpus",
    "EvalOnlyBenchmarkItem",
    "LabV0Corpus",
    "OPEN_SCIENCE_DATASETS_2026",
    "OpenScience2026Corpus",
    "OpenScience2026Env",
    "OpenScienceDatasetSpec",
    "PartitionedOpenDataset",
    "ScientificCorpus",
    "SplitGuard",
    "assert_trainable_dataset",
    "compute_open_science_group_key",
    "get_open_dataset_spec",
    "list_open_datasets_2026",
    "make_corpus",
    "normalize_eval_only_item",
    "normalize_open_science_record",
    "partition_open_science_records",
]
