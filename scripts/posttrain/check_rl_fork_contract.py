"""CPU contract checker between `labbench-posttrain` and pinned `yaniguan/RL` (#52).

Verifies without GPU or heavy Ray/vLLM installation:
  (a) AST symbol check: every symbol imported from `nemo_rl` by `scripts/posttrain/run_nemo_rl.py`
      (and every stage entrypoint module / setup / train function) exists at the pinned commit.
  (b) AST call-contract check: `setup()` return-tuple unpacking -> `train_fn(...)` explicit
      keyword-argument mapping in `launch_nemo_rl`, plus loud `NotImplementedError` on `mpo_offline`.
  (c) Real Pydantic `MasterConfig` + `check_outdated_config` + `LengthPenaltyConfig` +
      `should_use_nemo_gym` + GDPO `reward_weights` validation across `sft`, `dpo`, `gspo_longcot`,
      `dapo`, and `gdpo_multireward` recipes, experiment YAMLs, and README commands.

Usage:
    python scripts/posttrain/check_rl_fork_contract.py --fork-src /path/to/RL
"""
from __future__ import annotations

import argparse
import ast
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any
import typing
import typing_extensions

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from labpt_nemo_rl.config_validation import compose_recipe, load_yaml  # noqa: E402
from labpt_nemo_rl.envs.bioprobench_env import BioProBenchEnv  # noqa: E402
from labpt_nemo_rl.envs.calisol_env import CALiSolEnv  # noqa: E402
from scripts.posttrain.run_nemo_rl import PINNED_RL_COMMIT, resolve_nemo_entrypoint  # noqa: E402


# ---------------------------------------------------------------------------
# Part (a): AST module & symbol resolution against `--fork-src`
# ---------------------------------------------------------------------------
def _resolve_module_file(fork_src: Path, module_name: str) -> Path | None:
    parts = module_name.split(".")
    pkg_init = fork_src.joinpath(*parts, "__init__.py")
    if pkg_init.is_file():
        return pkg_init
    mod_py = fork_src.joinpath(*parts[:-1], f"{parts[-1]}.py") if len(parts) > 1 else fork_src / f"{parts[0]}.py"
    if mod_py.is_file():
        return mod_py
    return None


def _module_exported_names(fork_src: Path, module_name: str, _seen: set[str] | None = None) -> set[str] | None:
    seen = _seen if _seen is not None else set()
    if module_name in seen:
        return set()
    seen.add(module_name)
    mod_file = _resolve_module_file(fork_src, module_name)
    if mod_file is None:
        return None
    tree = ast.parse(mod_file.read_text(encoding="utf-8"), filename=str(mod_file))
    names: set[str] = set()
    pkg_dir = mod_file.parent if mod_file.name == "__init__.py" else None

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                base_mod = node.module or ""
                if node.level > 0:
                    parent_parts = module_name.split(".")
                    if mod_file.name != "__init__.py":
                        parent_parts = parent_parts[:-1]
                    up = max(0, len(parent_parts) - (node.level - 1))
                    prefix = ".".join(parent_parts[:up])
                    full_sub = f"{prefix}.{base_mod}".strip(".")
                else:
                    full_sub = base_mod
                sub_names = _module_exported_names(fork_src, full_sub, seen)
                if sub_names:
                    names.update(sub_names)
            else:
                for alias in node.names:
                    names.add(alias.asname or alias.name)

    if pkg_dir is not None:
        for child in pkg_dir.iterdir():
            if child.is_file() and child.suffix == ".py" and child.name != "__init__.py":
                names.add(child.stem)
            elif child.is_dir() and (child / "__init__.py").is_file():
                names.add(child.name)
    return names


def check_ast_imports(fork_src: Path, launcher_path: Path | None = None) -> list[str]:
    """Verify via AST that every `nemo_rl` import and stage entrypoint in `run_nemo_rl.py` exists."""
    target = launcher_path or (REPO_ROOT / "scripts" / "posttrain" / "run_nemo_rl.py")
    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    errors: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("nemo_rl"):
            exported = _module_exported_names(fork_src, node.module)
            if exported is None:
                errors.append(f"Missing fork module {node.module!r} imported at {target.name}:{node.lineno}")
                continue
            for alias in node.names:
                if alias.name not in exported:
                    errors.append(
                        f"Missing symbol {alias.name!r} in fork module {node.module!r} "
                        f"(imported at {target.name}:{node.lineno})"
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("nemo_rl") and _resolve_module_file(fork_src, alias.name) is None:
                    errors.append(f"Missing fork module {alias.name!r} imported at {target.name}:{node.lineno}")

    for algo in ("sft", "dpo", "gspo_longcot"):
        entry = resolve_nemo_entrypoint(algo, {})
        exported = _module_exported_names(fork_src, entry["module"])
        if exported is None:
            errors.append(f"Missing stage module {entry['module']!r} for algo={algo!r}")
            continue
        for key in ("setup_fn", "train_fn"):
            sym = entry[key]
            if sym not in exported:
                errors.append(f"Missing {key} symbol {sym!r} in {entry['module']!r} for algo={algo!r}")
        if "MasterConfig" not in exported:
            errors.append(f"Missing MasterConfig in {entry['module']!r} for algo={algo!r}")

    return errors


# ---------------------------------------------------------------------------
# Part (b): AST `setup()` tuple -> `train_fn` named-argument contract check
# ---------------------------------------------------------------------------
def _fork_setup_and_train_specs(fork_src: Path, rel_path: str, train_fn_name: str) -> dict[str, Any]:
    mod_path = fork_src / rel_path
    tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
    setup_ret_len: int | None = None
    setup_ret_names: list[str] = []
    train_required: list[str] = []
    train_all: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "setup":
            # Look at return annotation tuple or final return tuple
            if isinstance(node.returns, ast.Subscript) and isinstance(node.returns.slice, ast.Tuple):
                setup_ret_len = len(node.returns.slice.elts)
            rets = [r.value for r in ast.walk(node) if isinstance(r, ast.Return) and isinstance(r.value, ast.Tuple)]
            if rets:
                last_tup = rets[-1]
                setup_ret_len = len(last_tup.elts)
                setup_ret_names = [elt.id if isinstance(elt, ast.Name) else ast.unparse(elt) for elt in last_tup.elts]
        elif isinstance(node, ast.FunctionDef) and node.name == train_fn_name:
            pos_args = [a.arg for a in node.args.args]
            n_defaults = len(node.args.defaults)
            req_pos = pos_args[: len(pos_args) - n_defaults] if n_defaults else pos_args
            req_kw = [
                a.arg
                for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if d is None and False  # kw_defaults None means no default in AST!
            ]
            for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if d is None:
                    req_kw.append(a.arg)
            train_required = req_pos + req_kw
            train_all = set(pos_args) | {a.arg for a in node.args.kwonlyargs}

    return {
        "setup_ret_len": setup_ret_len,
        "setup_ret_names": setup_ret_names,
        "train_required": train_required,
        "train_all": train_all,
    }


def check_setup_to_train_mapping(fork_src: Path, launcher_path: Path | None = None) -> list[str]:
    """Verify `launch_nemo_rl` unpacks `setup()` and calls `train_fn` with exact named arguments."""
    target = launcher_path or (REPO_ROOT / "scripts" / "posttrain" / "run_nemo_rl.py")
    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    errors: list[str] = []

    # 1. Verify mpo_offline fails loudly on the fork path
    try:
        resolve_nemo_entrypoint("mpo_offline", {"mpo": {}})
        errors.append("resolve_nemo_entrypoint('mpo_offline', ...) did not fail loudly with NotImplementedError")
    except NotImplementedError:
        pass

    launch_fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "launch_nemo_rl":
            launch_fn = node
            break
    if launch_fn is None:
        return ["Could not find `launch_nemo_rl` in run_nemo_rl.py"]

    stage_specs = {
        "sft": _fork_setup_and_train_specs(fork_src, "nemo_rl/algorithms/sft.py", "sft_train"),
        "dpo": _fork_setup_and_train_specs(fork_src, "nemo_rl/algorithms/dpo.py", "dpo_train"),
        "grpo": _fork_setup_and_train_specs(fork_src, "nemo_rl/algorithms/grpo.py", "grpo_train"),
    }
    # Expected setup() return-slot indices for grpo (whose setup delegates to _setup_impl returning 13-tuple)
    stage_specs["grpo"]["setup_ret_len"] = 13
    stage_specs["grpo"]["setup_ret_names"] = [
        "policy",
        "policy_generation",
        "nemo_gym",
        "cluster",
        "dataloader",
        "val_dataloader",
        "loss_fn",
        "logger",
        "checkpointer",
        "grpo_save_state",
        "master_config",
        "teacher_worker_groups",
        "alias_to_group_alias",
    ]

    def _inspect_block(stmts: list[ast.stmt], stage: str) -> None:
        spec = stage_specs[stage]
        unpacked_vars: list[str] | None = None
        train_calls: list[ast.Call] = []
        for sub in ast.walk(ast.Module(body=stmts, type_ignores=[])):
            if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call):
                if isinstance(sub.value.func, ast.Name) and sub.value.func.id == "setup_fn":
                    if len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Tuple):
                        unpacked_vars = [
                            e.id if isinstance(e, ast.Name) else ast.unparse(e) for e in sub.targets[0].elts
                        ]
                    else:
                        errors.append(f"Stage {stage!r}: `setup_fn(...)` result is not unpacked into explicit variables")
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "train_fn":
                train_calls.append(sub)

        if unpacked_vars is None:
            errors.append(f"Stage {stage!r}: missing explicit tuple unpack from `setup_fn(...)`")
        elif spec["setup_ret_len"] is not None and len(unpacked_vars) != spec["setup_ret_len"]:
            errors.append(
                f"Stage {stage!r}: `setup_fn` unpacks {len(unpacked_vars)} values, "
                f"but fork `setup()` returns {spec['setup_ret_len']}"
            )

        if not train_calls:
            errors.append(f"Stage {stage!r}: no `train_fn(...)` call found")
            return

        for call in train_calls:
            if call.args:
                errors.append(
                    f"Stage {stage!r}: `train_fn` called with positional/starred args at line {call.lineno}; "
                    "must use explicit named keyword arguments"
                )
                continue
            kw_map = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
            if len(kw_map) != len(call.keywords):
                errors.append(f"Stage {stage!r}: `train_fn` uses `**kwargs` unpacking at line {call.lineno}")
            missing_req = [p for p in spec["train_required"] if p not in kw_map]
            if missing_req:
                errors.append(f"Stage {stage!r}: `train_fn` missing required named parameters {missing_req}")
            unknown_kw = sorted(set(kw_map.keys()) - spec["train_all"])
            if unknown_kw:
                errors.append(f"Stage {stage!r}: `train_fn` passes unknown keyword parameters {unknown_kw}")

            # Verify cluster is never bound to any train_fn parameter and slot names align
            if unpacked_vars and len(unpacked_vars) == len(spec["setup_ret_names"]):
                local_to_slot = dict(zip(unpacked_vars, spec["setup_ret_names"]))
                for param_name, val_node in kw_map.items():
                    if isinstance(val_node, ast.Name) and val_node.id in local_to_slot:
                        slot_name = local_to_slot[val_node.id]
                        if slot_name == "cluster":
                            errors.append(
                                f"Stage {stage!r}: `cluster` from `setup()` erroneously passed as `{param_name}`"
                            )
                        if param_name == "wrapped_dataloader" and slot_name != "dataloader":
                            errors.append(
                                f"Stage {stage!r}: `{param_name}` bound to `{slot_name}` instead of `dataloader`"
                            )
                        elif param_name != "wrapped_dataloader" and param_name in spec["setup_ret_names"]:
                            if slot_name != param_name:
                                errors.append(
                                    f"Stage {stage!r}: `{param_name}` bound to `{slot_name}` instead of `{param_name}`"
                                )

    sft_checked = dpo_checked = grpo_checked = False
    for stmt in launch_fn.body:
        if isinstance(stmt, ast.If):
            cond_src = ast.unparse(stmt.test)
            if "'sft'" in cond_src or '"sft"' in cond_src:
                _inspect_block(stmt.body, "sft")
                sft_checked = True
            elif "'dpo'" in cond_src or '"dpo"' in cond_src:
                _inspect_block(stmt.body, "dpo")
                dpo_checked = True
    # Remaining tail of launch_fn is the GRPO path
    tail_stmts = [
        s for s in launch_fn.body if not (isinstance(s, ast.If) and ("'sft'" in ast.unparse(s.test) or '"sft"' in ast.unparse(s.test) or "'dpo'" in ast.unparse(s.test) or '"dpo"' in ast.unparse(s.test)))
    ]
    _inspect_block(tail_stmts, "grpo")
    grpo_checked = True
    if not (sft_checked and dpo_checked and grpo_checked):
        errors.append(f"Incomplete stage branches in launch_nemo_rl (sft={sft_checked}, dpo={dpo_checked}, grpo={grpo_checked})")

    return errors


# ---------------------------------------------------------------------------
# Part (c): Real Fork Pydantic Config Validation on CPU
# ---------------------------------------------------------------------------
class _AttrStub:
    def __init__(self, name: str = "stub") -> None:
        self._name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _AttrStub(self._name)

    def __getattr__(self, key: str) -> Any:
        return _AttrStub(f"{self._name}.{key}")

    def __getitem__(self, key: Any) -> Any:
        return Any

    def __or__(self, other: Any) -> Any:
        return Any

    def __ror__(self, other: Any) -> Any:
        return Any

    def __mro_entries__(self, bases: tuple[type, ...]) -> tuple[type, ...]:
        return (object,)


class _ForkDepStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Stub uninstalled GPU/cluster runtime deps imported by `nemo_rl` while keeping real Pydantic models."""

    def __init__(self, fork_dir: str) -> None:
        self._fork_dir = os.path.normcase(os.path.abspath(fork_dir))
        self._roots: set[str] = {"tensorboard"}

    def _from_fork(self) -> bool:
        frame = sys._getframe(1)
        while frame is not None:
            fn = os.path.normcase(frame.f_code.co_filename or "")
            if "site-packages" in fn:
                return False
            if fn.startswith(self._fork_dir):
                return True
            frame = frame.f_back
        return False

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> importlib.machinery.ModuleSpec | None:
        root = fullname.split(".")[0]
        if root == "nemo_rl":
            return None
        if path is None and self._from_fork():
            self._roots.add(root)
        if root in self._roots:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        mod = types.ModuleType(spec.name)
        mod.__path__ = []  # type: ignore[attr-defined]
        mod.__version__ = "2.18.0"  # type: ignore[attr-defined]

        def _getattr(key: str, _n: str = spec.name) -> Any:
            if key.startswith("__") and key.endswith("__"):
                raise AttributeError(key)
            return _AttrStub(f"{_n}.{key}")

        mod.__getattr__ = _getattr  # type: ignore[attr-defined]
        return mod

    def exec_module(self, module: types.ModuleType) -> None:
        return None


def load_fork_config_validators(fork_src: Path) -> dict[str, Any]:
    """Import real `MasterConfig` & pre-flight validators from `--fork-src` on CPU."""
    if sys.version_info < (3, 12):
        typing.TypedDict = typing_extensions.TypedDict  # type: ignore[misc,assignment]

    import packaging.version as _pv
    import transformers as _tf

    tf_ver = str(_tf.__version__)
    orig_ver_cls = _pv.Version

    class _CompatVersion(orig_ver_cls):  # type: ignore[misc,valid-type]
        def __init__(self, version: str) -> None:
            if version == tf_ver and orig_ver_cls(version) >= orig_ver_cls("5.20.0"):
                version = "5.19.0"
            super().__init__(version)

    fork_str = str(fork_src.resolve())
    if fork_str not in sys.path:
        sys.path.insert(0, fork_str)
    finder = _ForkDepStubFinder(fork_str)
    sys.meta_path.append(finder)
    _pv.Version = _CompatVersion  # type: ignore[assignment]
    try:
        from nemo_rl.algorithms.dpo import MasterConfig as DPOMasterConfig  # type: ignore
        from nemo_rl.algorithms.grpo import (  # type: ignore
            MasterConfig as GRPOMasterConfig,
            _raise_if_length_penalty_enabled_without_nemo_gym,
            _raise_if_reward_penalties_enabled_without_nemo_gym,
        )
        from nemo_rl.algorithms.sft import MasterConfig as SFTMasterConfig  # type: ignore
        from nemo_rl.environments.nemo_gym import should_use_nemo_gym  # type: ignore
        from nemo_rl.utils.outdated_config_checks import check_outdated_config  # type: ignore
    finally:
        _pv.Version = orig_ver_cls  # type: ignore[assignment]

    return {
        "sft": SFTMasterConfig,
        "dpo": DPOMasterConfig,
        "grpo": GRPOMasterConfig,
        "check_outdated_config": check_outdated_config,
        "should_use_nemo_gym": should_use_nemo_gym,
        "check_length_penalty_gym": _raise_if_length_penalty_enabled_without_nemo_gym,
        "check_reward_penalties_gym": _raise_if_reward_penalties_enabled_without_nemo_gym,
    }


_ENV_CHANNEL_CACHE: dict[tuple[str, str], list[str]] = {}


def _env_sorted_reward_channels(domain: str, cfg: dict[str, Any]) -> list[str]:
    env_block = cfg.get("env") or {}
    sub = dict(env_block.get("calisol") or env_block.get("bioprobench") or {})
    cache_key = (domain, str(sub.get("channel_schema", "unified")))
    if cache_key in _ENV_CHANNEL_CACHE:
        return _ENV_CHANNEL_CACHE[cache_key]
    if domain == "calisol_conductivity" or "calisol" in env_block:
        env = CALiSolEnv(dict(env_block.get("calisol") or {}), max_items_per_task=2)
    else:
        env = BioProBenchEnv(dict(env_block.get("bioprobench") or {}), max_items_per_task=2)
    rows = env.build_dataset(n_per_task=1, include_counterfactuals=False)
    msgs = [[{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": "[ANSWER_START]ABSTAIN[ANSWER_END]"}] for r in rows]
    metas = [r["metadata"] for r in rows]
    step_out = env.step(msgs, metas)
    assert isinstance(step_out.rewards, dict)
    chans = sorted(k for k in step_out.rewards.keys() if k.startswith("reward/"))
    _ENV_CHANNEL_CACHE[cache_key] = chans
    return chans


def check_composed_recipes_against_fork(fork_src: Path) -> list[str]:
    """Validate all composed recipes, experiment configs, and GDPO channel weights against fork classes."""
    v = load_fork_config_validators(fork_src)
    errors: list[str] = []

    matrix: list[tuple[str, str, str, str, list[str]]] = []
    for algo in ("sft", "dpo", "gspo_longcot", "dapo", "gdpo_multireward"):
        for domain in ("bioenvs_pqa_ord_err", "bioenvs_gen", "calisol_conductivity"):
            for hw, model in (("local_1gpu_colocated", "qwen3_1.7b"), ("node_8xa100_colocated", "qwen3_8b")):
                matrix.append((algo, domain, model, hw, []))

    # README 1-GPU GSPO command with `--set grpo.max_num_steps=1`
    matrix.append(("gspo_longcot", "bioenvs_pqa_ord_err", "qwen3_1.7b", "local_1gpu_colocated", ["grpo.max_num_steps=1"]))

    # Experiment configs under configs/posttrain/experiments/*.yaml
    exp_dir = REPO_ROOT / "configs" / "posttrain" / "experiments"
    for exp_file in sorted(exp_dir.glob("*.yaml")):
        exp_data = load_yaml(exp_file)
        rec = exp_data.get("recipe")
        if isinstance(rec, dict) and rec.get("algo"):
            matrix.append((
                str(rec["algo"]),
                str(rec["domain"]),
                str(rec["model"]),
                str(rec["hardware"]),
                [],
            ))

    seen: set[tuple[str, str, str, str, tuple[str, ...]]] = set()
    for algo, domain, model, hw, ov in matrix:
        key = (algo, domain, model, hw, tuple(ov))
        if key in seen:
            continue
        seen.add(key)
        label = f"{algo}:{domain}:{model}:{hw}" + (f":{ov}" if ov else "")
        try:
            composed = compose_recipe(algo=algo, domain=domain, model=model, hardware=hw, overrides=ov)
            export_cfg = {k: val for k, val in composed.items() if k != "recipe_meta"}
            # Ensure YAML round-trip matches what run_nemo_rl writes
            export_cfg = yaml.safe_load(yaml.safe_dump(export_cfg, sort_keys=False))
            v["check_outdated_config"](export_cfg)
            stage = "sft" if algo == "sft" else "dpo" if algo == "dpo" else "grpo"
            mc = v[stage](**export_cfg)
            if stage == "grpo":
                enable_gym = bool(v["should_use_nemo_gym"](mc))
                v["check_length_penalty_gym"](mc, enable_nemo_gym=enable_gym)
                v["check_reward_penalties_gym"](mc, enable_nemo_gym=enable_gym)
                if algo == "gdpo_multireward":
                    env_channels = _env_sorted_reward_channels(domain, export_cfg)
                    rw = mc.grpo.adv_estimator.reward_weights
                    w_map = (export_cfg.get("grpo") or {}).get("adv_estimator", {}).get("gdpo_channel_weights") or {}
                    expected_rw = [float(w_map[ch]) for ch in env_channels if ch in w_map]
                    if rw is None:
                        errors.append(f"{label}: GDPO `reward_weights` is None (weights silently dropped)")
                    elif len(rw) != len(env_channels) or list(rw) != expected_rw:
                        errors.append(
                            f"{label}: GDPO `reward_weights` {rw} does not match sorted env channels "
                            f"{env_channels} (expected {expected_rw})"
                        )
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    return errors


def verify_fork_commit(fork_src: Path, *, allow_unpinned: bool = False) -> list[str]:
    if not (fork_src / "nemo_rl").is_dir():
        return [f"--fork-src {fork_src} does not contain a `nemo_rl/` directory"]
    if allow_unpinned or not (fork_src / ".git").exists():
        return []
    res = subprocess.run(
        ["git", "-C", str(fork_src), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return []
    actual = res.stdout.strip()
    if actual != PINNED_RL_COMMIT:
        return [f"Fork HEAD {actual} does not match PINNED_RL_COMMIT {PINNED_RL_COMMIT}"]
    return []


def run_all_checks(fork_src: Path, *, allow_unpinned: bool = False) -> dict[str, Any]:
    commit_errors = verify_fork_commit(fork_src, allow_unpinned=allow_unpinned)
    if commit_errors and not (fork_src / "nemo_rl").is_dir():
        return {"ok": False, "errors": commit_errors}
    ast_errors = check_ast_imports(fork_src)
    mapping_errors = check_setup_to_train_mapping(fork_src)
    recipe_errors = check_composed_recipes_against_fork(fork_src)
    all_errors = commit_errors + ast_errors + mapping_errors + recipe_errors
    return {
        "ok": len(all_errors) == 0,
        "pinned_rl_commit": PINNED_RL_COMMIT,
        "fork_src": str(fork_src.resolve()),
        "counts": {
            "commit_errors": len(commit_errors),
            "ast_import_errors": len(ast_errors),
            "setup_train_mapping_errors": len(mapping_errors),
            "recipe_schema_errors": len(recipe_errors),
            "total_errors": len(all_errors),
        },
        "errors": all_errors,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate labbench-posttrain contract against pinned yaniguan/RL source (#52).")
    ap.add_argument("--fork-src", type=Path, required=True, help="Path to yaniguan/RL checkout at PINNED_RL_COMMIT")
    ap.add_argument("--allow-unpinned", action="store_true", help="Skip exact git HEAD SHA match check")
    args = ap.parse_args(argv)

    report = run_all_checks(args.fork_src, allow_unpinned=args.allow_unpinned)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
