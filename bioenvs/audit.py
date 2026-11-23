"""Scripted exploit policies and control-reward audit for all RLVR environments.

Usage:
    python -m bioenvs.audit [--json PATH] [--thresholds configs/audit_thresholds.yaml]

Reward hacking guard: each env ships policies that try to collect reward without doing the task. Re-run this
against any change to a reward function (or any checkpoint's behaviour), the numbers must stay below the
configured thresholds in `configs/audit_thresholds.yaml`.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any

import yaml

from bioenvs import make_env
from bioenvs.calisol import (
    best_constant_train_design_answer,
    design_oracle_share_stats,
    nearest_train_design_answer,
)
from common.run_meta import build_run_meta

DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "configs" / "audit_thresholds.yaml"


def _w(x: str) -> str:
    return f"[ANSWER_START]{x}[ANSWER_END]"


def _rng(s) -> random.Random:
    return random.Random(s.id)


def _rand_design_json(s) -> str:
    r = _rng(s)
    salt = r.choice(s.info["candidate_salts"])
    solvs = r.sample(s.info["candidate_solvents"], k=min(2, len(s.info["candidate_solvents"])))
    if len(solvs) == 1:
        s_map = {solvs[0]: 1.0}
    else:
        w0 = round(r.uniform(0.2, 0.8), 3)
        s_map = {solvs[0]: w0, solvs[1]: round(1.0 - w0, 3)}
    payload = {
        "salt": salt,
        "c": round(r.uniform(0.3, 2.5), 3),
        "c_units": s.info["c_units"],
        "solvent_ratio_type": s.info["solvent_ratio_type"],
        "solvents": s_map,
    }
    return _w(json.dumps(payload))


def _const_train_design_json(s) -> str:
    base = best_constant_train_design_answer()
    payload = {
        **base,
        "c_units": s.info["c_units"],
        "solvent_ratio_type": s.info["solvent_ratio_type"],
    }
    return _w(json.dumps(payload))


POLICIES = {
    "pqa": {
        "oracle": lambda s: _w(f"{s.info['answer']} & 100"),
        "random choice, conf 100": lambda s: _w(f"{_rng(s).choice(s.info['choices'])} & 100"),
        "random choice, honest conf 20": lambda s: _w(f"{_rng(s).choice(s.info['choices'])} & 20"),
        "always first choice": lambda s: _w(f"{s.info['choices'][0]} & 50"),
        "no answer tags": lambda s: s.info["answer"],
    },
    "ord": {
        "oracle": lambda s: _w(str(s.info["gold"])),
        "identity order": lambda s: _w(str(list(range(s.info["n"])))),
        "reversed gold": lambda s: _w(str(s.info["gold"][::-1])),
        "random order": lambda s: _w(str(random.Random(s.id).sample(range(s.info["n"]), s.info["n"]))),
    },
    "err": {
        "oracle": lambda s: _w("True" if s.info["is_correct"] else f"False || {s.info['gold']}"),
        "always True": lambda s: _w("True"),
        "always False, copy the step": lambda s: _w(f"False || {s.info['shown']}"),
        "always False, empty fix": lambda s: _w("False"),
    },
    "gen": {
        "oracle": lambda s: "\n".join(f"{i + 1}. {x}" for i, x in enumerate(s.info["ref"])),
        "boilerplate protocol": lambda s: "1. Prepare the samples.\n2. Add the reagents.\n3. Incubate.\n4. Wash.\n5. Analyze the results.",
        "echo the prompt": lambda s: s.prompt,
        "oracle padded 5x": lambda s: "\n".join(f"{i + 1}. {x}" for i, x in enumerate(s.info["ref"] + s.info["ref"][:1] * 5 * len(s.info["ref"]))),
        "empty": lambda s: "",
    },
    "calisol_design": {
        "oracle": lambda s: _w(json.dumps(s.info["oracle_formulation"])),
        "best constant train answer": _const_train_design_json,
        "random valid answer": _rand_design_json,
        "nearest train record": lambda s: _w(json.dumps(nearest_train_design_answer(s.info))),
    },
    "calisol_predict": {
        "oracle": lambda s: _w(json.dumps({"k_mS_cm": s.info["k_mS_cm"], "sigma_mS_cm": 0.005})),
        "constant k=5.0, sigma=2.0": lambda s: _w(json.dumps({"k_mS_cm": 5.0, "sigma_mS_cm": 2.0})),
        "random valid k, inflated sigma=30": lambda s: _w(
            json.dumps({"k_mS_cm": round(_rng(s).uniform(0.1, 25.0), 3), "sigma_mS_cm": 30.0})
        ),
        "no answer tags": lambda s: "5.0 mS/cm",
    },
}


def mean_reward(env, policy, n: int, use_mode: bool = False) -> float:
    f = env.reward if use_mode else env.score
    return st.mean(f(env.sample(i), policy(env.sample(i))).value for i in range(min(n, len(env))))


def audit(n: int = 200, *, split: str = "test") -> dict[str, dict[str, float]]:
    res: dict[str, dict[str, float]] = {}
    for name, pols in POLICIES.items():
        env = make_env(name, split=split, seed=1)
        res[name] = {p: mean_reward(env, f, n) for p, f in pols.items()}
    return res


def audit_modes(n: int = 200) -> dict[str, dict[tuple[str, str], float]]:
    res: dict[str, dict[tuple[str, str], float]] = {}
    for name, pols in POLICIES.items():
        row: dict[tuple[str, str], float] = {}
        for mode in ("true", "random", "format", "incorrect"):
            env = make_env(name, split="test", seed=1, reward_mode=mode)
            first_bad = next(k for k in pols if k not in ("oracle", "no answer tags"))
            for pol in ("oracle", first_bad):
                row[(mode, pol)] = mean_reward(env, pols[pol], n, use_mode=True)
        res[name] = row
    return res


def load_thresholds(path: str | Path = DEFAULT_THRESHOLDS_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def check_thresholds(
    scores: dict[str, dict[str, float]],
    thresholds_cfg: dict[str, Any],
) -> list[str]:
    """Return human-readable violation strings if any oracle < min_oracle or exploit > max."""
    violations: list[str] = []
    envs_cfg = thresholds_cfg.get("environments", thresholds_cfg)
    for env_name, rules in envs_cfg.items():
        if env_name not in scores:
            violations.append(f"Missing audited environment: {env_name}")
            continue
        env_scores = scores[env_name]
        oracle_rule = rules.get("oracle", {})
        min_oracle = float(oracle_rule["min"] if isinstance(oracle_rule, dict) else oracle_rule)
        oracle_val = float(env_scores.get("oracle", 0.0))
        if oracle_val < min_oracle:
            violations.append(
                f"{env_name}.oracle={oracle_val:.4f} < min={min_oracle:.4f}"
            )
        for pol_name, bound_spec in (rules.get("exploits") or {}).items():
            if pol_name not in env_scores:
                violations.append(f"Missing audited exploit policy: {env_name}.{pol_name!r}")
                continue
            max_val = float(bound_spec["max"] if isinstance(bound_spec, dict) else bound_spec)
            got = float(env_scores[pol_name])
            if got > max_val:
                violations.append(
                    f"{env_name}.{pol_name!r}={got:.4f} > max={max_val:.4f}"
                )
    return violations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run verifiable reward exploit & control-mode audit.")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--json", dest="json_path", type=str, default=None, help="Write JSON report to PATH.")
    ap.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Optional YAML thresholds file to enforce (exits non-zero on violation).",
    )
    args = ap.parse_args(argv)

    scores = audit(n=args.n, split="test")
    val_design_scores = {
        p: mean_reward(make_env("calisol_design", split="val", seed=1), f, args.n)
        for p, f in POLICIES["calisol_design"].items()
    }
    design_stats = {sp: design_oracle_share_stats(sp) for sp in ("train", "val", "test")}
    modes = audit_modes(n=args.n)

    for name, rows in scores.items():
        print(f"\n[{name}] true reward of exploit policies (test)")
        for p, v in rows.items():
            print(f"  {p:34s} {v:.3f}")
    print("\n[calisol_design] val reward & oracle deduplication stats")
    for p, v in val_design_scores.items():
        print(f"  val / {p:28s} {v:.3f}")
    for sp, st_d in design_stats.items():
        print(
            f"  {sp:5s}: raw_tasks={st_d['raw_task_count']} -> merged_tasks={st_d['merged_task_count']} "
            f"(distinct_oracles={st_d['distinct_oracles']}, share_per_oracle={st_d['merged_share_per_oracle']:.3f})"
        )

    print("\ncontrol modes: mean reward of the oracle / the first exploit policy")
    for name, rows in modes.items():
        print(f"  {name}: " + "  ".join(f"{m}[{p[:10]}]={v:.2f}" for (m, p), v in rows.items()))

    violations: list[str] = []
    if args.thresholds:
        t_cfg = load_thresholds(args.thresholds)
        violations = check_thresholds(scores, t_cfg)
        if violations:
            print("\nAUDIT THRESHOLD VIOLATIONS:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)

    if args.json_path:
        out_p = Path(args.json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        modes_json = {
            name: {f"{m}:{p}": val for (m, p), val in row.items()}
            for name, row in modes.items()
        }
        payload = {
            "meta": build_run_meta(reward_version="v3", extra={"n": args.n}),
            "scores_test": scores,
            "calisol_design_val": val_design_scores,
            "calisol_design_oracle_stats": design_stats,
            "control_modes": modes_json,
            "threshold_violations": violations,
        }
        out_p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
