"""Compile `lean/LabPostTrainLean.lean`, verify `#print axioms` on every theorem, and run `#eval` parity (#55).

1. Runs `lake build` in `lean/`.
2. Extracts every `theorem <name>` from `lean/LabPostTrainLean.lean` and checks `#print axioms`
   (failing on `sorryAx` or any non-standard axiom outside `{propext, Classical.choice, Quot.sound}`).
3. Generates `>= 250` seeded random test cases (default `300`) and evaluates all 7 mirrored
   functions in Lean via `#eval`, comparing 1-for-1 against the Python mirrors in `autolab_wire.py`.
4. Optional `--self-test-sorry`: injects a temporary theorem proved with `sorry` in a scratch
   Lean file and verifies that the axiom checker rejects `sorryAx`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAN_DIR = REPO_ROOT / "lean"
LEAN_FILE = LEAN_DIR / "LabPostTrainLean.lean"

for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autolab_wire import (  # noqa: E402
    check_gdpo_pareto_weights,
    check_kl_anchor,
    gated_gdpo_channels_permille,
    gated_reward_permille,
    gdpo_channel_zscore_scaled,
    group_centered_scaled,
    pareto_normalized_permille,
)

ALLOWED_STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
_THEOREM_RE = re.compile(r"^\s*(?:@\[[^\]]+\]\s*)?theorem\s+([A-Za-z0-9_']+)\b", re.MULTILINE)


def find_lake_executable() -> str | None:
    found = shutil.which("lake")
    if found:
        return found
    for cand in (
        Path.home() / ".elan" / "bin" / "lake.exe",
        Path.home() / ".elan" / "bin" / "lake",
    ):
        if cand.is_file():
            return str(cand)
    return None


def extract_theorem_names(lean_source: str) -> list[str]:
    names = _THEOREM_RE.findall(lean_source)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def parse_axiom_output(stdout: str, expected_theorems: list[str]) -> list[str]:
    """Parse `lake env lean` `#print axioms` lines and return any violations."""
    errors: list[str] = []
    seen_theorems: dict[str, set[str]] = {}

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if "does not depend on any axioms" in line:
            m = re.match(r"^'([^']+)'\s+does not depend on any axioms", line)
            if m:
                short = m.group(1).split(".")[-1]
                seen_theorems[short] = set()
        elif "depends on axioms:" in line:
            m = re.match(r"^'([^']+)'\s+depends on axioms:\s*\[([^\]]*)\]", line)
            if m:
                short = m.group(1).split(".")[-1]
                axioms = {a.strip() for a in m.group(2).split(",") if a.strip()}
                seen_theorems[short] = axioms

    for thm in expected_theorems:
        if thm not in seen_theorems:
            errors.append(f"Theorem {thm!r} missing from `#print axioms` output")
            continue
        axioms = seen_theorems[thm]
        if "sorryAx" in axioms:
            errors.append(f"Theorem {thm!r} depends on `sorryAx` (incomplete proof)")
        disallowed = sorted(axioms - ALLOWED_STANDARD_AXIOMS)
        if disallowed:
            errors.append(f"Theorem {thm!r} depends on non-standard axioms: {disallowed}")

    return errors


def _lean_bool(val: bool) -> str:
    return "true" if val else "false"


def _lean_nat_list(xs: list[int]) -> str:
    if not xs:
        return "([] : List Nat)"
    return "[" + ", ".join(str(int(x)) for x in xs) + "]"


def _lean_int_list(xs: list[int]) -> str:
    if not xs:
        return "([] : List Int)"
    items = [f"({int(x)} : Int)" for x in xs]
    return "[" + ", ".join(items) + "]"


def generate_random_cases(n_cases: int = 300, seed: int = 20261127) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    for idx in range(n_cases):
        core_ok = rng.choice([True, False])
        oracle_ok = rng.choice([True, False])
        max_think = rng.randint(0, 1024)
        # Mix in-bounds, exact-boundary, and over-budget think_tokens
        mode = idx % 4
        if mode == 0:
            think = max_think
        elif mode == 1:
            think = rng.randint(0, max(0, max_think))
        else:
            think = rng.randint(0, 1500)
        task_pm = rng.randint(0, 2000)
        fmt_pm = rng.randint(0, 500)

        # Channels and weights (include edge cases: 0/1 channels, mismatched lengths, all-zero weights)
        if idx % 10 == 0:
            n_ch = rng.choice([0, 1])
            n_w = n_ch
        elif idx % 10 == 1:
            n_ch = rng.randint(2, 5)
            n_w = n_ch + rng.choice([-1, 1])
        else:
            n_ch = rng.randint(2, 6)
            n_w = n_ch

        channels = [rng.randint(0, 1000) for _ in range(n_ch)]
        if idx % 10 == 2 and n_w > 0:
            weights = [0 for _ in range(n_w)]
        else:
            weights = [rng.randint(0, 25) for _ in range(n_w)]

        group_len = rng.randint(0, 8)
        group_rewards = [rng.randint(-1000, 1000) for _ in range(group_len)]
        spread_scale = 0 if (idx % 7 == 0) else rng.randint(1, 50)

        kl_coef_pm = 0 if (idx % 6 == 0) else rng.randint(1, 200)
        max_kl_pm = rng.randint(0, 500)
        kl_sft_pm = max_kl_pm if (idx % 8 == 0) else rng.randint(0, 700)
        sft_frozen = rng.choice([True, False])

        cases.append({
            "idx": idx,
            "core_ok": core_ok,
            "oracle_ok": oracle_ok,
            "think": think,
            "max_think": max_think,
            "task_pm": task_pm,
            "fmt_pm": fmt_pm,
            "channels": channels,
            "weights": weights,
            "group_rewards": group_rewards,
            "spread_scale": spread_scale,
            "kl_coef_pm": kl_coef_pm,
            "kl_sft_pm": kl_sft_pm,
            "max_kl_pm": max_kl_pm,
            "sft_frozen": sft_frozen,
        })
    return cases


def expected_python_outputs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in cases:
        out.append({
            "idx": c["idx"],
            "gated_reward": gated_reward_permille(
                core_scientific_verified=c["core_ok"],
                oracle_field_stripped=c["oracle_ok"],
                think_tokens=c["think"],
                max_think_tokens=c["max_think"],
                task_reward_permille=c["task_pm"],
                format_bonus_permille=c["fmt_pm"],
            ),
            "gdpo_weights_rule": check_gdpo_pareto_weights(c["channels"], c["weights"]) or "OK",
            "gated_channels": gated_gdpo_channels_permille(
                core_scientific_verified=c["core_ok"],
                oracle_field_stripped=c["oracle_ok"],
                think_tokens=c["think"],
                max_think_tokens=c["max_think"],
                channels=c["channels"],
            ),
            "pareto_norm": pareto_normalized_permille(c["weights"], c["channels"]),
            "group_centered": group_centered_scaled(c["group_rewards"]),
            "gdpo_zscore": gdpo_channel_zscore_scaled(c["group_rewards"], c["spread_scale"]),
            "kl_rule": check_kl_anchor(
                kl_coef_permille=c["kl_coef_pm"],
                kl_to_sft_permille=c["kl_sft_pm"],
                max_kl_drift_permille=c["max_kl_pm"],
                sft_anchor_frozen=c["sft_frozen"],
            ) or "OK",
        })
    return out


def build_lean_harness_source(theorems: list[str], cases: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "import LabPostTrainLean",
        "open LabPostTrainLean",
        "",
        "def optRule (o : Option RegimeViolation) : String :=",
        "  match o with",
        "  | none => \"OK\"",
        "  | some v => v.rule",
        "",
        "def natListJson (xs : List Nat) : String :=",
        "  \"[\" ++ String.intercalate \",\" (xs.map toString) ++ \"]\"",
        "",
        "def intListJson (xs : List Int) : String :=",
        "  \"[\" ++ String.intercalate \",\" (xs.map toString) ++ \"]\"",
        "",
    ]
    for thm in theorems:
        lines.append(f"#print axioms LabPostTrainLean.{thm}")
    lines.append("")

    eval_entries: list[str] = []
    for c in cases:
        r_spec = (
            f"{{ coreScientificVerified := {_lean_bool(c['core_ok'])}, "
            f"oracleFieldStripped := {_lean_bool(c['oracle_ok'])}, "
            f"thinkTokens := {c['think']}, maxThinkTokens := {c['max_think']}, "
            f"taskRewardPermille := {c['task_pm']}, formatBonusPermille := {c['fmt_pm']} }}"
        )
        g_spec = (
            f"{{ coreScientificVerified := {_lean_bool(c['core_ok'])}, "
            f"oracleFieldStripped := {_lean_bool(c['oracle_ok'])}, "
            f"thinkTokens := {c['think']}, maxThinkTokens := {c['max_think']}, "
            f"channels := {_lean_nat_list(c['channels'])}, weights := {_lean_nat_list(c['weights'])} }}"
        )
        k_spec = (
            f"{{ klCoefPermille := {c['kl_coef_pm']}, klToSftPermille := {c['kl_sft_pm']}, "
            f"maxKlDriftPermille := {c['max_kl_pm']}, sftAnchorFrozen := {_lean_bool(c['sft_frozen'])} }}"
        )
        ch_lit = _lean_nat_list(c["channels"])
        w_lit = _lean_nat_list(c["weights"])
        gr_lit = _lean_int_list(c["group_rewards"])
        entry = (
            f"\"PARITY|\" ++ toString {c['idx']} ++ \"|\" ++ "
            f"toString (gatedRewardPermille {r_spec}) ++ \"|\" ++ "
            f"optRule (checkGDPOParetoWeights {g_spec}) ++ \"|\" ++ "
            f"natListJson (gatedGDPOChannelsPermille {g_spec}) ++ \"|\" ++ "
            f"toString (paretoNormalizedPermille {w_lit} {ch_lit}) ++ \"|\" ++ "
            f"intListJson (groupCenteredScaled {gr_lit}) ++ \"|\" ++ "
            f"intListJson (gdpoChannelZScoreScaled {gr_lit} {c['spread_scale']}) ++ \"|\" ++ "
            f"optRule (checkKLAnchor {k_spec})"
        )
        eval_entries.append(f"  IO.println ({entry})")

    chunk_size = 25
    for chunk_idx in range(0, len(eval_entries), chunk_size):
        fn_name = f"runParityChunk{chunk_idx // chunk_size}"
        lines.append(f"def {fn_name} : IO Unit := do")
        lines.extend(eval_entries[chunk_idx : chunk_idx + chunk_size])
        lines.append(f"#eval {fn_name}")
        lines.append("")
    return "\n".join(lines)


def run_sorry_self_test(lake_exe: str) -> list[str]:
    """Verify that injecting `sorry` into a theorem causes `parse_axiom_output` to fail (#55)."""
    probe_file = LEAN_DIR / "_SorrySelfTestProbe.lean"
    try:
        probe_file.write_text(
            "theorem injected_sorry_probe : 1 + 1 = 3 := by sorry\n"
            "#print axioms injected_sorry_probe\n",
            encoding="utf-8",
        )
        res = subprocess.run(
            [lake_exe, "env", "lean", str(probe_file.name)],
            cwd=str(LEAN_DIR),
            check=False,
            capture_output=True,
            text=True,
        )
        combined = (res.stdout or "") + "\n" + (res.stderr or "")
        errs = parse_axiom_output(combined, ["injected_sorry_probe"])
        if not any("sorryAx" in e for e in errs):
            return [f"Self-test failed: injected `sorry` was NOT caught! Output: {combined!r}"]
        return []
    finally:
        probe_file.unlink(missing_ok=True)


def verify_lean_and_parity(*, n_cases: int = 300, seed: int = 20261127, self_test_sorry: bool = False) -> dict[str, Any]:
    if n_cases < 200:
        raise ValueError(f"Differential test requires at least 200 cases (#55), got {n_cases}")
    lake_exe = find_lake_executable()
    if lake_exe is None:
        return {"ok": False, "errors": ["`lake` executable not found on PATH or ~/.elan/bin"]}

    env = dict(os.environ)
    elan_bin = str(Path(lake_exe).parent)
    env["PATH"] = elan_bin + os.pathsep + env.get("PATH", "")

    build_res = subprocess.run(
        [lake_exe, "build"],
        cwd=str(LEAN_DIR),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if build_res.returncode != 0:
        return {
            "ok": False,
            "errors": [f"`lake build` failed (exit {build_res.returncode}):\n{build_res.stdout}\n{build_res.stderr}"],
        }

    theorems = extract_theorem_names(LEAN_FILE.read_text(encoding="utf-8"))
    if len(theorems) < 15:
        return {"ok": False, "errors": [f"Expected >= 15 theorems in LabPostTrainLean.lean, found {len(theorems)}"]}

    cases = generate_random_cases(n_cases=n_cases, seed=seed)
    py_expected = expected_python_outputs(cases)
    harness_file = LEAN_DIR / "_ParityAndAxiomsCheck.lean"
    errors: list[str] = []

    try:
        harness_file.write_text(build_lean_harness_source(theorems, cases), encoding="utf-8")
        eval_res = subprocess.run(
            [lake_exe, "env", "lean", str(harness_file.name)],
            cwd=str(LEAN_DIR),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        combined = (eval_res.stdout or "") + "\n" + (eval_res.stderr or "")
        if eval_res.returncode != 0:
            errors.append(f"`lake env lean` failed (exit {eval_res.returncode}):\n{combined}")
        else:
            errors.extend(parse_axiom_output(combined, theorems))

            lean_by_idx: dict[int, dict[str, Any]] = {}
            for line in (eval_res.stdout or "").splitlines():
                if not line.startswith("PARITY|"):
                    continue
                parts = line.strip().split("|")
                if len(parts) != 9:
                    errors.append(f"Malformed PARITY line: {line!r}")
                    continue
                idx = int(parts[1])
                lean_by_idx[idx] = {
                    "idx": idx,
                    "gated_reward": int(parts[2]),
                    "gdpo_weights_rule": parts[3],
                    "gated_channels": json.loads(parts[4]),
                    "pareto_norm": int(parts[5]),
                    "group_centered": json.loads(parts[6]),
                    "gdpo_zscore": json.loads(parts[7]),
                    "kl_rule": parts[8],
                }

            if len(lean_by_idx) != len(py_expected):
                errors.append(f"Expected {len(py_expected)} `#eval` parity rows, got {len(lean_by_idx)}")
            else:
                for exp in py_expected:
                    got = lean_by_idx.get(exp["idx"])
                    if got != exp:
                        errors.append(f"Parity mismatch at case {exp['idx']}: Lean={got} vs Python={exp}")
                        if len(errors) >= 10:
                            break
    finally:
        harness_file.unlink(missing_ok=True)

    if self_test_sorry:
        errors.extend(run_sorry_self_test(lake_exe))

    return {
        "ok": len(errors) == 0,
        "lake_executable": lake_exe,
        "theorems_checked": len(theorems),
        "theorem_names": theorems,
        "differential_cases": n_cases,
        "mirrored_functions": 7,
        "self_test_sorry": self_test_sorry,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compile Lean 4.36.0 kernel, audit `#print axioms`, and run `#eval` parity (#55).")
    ap.add_argument("--cases", type=int, default=300, help="Number of random differential cases (>= 200)")
    ap.add_argument("--seed", type=int, default=20261127, help="RNG seed for differential cases")
    ap.add_argument("--self-test-sorry", action="store_true", help="Verify injected `sorry` fails axiom audit")
    args = ap.parse_args(argv)

    report = verify_lean_and_parity(
        n_cases=args.cases,
        seed=args.seed,
        self_test_sorry=args.self_test_sorry,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
