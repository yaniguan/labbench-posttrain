"""Plot proxy reward against KL to the SFT model.

Reads a JSONL log written by posttrain.metrics.StepLogger. Does not train.
If the file is missing, it prints 'not run' and writes nothing.

The points are whatever the log contains. This script does not invent a curve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, golds = [], [], []
    for row in rows:
        kl = row.get("kl_to_sft")
        reward = row.get("proxy_reward", row.get("reward_mean"))
        if kl is None or reward is None:
            continue
        xs.append(float(kl))
        ys.append(float(reward))
        if row.get("gold_metric") is not None:
            golds.append(float(row["gold_metric"]))
    if not xs:
        raise SystemExit("not run: the log has no kl_to_sft and reward columns")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, label="proxy reward")
    if len(golds) == len(xs):
        ax.scatter(xs, golds, marker="x", label="held-out gold")
    ax.set_xlabel("KL to the SFT model (per token)")
    ax.set_ylabel("Reward")
    ax.set_title("Reward versus KL to the SFT model")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Plot reward against KL to the SFT reference.")
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/posttrain/reward_vs_kl.png"))
    args = p.parse_args(argv)
    if not args.log.exists():
        print("not run: no log file")
        return 0
    rows = load_rows(args.log)
    if not rows:
        print("not run: empty log")
        return 0
    plot(rows, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
