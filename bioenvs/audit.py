"""Scripted exploit policies and control-reward audit for the four envs.   python -m bioenvs.audit

Reward hacking guard: each env ships policies that try to collect reward without doing the task. Re-run this
against any change to a reward function (or any checkpoint's behaviour), the numbers must stay low.
The second table shows the control reward modes (random / format / incorrect) for an oracle and a random policy:
under a control mode the oracle must no longer stand out, otherwise the mode is leaking the true label.
"""
from __future__ import annotations

import random
import statistics as st

from bioenvs import make_env


def _w(x):
    return f"[ANSWER_START]{x}[ANSWER_END]"


def _rng(s):
    return random.Random(s.id)


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
}


def mean_reward(env, policy, n, use_mode=False):
    f = env.reward if use_mode else env.score
    return st.mean(f(env.sample(i), policy(env.sample(i))).value for i in range(min(n, len(env))))


def audit(n: int = 200):
    res = {}
    for name, pols in POLICIES.items():
        env = make_env(name, split="test", seed=1)
        res[name] = {p: mean_reward(env, f, n) for p, f in pols.items()}
    return res


def audit_modes(n: int = 200):
    res = {}
    for name, pols in POLICIES.items():
        row = {}
        for mode in ("true", "random", "format", "incorrect"):
            env = make_env(name, split="test", seed=1, reward_mode=mode)
            for pol in ("oracle", next(k for k in pols if k != "oracle")):
                row[(mode, pol)] = mean_reward(env, pols[pol], n, use_mode=True)
        res[name] = row
    return res


if __name__ == "__main__":
    for name, rows in audit().items():
        print(f"\n[{name}] true reward of exploit policies")
        for p, v in rows.items():
            print(f"  {p:34s} {v:.3f}")
    print("\ncontrol modes: mean reward of the oracle / the first exploit policy")
    for name, rows in audit_modes().items():
        print(f"  {name}: " + "  ".join(f"{m}[{p[:10]}]={v:.2f}" for (m, p), v in rows.items()))
