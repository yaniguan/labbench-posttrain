"""Reward sanity: oracle ~1, degenerate / hacking policies low, random in between.   pytest -q"""
import json
import random
import statistics as st

import pytest

from bioenvs import make_env

N = 300


def mean_reward(env, policy, n=N):
    vals = []
    for i in range(min(n, len(env))):
        s = env.sample(i)
        vals.append(env.score(s, policy(s)).value)
    return st.mean(vals)


def wrap(x):
    return f"[ANSWER_START]{x}[ANSWER_END]"


# ---------------------------------------------------------------- PQA
def test_pqa_parse_tolerates_delimiters_not_content():
    from bioenvs.pqa import parse
    ch = ["0.3", "0.5", "Pour", "centrifugation"]
    assert parse("0.3 & 80", ch) == ("0.3", 0.8)
    assert parse("0.3, 80", ch) == ("0.3", 0.8)
    assert parse("Pour 85", ch) == ("Pour", 0.85)
    assert parse("0.3", ch) == ("0.3", None)          # choice that ends in a number is not split
    assert parse("0.4 & 80", ch)[0] is None            # not an option
    assert parse("Pour", ch) == ("Pour", None)


def test_pqa():
    env = make_env("pqa", split="test", seed=1)
    oracle = lambda s: wrap(f"{s.info['answer']} & 100")
    rnd = lambda s: wrap(f"{random.Random(s.id).choice(s.info['choices'])} & 100")
    assert mean_reward(env, oracle) == pytest.approx(1.0)
    assert mean_reward(env, rnd) < 0.35                        # ~20% right, confidently wrong -> ~0.16
    assert mean_reward(env, lambda s: "") == 0
    assert mean_reward(env, lambda s: wrap("not a choice & 100")) == 0   # tags alone are not enough
    # calibration: same accuracy, honest 20% confidence beats claiming 100%
    assert mean_reward(env, lambda s: wrap(f"{random.Random(s.id).choice(s.info['choices'])} & 20")) > \
        mean_reward(env, rnd)


# ---------------------------------------------------------------- ORD
def test_ord():
    env = make_env("ord", split="test", seed=1)
    oracle = lambda s: wrap(str(s.info["gold"]))
    ident = lambda s: wrap(str(list(range(s.info["n"]))))
    rev = lambda s: wrap(str(s.info["gold"][::-1]))
    def rnd(s):
        p = list(range(s.info["n"])); random.Random(s.id).shuffle(p); return wrap(str(p))
    assert mean_reward(env, oracle) == pytest.approx(1.0)
    assert mean_reward(env, rev) == pytest.approx(0.0)
    assert mean_reward(env, rnd) < 0.3
    assert mean_reward(env, ident) < 0.35                      # identity order must not be a free win
    assert mean_reward(env, lambda s: wrap("[0,0,1]")) == 0     # not a permutation
    assert mean_reward(env, lambda s: "1 2 3") == 0


# ---------------------------------------------------------------- ERR
def test_err():
    env = make_env("err", split="test", seed=1)
    oracle = lambda s: wrap("True" if s.info["is_correct"] else f"False || {s.info['gold']}")
    assert mean_reward(env, oracle) == pytest.approx(1.0)
    true_ = mean_reward(env, lambda s: wrap("True"))
    false_copy = mean_reward(env, lambda s: wrap(f"False || {s.info['shown']}"))   # flag all, repair nothing
    assert true_ < 0.6 and false_copy < 0.5
    # flagging only gives the detection share; the repair share must be earned
    errs = [env.sample(i) for i in range(len(env)) if not env.sample(i).info["is_correct"]][:100]
    detect_only = st.mean(env.score(s, wrap(f"False || {s.info['shown']}")).value for s in errs)
    assert detect_only == pytest.approx(0.4, abs=1e-6)
    # a repair on an already-correct step must score 0
    oks = [env.sample(i) for i in range(len(env)) if env.sample(i).info["is_correct"]][:50]
    assert all(env.score(s, wrap("False || whatever")).value == 0 for s in oks)


# ---------------------------------------------------------------- GEN
def test_gen():
    env = make_env("gen", split="test", seed=1)
    oracle = lambda s: "\n".join(f"{i+1}. {x}" for i, x in enumerate(s.info["ref"]))
    assert mean_reward(env, oracle, 100) == pytest.approx(1.0, abs=1e-6)
    assert mean_reward(env, lambda s: "", 100) == 0
    # padding: gold steps + 5x repeated gold step must score below the clean gold protocol
    pad = lambda s: oracle(s) + "\n" + "\n".join(f"{i+9}. {s.info['ref'][0]}" for i in range(5 * len(s.info["ref"])))
    assert mean_reward(env, pad, 100) < mean_reward(env, oracle, 100) - 0.1
    # reversed order loses the order term but keeps content
    rev = lambda s: "\n".join(f"{i+1}. {x}" for i, x in enumerate(s.info["ref"][::-1]))
    assert 0.6 < mean_reward(env, rev, 100) < 0.95
    # echoing the question is not a protocol
    echo = lambda s: s.prompt
    assert mean_reward(env, echo, 100) < 0.2
    # a generic boilerplate protocol must not score high
    boiler = lambda s: "\n".join(["1. Prepare the samples.", "2. Add the reagents.", "3. Incubate.", "4. Wash.", "5. Analyze the results."])
    assert mean_reward(env, boiler, 100) < 0.15


def test_gym_and_trl_interface():
    env = make_env("ord", split="test", seed=0)
    obs = env.reset()
    assert "steps are" in obs and "[0] " in obs
    r, done, info = env.step("garbage")
    assert (r, done, info["parsed"]) == (0.0, True, False)
    rows = env.dataset(5)
    out = env.reward_fn(["x"] * 5, info=[r_["info"] for r_ in rows])
    assert out == [0.0] * 5


# ---------------------------------------------------------------- control reward modes (spurious-reward baselines)
@pytest.mark.parametrize("name", ["pqa", "ord", "err", "gen"])
def test_control_modes(name):
    from bioenvs.audit import POLICIES, mean_reward
    pols = POLICIES[name]
    oracle, bad = pols["oracle"], next(k for k in pols if k not in ("oracle", "no answer tags"))
    true_env = make_env(name, split="test", seed=1)
    assert mean_reward(true_env, oracle, 150, use_mode=True) > 0.9
    # random: independent of the completion, so the oracle and a bad policy earn the same ~p
    rnd = make_env(name, split="test", seed=1, reward_mode="random", random_p=0.5)
    a, b = mean_reward(rnd, oracle, 400, True), mean_reward(rnd, pols[bad], 400, True)
    assert abs(a - 0.5) < 0.1 and abs(b - 0.5) < 0.1
    # format: well-formed output always gets 1 whatever its content, unformatted gets 0
    fmt = make_env(name, split="test", seed=1, reward_mode="format")
    assert mean_reward(fmt, pols[bad], 100, True) > 0.9
    assert mean_reward(fmt, lambda s: "", 50, True) == 0
    # incorrect label: the oracle must stop standing out (it is now rewarded for a wrong label)
    inc = make_env(name, split="test", seed=1, reward_mode="incorrect")
    assert mean_reward(inc, oracle, 150, True) < 0.5
    assert mean_reward(inc, oracle, 150, True) < mean_reward(true_env, oracle, 150, True) - 0.5


def test_unknown_mode_rejected():
    with pytest.raises(AssertionError):
        make_env("pqa", split="test", reward_mode="oops")
