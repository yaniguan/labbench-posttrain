import json
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, "src")
from labpt.actions import ACTION_SCHEMA, ActionResult, run_log_to_reward_pairs, validate_action
from labpt.constraints import check_constraints
from labpt.logs import LogWriter, read_log


@pytest.mark.parametrize("action,params", [
    ("aspirate", {"volume_uL": 50, "well": "A12", "labware": "plate1"}),
    ("dispense", {"volume_uL": 50, "well": "H1", "labware": "plate1"}),
    ("move", {"position_m": [0, 0, 0.2], "joints_rad": [0, 1]}),
    ("heat", {"temperature_C": 37, "duration_s": 60}),
    ("wait", {"duration_s": 1}),
    ("measure", {"instrument": "reader", "channel": "OD600"}),
])
def test_schema_actions(action, params):
    result = validate_action({"action": action, "params": params})
    assert validate_action(result.model_dump_json()).model_dump() == result.model_dump()
    assert ACTION_SCHEMA["discriminator"]["propertyName"] == "action"


@pytest.mark.parametrize("action", [
    {"action": "teleport", "params": {}},
    {"action": "heat", "params": {"temperature_C": 900, "duration_s": 1}},
    {"action": "wait", "params": {"duration_s": -1}},
    {"action": "wait", "params": {"duration_s": True}},
    {"action": "wait", "params": {"duration_s": float("nan")}},
    {"action": "wait", "params": {"duration_s": "1"}},
    {"action": "aspirate", "params": {"volume_uL": 10, "well": "A13", "labware": "plate"}},
    {"action": "dispense", "params": {"volume_uL": 1001, "well": "A1", "labware": "plate"}},
    {"action": "measure", "params": {"instrument": "reader"}},
    {"action": "move", "params": {"position_m": [0, 0], "joints_rad": [0]}},
    {"action": "wait", "params": {"duration_s": 1, "secret": "ignored?"}},
])
def test_reject_bad_calls(action):
    with pytest.raises(ValidationError):
        validate_action(action)


@pytest.mark.parametrize("result", [
    {"status": "ok", "data": None, "error": None},
    {"status": "ok", "data": {}, "error": "bad"},
    {"status": "error", "data": {}, "error": "bad"},
    {"status": "retried", "data": None, "error": None},
    {"status": "skipped", "data": None, "error": ""},
])
def test_data_error_semantics(result):
    with pytest.raises(ValidationError):
        ActionResult.model_validate(result)


def event(status, attempt=1, step="wait"):
    return {"run_id": "run1", "step_id": step, "attempt": attempt,
            "action": {"action": "wait", "params": {"duration_s": 1}},
            "result": {"status": status, "data": {} if status == "ok" else None,
                       "error": None if status == "ok" else "instrument busy"}}


def test_log_rewards_use_observed_recovery_and_plan():
    rows = [event("error"), event("retried", 2), event("ok", 3)]
    res = run_log_to_reward_pairs(rows, ["wait", "measure"])
    assert res["t1_reward"] == 0.5
    pair, = res["dpo_pairs"]
    assert json.loads(pair["chosen"]) == rows
    assert json.loads(pair["rejected"]) == rows[:1]
    assert run_log_to_reward_pairs([event("retried")])["t1_reward"] == 0
    assert run_log_to_reward_pairs([event("skipped")])["dpo_pairs"] == []
    assert run_log_to_reward_pairs([])["t1_reward"] == 0


def test_log_does_not_invent_pairs_or_mix_runs():
    assert run_log_to_reward_pairs([event("error")])["dpo_pairs"] == []
    assert run_log_to_reward_pairs([event("ok")])["dpo_pairs"] == []
    with pytest.raises(ValueError):
        run_log_to_reward_pairs([event("error"), event("ok")])
    with pytest.raises(ValueError):
        run_log_to_reward_pairs([event("ok")], ["other"])
    with pytest.raises(ValueError):
        run_log_to_reward_pairs([event("ok"), dict(event("error"), run_id="run2")])


def test_jsonl_round_trip_append_filter_and_bad_record(tmp_path):
    path = tmp_path / "events.jsonl"
    with LogWriter(path) as log:
        log.write("/actions", event("error"), timestamp=100)
    with LogWriter(path) as log:
        log.write("/actions", event("ok", 2), timestamp=200)
        log.write("/state", {"joints": [0]}, timestamp=201)
    rows = list(read_log(path, "/actions"))
    assert [r["timestamp"] for r in rows] == [100, 200]
    assert run_log_to_reward_pairs([r["message"] for r in rows])["t1_reward"] == 1
    with path.open("a") as f:
        f.write('{"topic":"bad"}\n')
    with pytest.raises(ValidationError):
        list(read_log(path))


SPEC = {"workspace": {"lower": [0, 0, 0], "upper": [1, 1, 1]},
        "joint_limits_rad": [[-1, 1], [-2, 2]],
        "collision_boxes": [{"lower": [0.4, 0.4, 0.4], "upper": [0.6, 0.6, 0.6]}]}


def plan(*positions, joints=None):
    return {"skill": "transfer", "waypoints": [{"position_m": list(x), "joints_rad": joints or [0, 0]} for x in positions]}


def test_constraint_reward_bounds_joints_and_swept_collision():
    assert check_constraints(plan([0, 0, 0], [1, 0, 0]), SPEC)["reward"] == 1
    assert check_constraints(plan([-0.1, 0, 0]), SPEC)["reward"] == 0
    assert check_constraints(plan([0, 0, 0], joints=[0, 3]), SPEC)["reward"] == 0
    assert check_constraints(plan([0, 0, 0], joints=[0]), SPEC)["reward"] == 0
    # Endpoints outside the obstacle still cross it.
    assert check_constraints(plan([0.1, 0.5, 0.5], [0.9, 0.5, 0.5]), SPEC)["reward"] == 0
    assert check_constraints(plan([0.5, 0.5, 0.5]), SPEC)["reward"] == 0
    assert check_constraints(plan([float("inf"), 0, 0]), SPEC)["reward"] == 0
    assert check_constraints({"skill": "skip", "waypoints": []}, SPEC)["reward"] == 0
    assert check_constraints({}, SPEC)["reward"] == 0


def test_model_constraints_only_tighten_trusted_constraints():
    proposed = plan([0.1, 0.1, 0.1], [0.9, 0.1, 0.1])
    assert check_constraints(proposed, SPEC)["reward"] == 1
    proposed["constraints"] = {"avoid_boxes": [{"lower": [0.4, 0, 0], "upper": [0.6, 0.2, 0.2]}]}
    assert check_constraints(proposed, SPEC)["reward"] == 0
    crossing = plan([0.1, 0.5, 0.5], [0.9, 0.5, 0.5])
    crossing["constraints"] = {"avoid_boxes": []}
    assert check_constraints(crossing, SPEC)["reward"] == 0
    crossing["constraints"] = {"workspace": {"lower": [-10, -10, -10], "upper": [10, 10, 10]}}
    assert check_constraints(crossing, SPEC)["reward"] == 0
