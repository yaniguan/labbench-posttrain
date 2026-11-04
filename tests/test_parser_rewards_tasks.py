import json
import sys

sys.path.insert(0, "src")
from labpt import rewards
from labpt.annotate import safety_level
from labpt.protocol_parser import parse_procedure, robot_plan_from_instruction
from labpt.tasks import corrupt_actions


def test_parse_procedure():
    steps = parse_procedure("The ester (19.5 g) is dissolved in THF (150 ml). The reaction is heated at reflux for 3 hours. "
                            "The solvent is removed in vacuo. The precipitate is filtered and dried.")
    names = [s.action for s in steps]
    assert names[:3] == ["dissolve", "reflux", "concentrate"]
    assert "filter" in names and "dry" in names
    assert steps[1].params["duration"] == "3 hours"


def test_robot_plan():
    plan = [a.action for a in robot_plan_from_instruction("pour water into the cup")]
    assert "pour" in plan and plan[0] == "move"


def test_safety():
    assert safety_level("NaH in DMF") == "high"
    assert safety_level("stirred in water", max_temp=25) == "low"


def test_rewards():
    good = json.dumps([{"action": "add", "params": {}}, {"action": "stir", "params": {}}])
    assert rewards.schema_valid_actions(good) == 1.0
    assert rewards.schema_valid_actions("[]") == 0.0  # empty protocol is not rewarded
    assert rewards.schema_valid_actions('[{"action":"heat","params":{"temperature_C":900}}]') == 0.0  # beyond hotplate limit
    assert rewards.action_overlap(good, good) == 1.0
    assert rewards.provenance_correct('{"provenance": "compute"}', "compute") == 1.0
    assert rewards.unit_consistent('{"units": "Ha"}', "gap", "Ha") == 1.0
    assert rewards.unit_consistent('{"units": "kg"}', "gap", "Ha") == 0.0


def test_corrupt_changes_answer():
    import random
    a = json.dumps([{"action": "add", "params": {}}, {"action": "stir", "params": {}}, {"action": "filter", "params": {}}])
    assert corrupt_actions(a, random.Random(0)) != a


def test_task_files_exist_and_parse():
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(f"data/v0/tasks/sft_{split}.jsonl")]
        assert rows and all(r["answer"] and r["prompt"][-1]["role"] == "user" for r in rows)
        # the provenance task must not leak the source name
        for r in rows:
            if r["task"] == "provenance":
                assert r["meta"]["source"] not in r["prompt"][-1]["content"]
