"""Typed instrument calls, Data/Error results, and rewards from observed retries.

These stricter calls complement the weak protocol labels in schema.Action.
No hardware is executed here. Units are explicit in parameter names.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class LiquidParams(StrictModel):
    volume_uL: float = Field(gt=0, le=1000)
    well: str = Field(pattern=r"^[A-H](?:[1-9]|1[0-2])$")
    labware: str = Field(min_length=1)


class Aspirate(StrictModel):
    action: Literal["aspirate"]
    params: LiquidParams


class Dispense(StrictModel):
    action: Literal["dispense"]
    params: LiquidParams


class MoveParams(StrictModel):
    position_m: list[float] = Field(min_length=3, max_length=3)
    joints_rad: list[float] = Field(min_length=1)


class Move(StrictModel):
    action: Literal["move"]
    params: MoveParams


class HeatParams(StrictModel):
    temperature_C: float = Field(ge=-100, le=250)
    duration_s: float = Field(gt=0)


class Heat(StrictModel):
    action: Literal["heat"]
    params: HeatParams


class WaitParams(StrictModel):
    duration_s: float = Field(gt=0)


class Wait(StrictModel):
    action: Literal["wait"]
    params: WaitParams


class MeasureParams(StrictModel):
    instrument: str = Field(min_length=1)
    channel: str = Field(min_length=1)


class Measure(StrictModel):
    action: Literal["measure"]
    params: MeasureParams


InstrumentAction = Annotated[Aspirate | Dispense | Move | Heat | Wait | Measure, Field(discriminator="action")]
ACTION_ADAPTER = TypeAdapter(InstrumentAction)
ACTION_SCHEMA = ACTION_ADAPTER.json_schema()


def validate_action(action: dict | str):
    """Return a typed call or raise pydantic.ValidationError."""
    if isinstance(action, str):
        action = json.loads(action)
    return ACTION_ADAPTER.validate_python(action)


class ActionResult(StrictModel):
    # retried means retry requested; only a later ok means execution succeeded.
    status: Literal["ok", "error", "skipped", "retried"]
    data: dict[str, Any] | None
    error: str | None

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "ok":
            if self.data is None or self.error is not None:
                raise ValueError("ok requires Data and no Error")
        elif self.data is not None or not self.error:
            raise ValueError("error/skip/retry requires an Error/reason and no Data")
        return self


RESULT_SCHEMA = ActionResult.model_json_schema()


class RunEvent(StrictModel):
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    action: InstrumentAction
    result: ActionResult


def run_log_to_reward_pairs(log: list[dict], expected_steps: list[str] | None = None) -> dict:
    """T1 = terminal ok steps / planned steps; skip/retry alone gets no credit.

    Pairs compare real prefixes for the SAME run, step and call: a recovered ok
    trace against its earlier error trace. No synthetic successful results.
    Pass expected_steps to penalize omitted steps. Without a plan this measures
    observed completion only, not protocol completeness or scientific quality.
    """
    events = [RunEvent.model_validate(row) for row in log]
    if len({e.run_id for e in events}) > 1:
        raise ValueError("pass one run at a time")
    planned = set(expected_steps) if expected_steps is not None else {e.step_id for e in events}
    if expected_steps is not None and len(planned) != len(expected_steps):
        raise ValueError("duplicate planned step")
    terminal, traces, pairs = {}, {}, []
    for event in events:
        if event.step_id not in planned:
            raise ValueError("unplanned step")
        previous = terminal.get(event.step_id)
        if previous and event.attempt <= previous.attempt:
            raise ValueError("attempts must increase per step")
        trace = traces.setdefault(event.step_id, [])
        row = event.model_dump()
        if event.result.status == "ok":
            errors = [i for i, r in enumerate(trace)
                      if r["result"]["status"] == "error" and r["action"] == row["action"]]
            if errors:
                pairs.append({
                    "prompt": [{"role": "user", "content": "Complete this instrument call, resolving execution errors: "
                                + json.dumps(row["action"], sort_keys=True)}],
                    "chosen": json.dumps(trace + [row], sort_keys=True),
                    "rejected": json.dumps(trace[:errors[-1] + 1], sort_keys=True),
                    "kind": "execution_recovery", "run_id": event.run_id, "step_id": event.step_id,
                })
        trace.append(row)
        terminal[event.step_id] = event
    completed = sum(e.result.status == "ok" for e in terminal.values())
    return {"t1_reward": completed / len(planned) if planned else 0.0,
            "completed_steps": completed, "planned_steps": len(planned), "dpo_pairs": pairs}


# ---------------------------------------------------------------------------
# L3 <-> L4 (`lab-rehearsal` / `lab-verify`) Typed Protocol Interchange (#42)
# ---------------------------------------------------------------------------

def _to_rehearsal_loc(labware: str, well: str) -> str:
    lw = labware.strip()
    w = well.strip()
    if ":" in lw:
        return lw
    low = lw.lower()
    if low == "res" or low.startswith("res") or low.startswith("reservoir"):
        import re as _re

        m = _re.search(r"(\d+)$", w)
        col = m.group(1) if m else "1"
        return f"res:{col}"
    if low in ("plate", "plate1", "96well", "dest_plate", "source_plate") or low.startswith("plate"):
        return f"plate:{w}"
    return f"{lw}:{w}"


def to_rehearsal_protocol(
    actions: list[InstrumentAction | dict[str, Any]],
    *,
    auto_tips: bool = True,
) -> list[dict[str, Any]]:
    """Translate `labpt` `InstrumentAction` calls into `lab-rehearsal` deck steps.

    `lab-rehearsal` (`L4`) verifies `{op, loc: 'plate:A1' | 'res:1', vol}` steps
    with explicit `pick_up_tip` and `drop_tip` gating on a 96-well virtual deck.
    """
    steps: list[dict[str, Any]] = []
    tip_held = False
    for raw in actions:
        if isinstance(raw, BaseModel):
            item = raw.model_dump()
        elif isinstance(raw, str):
            item = json.loads(raw)
        else:
            item = dict(raw)

        act = str(item.get("action") or item.get("op") or "").strip()
        params = dict(item.get("params") or {})

        if act == "pick_up_tip":
            steps.append({"op": "pick_up_tip"})
            tip_held = True
        elif act == "drop_tip":
            steps.append({"op": "drop_tip"})
            tip_held = False
        elif act in ("aspirate", "dispense", "mix"):
            if auto_tips and not tip_held:
                steps.append({"op": "pick_up_tip"})
                tip_held = True
            if "loc" in item:
                loc = str(item["loc"])
                vol = float(item.get("vol", params.get("volume_uL", 0.0)))
            else:
                validated = validate_action({"action": act, "params": params}) if act in ("aspirate", "dispense") else None
                if validated is not None:
                    loc = _to_rehearsal_loc(validated.params.labware, validated.params.well)
                    vol = float(validated.params.volume_uL)
                else:
                    loc = _to_rehearsal_loc(str(params.get("labware", "plate")), str(params.get("well", "A1")))
                    vol = float(params.get("volume_uL", 0.0))
            steps.append({"op": act, "loc": loc, "vol": vol})
    if auto_tips and tip_held:
        steps.append({"op": "drop_tip"})
    return steps


def from_rehearsal_protocol(steps: list[dict[str, Any]]) -> list[InstrumentAction]:
    """Parse `lab-rehearsal` liquid-handling steps back into validated `InstrumentAction` objects."""
    actions: list[InstrumentAction] = []
    for step in steps:
        op = str(step.get("op", "")).strip()
        if op in ("pick_up_tip", "drop_tip"):
            continue
        if op not in ("aspirate", "dispense"):
            raise ValueError(f"Unsupported rehearsal liquid op for InstrumentAction conversion: {op!r}")
        loc = str(step.get("loc", ""))
        vol = float(step.get("vol", 0.0))
        if ":" in loc:
            prefix, target = loc.split(":", 1)
        else:
            prefix, target = "plate", loc
        if prefix == "res":
            col_int = int(target)
            if not 1 <= col_int <= 12:
                raise ValueError(f"Reservoir column out of range 1..12: {loc!r}")
            well = f"A{col_int}"
            labware = "reservoir"
        else:
            well = target
            labware = prefix
        actions.append(
            validate_action({
                "action": op,
                "params": {"volume_uL": vol, "well": well, "labware": labware},
            })
        )
    return actions
