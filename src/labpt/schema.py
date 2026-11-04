"""Unified record schema. Every record carries a provenance label."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Provenance = Literal["measurement", "compute", "ml_prediction", "human_annotation"]
Modality = Literal["robot_episode", "reaction", "bioassay", "molecular_property",
                   "spectrum", "run_log", "lab_note", "image", "table"]
Scale = Literal["atomic", "molecular", "material", "cell", "device", "lab-process"]

ACTION_VOCAB = [
    # wet-lab unit operations
    "add", "aspirate", "dispense", "stir", "heat", "cool", "wait", "reflux", "filter", "wash", "dry",
    "concentrate", "purify", "extract", "quench", "dissolve", "transfer",
    "degas", "crystallize", "centrifuge", "measure",
    # robot primitives
    "move", "pick", "place", "grasp", "release", "open", "close", "pour", "push",
    "wipe", "rotate",
]


class Action(BaseModel):
    action: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action")
    @classmethod
    def _in_vocab(cls, v: str) -> str:
        if v not in ACTION_VOCAB:
            raise ValueError(f"unknown action {v!r}")
        return v


class Annotation(BaseModel):
    key: Literal["step_type", "failure_reason", "safety_level", "task_category", "note"]
    value: Any
    annotator: str  # e.g. "rule:v0", "llm:codex", "human:<name>"
    auto: bool = True  # False only when a human produced/verified it


class Outcome(BaseModel):
    success: Optional[bool] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    units: Optional[str] = None


class Record(BaseModel):
    id: str
    source: str
    source_id: str
    source_url: Optional[str] = None
    license: str
    modality: Modality
    provenance: Provenance
    physical_scale: Scale = "molecular"
    raw: Optional[str] = None  # pointer/excerpt of the messy original (file path or raw text)
    method: Optional[str] = None
    instrument: Optional[str] = None
    property: Optional[str] = None
    value: Optional[float] = None
    value_relation: Optional[str] = None  # '=', '<', '>' ...
    units: Optional[str] = None
    uncertainty: Optional[float] = None
    entity: dict[str, Any] = Field(default_factory=dict)  # smiles, target, task text...
    conditions: dict[str, Any] = Field(default_factory=dict)
    protocol_text: Optional[str] = None
    protocol_steps: list[Action] = Field(default_factory=list)
    outcome: Outcome = Field(default_factory=Outcome)
    annotations: list[Annotation] = Field(default_factory=list)
    group_key: str = ""  # leakage group (scaffold / dataset / episode set)
    split: Optional[Literal["train", "val", "test", "prospective_blind"]] = None

    def dedup_key(self) -> str:
        payload = json.dumps(
            [self.source, self.entity, self.property, self.value, self.units,
             self.conditions, self.protocol_text],
            sort_keys=True, default=str,
        )
        return hashlib.sha1(payload.encode()).hexdigest()


def make_id(source: str, *parts: Any) -> str:
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]
    return f"{source}:{h}"


def json_schema() -> dict:
    return Record.model_json_schema()
