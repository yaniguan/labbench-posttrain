"""Small geometric reward checker. No IK, robot link geometry, or dynamics."""
from pydantic import Field, model_validator

from .actions import StrictModel


class Box(StrictModel):
    lower: list[float] = Field(min_length=3, max_length=3)
    upper: list[float] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def ordered(self):
        if any(a >= b for a, b in zip(self.lower, self.upper)):
            raise ValueError("box lower must be below upper")
        return self


class ConstraintSpec(StrictModel):
    workspace: Box
    joint_limits_rad: list[list[float]] = Field(min_length=1)
    collision_boxes: list[Box] = Field(default_factory=list)

    @model_validator(mode="after")
    def joint_ranges(self):
        if any(len(r) != 2 or r[0] >= r[1] for r in self.joint_limits_rad):
            raise ValueError("joint limits need ordered [lower, upper] pairs")
        return self


class Waypoint(StrictModel):
    position_m: list[float] = Field(min_length=3, max_length=3)
    joints_rad: list[float] = Field(min_length=1)


class PlanConstraints(StrictModel):
    # Model-proposed restrictions can add obstacles, never relax trusted limits.
    avoid_boxes: list[Box] = Field(default_factory=list)


class OperatorPlan(StrictModel):
    skill: str = Field(min_length=1)
    waypoints: list[Waypoint] = Field(min_length=1)
    constraints: PlanConstraints = Field(default_factory=PlanConstraints)


def segment_hits_box(start, end, box):
    """Slab test of the full swept point segment, including box boundaries."""
    t0, t1 = 0.0, 1.0
    for a, b, lo, hi in zip(start, end, box.lower, box.upper):
        delta = b - a
        if delta == 0:
            if a < lo or a > hi:
                return False
        else:
            enter, leave = sorted(((lo - a) / delta, (hi - a) / delta))
            t0, t1 = max(t0, enter), min(t1, leave)
            if t0 > t1:
                return False
    return True


def check_constraints(plan: dict, constraints: dict) -> dict:
    """Fail closed on malformed plans; constraints come from the trusted caller."""
    from pydantic import ValidationError
    spec = ConstraintSpec.model_validate(constraints)
    try:
        plan = OperatorPlan.model_validate(plan)
    except ValidationError as error:
        return {"reward": 0.0, "valid": False, "violations": [str(error)]}
    violations = []
    previous = None
    for i, point in enumerate(plan.waypoints):
        xyz, joints = point.position_m, point.joints_rad
        if any(x < lo or x > hi for x, lo, hi in zip(xyz, spec.workspace.lower, spec.workspace.upper)):
            violations.append(f"waypoint {i}: workspace")
        if len(joints) != len(spec.joint_limits_rad) or any(
                x < lo or x > hi for x, (lo, hi) in zip(joints, spec.joint_limits_rad)):
            violations.append(f"waypoint {i}: joint limits")
        if any(segment_hits_box(previous or xyz, xyz, box)
               for box in spec.collision_boxes + plan.constraints.avoid_boxes):
            violations.append(f"waypoint {i}: collision")
        previous = xyz
    return {"reward": float(not violations), "valid": not violations, "violations": violations}
