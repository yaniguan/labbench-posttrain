"""Composable multi-objective & safety-gated reward combinators (`labrl.rewards.composers`)."""
from __future__ import annotations

from typing import Any, Callable, Sequence

from labpt.rewards import counterfactual_consistency_reward
from labrl.core.registry import REWARD_TEMPLATE_REGISTRY
from labrl.core.types import RewardPacket
from labrl.rewards.extractors import ExtractedAnswer
from labrl.rewards.templates import VerifiedRewardTemplate


class GatedRewardTemplate(VerifiedRewardTemplate):
    """Hierarchical safety-gated multi-objective reward composer.

    Computes:
        G = prod_{gate in hard_gates} gate(info, completion) in [0, 1]
        R = G * sum_i w_i * template_i.score(info, completion)

    Ensuring that any hard physical/safety/stoichiometry gate failure (`G == 0`)
    zeros BOTH the scalar reward AND every downstream objective GDPO channel.
    """

    template_id = "gated_composite"
    description = "Multiplicative hard-safety-gated multi-objective reward combinator."

    def __init__(
        self,
        objectives: Sequence[tuple[str, VerifiedRewardTemplate, float]],
        *,
        hard_gates: Sequence[tuple[str, Callable[[dict[str, Any], str], float]]] = (),
        **kw: Any,
    ) -> None:
        if not objectives:
            raise ValueError("GatedRewardTemplate requires at least one (name, template, weight) objective")
        super().__init__(**kw)
        self.objectives = [(str(n), t, float(w)) for n, t, w in objectives]
        self.hard_gates = [(str(gn), gfn) for gn, gfn in hard_gates]
        w_sum = sum(w for _, _, w in self.objectives)
        if w_sum <= 0.0:
            raise ValueError("Sum of objective weights in GatedRewardTemplate must be positive")
        self._norm_weights = [w / w_sum for _, _, w in self.objectives]

    def score(self, info: dict[str, Any], completion: str) -> RewardPacket:
        gate_prod = 1.0
        channels: dict[str, float] = {}
        telemetry: dict[str, float] = {}

        for g_name, g_fn in self.hard_gates:
            g_val = max(0.0, min(1.0, float(g_fn(info, completion))))
            channels[f"reward/gate_{g_name}"] = g_val
            gate_prod *= g_val

        all_parsed = True
        weighted_scalar = 0.0
        for (obj_name, tmpl, _), nw in zip(self.objectives, self._norm_weights):
            sub_info = info.get(obj_name, info) if isinstance(info.get(obj_name), dict) else info
            pkt = tmpl.score(sub_info, completion)
            all_parsed = all_parsed and pkt.parsed
            weighted_scalar += nw * pkt.scalar
            for ck, cv in pkt.normalized_channels().items():
                short = ck.removeprefix("reward/")
                channels[f"reward/{obj_name}_{short}"] = gate_prod * cv
            for tk, tv in pkt.telemetry.items():
                telemetry[f"{obj_name}/{tk}"] = tv

        telemetry["gate_product"] = gate_prod
        return RewardPacket(
            scalar=gate_prod * weighted_scalar,
            parsed=all_parsed,
            channels=channels,
            telemetry=telemetry,
        )

    def _score_extracted(self, info: dict[str, Any], extracted: ExtractedAnswer) -> RewardPacket:
        return self.score(info, extracted.raw_completion)


class CounterfactualPairScorer:
    """Evaluates paired `(original, counterfactual)` completions to detect shortcut/template memorization."""

    def __init__(self, template: VerifiedRewardTemplate, *, threshold: float = 0.70) -> None:
        self.template = template
        self.threshold = float(threshold)

    def score_pair(
        self,
        orig_info: dict[str, Any],
        orig_completion: str,
        cf_info: dict[str, Any],
        cf_completion: str,
    ) -> RewardPacket:
        p_orig = self.template.score(orig_info, orig_completion)
        p_cf = self.template.score(cf_info, cf_completion)
        cons = counterfactual_consistency_reward(p_orig.scalar, p_cf.scalar, threshold=self.threshold)
        mean_scalar = 0.5 * (p_orig.scalar + p_cf.scalar)
        # Gate mean reward by counterfactual consistency so memorized answers cannot score high
        shaped = 0.60 * mean_scalar + 0.40 * cons
        return RewardPacket(
            scalar=shaped,
            parsed=p_orig.parsed and p_cf.parsed,
            channels={
                "reward/orig": p_orig.scalar,
                "reward/counterfactual": p_cf.scalar,
                "reward/consistency": cons,
            },
            telemetry={"mean_unpaired": mean_scalar},
        )


def make_reward_template(name_or_obj: str | VerifiedRewardTemplate, **kwargs: Any) -> VerifiedRewardTemplate:
    if isinstance(name_or_obj, VerifiedRewardTemplate):
        return name_or_obj
    if name_or_obj not in REWARD_TEMPLATE_REGISTRY:
        raise KeyError(
            f"Unknown reward template {name_or_obj!r}; registered: {sorted(REWARD_TEMPLATE_REGISTRY)}"
        )
    return REWARD_TEMPLATE_REGISTRY[name_or_obj](**kwargs)


def list_reward_templates() -> list[dict[str, Any]]:
    return [
        REWARD_TEMPLATE_REGISTRY[k].describe()
        for k in sorted(REWARD_TEMPLATE_REGISTRY.keys())
    ]
