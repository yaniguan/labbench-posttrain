"""CPU smokes. Each one is a few optimiser steps on the tiny model."""
from __future__ import annotations

import math

from posttrain.dpo import smoke as dpo_smoke
from posttrain.grpo import smoke as grpo_smoke
from posttrain.ppo import smoke as ppo_smoke
from posttrain.sft import smoke as sft_smoke


def _finite(losses):
    assert losses
    assert all(math.isfinite(x) for x in losses)


def test_sft_smoke_finite():
    out = sft_smoke()
    _finite(out["losses"])
    assert out["status"] == "ran"


def test_dpo_smoke_keeps_the_sft_reference_frozen():
    out = dpo_smoke()
    _finite(out["losses"])
    assert out["reference_unchanged"] is True
    assert out["beta"] > 0


def test_ppo_smoke_keeps_the_sft_reference_frozen():
    out = ppo_smoke()
    _finite(out["losses"])
    assert out["reference_unchanged"] is True
    assert out["sft_anchor_unchanged"] is True
    assert out["kl_coef"] > 0


def test_grpo_smoke_keeps_the_sft_reference_frozen():
    out = grpo_smoke()
    _finite(out["losses"])
    assert out["reference_unchanged"] is True
    assert out["sft_anchor_unchanged"] is True
    assert out["kl_coef"] > 0
