"""Loose-coupled NeMo-RL interface shim.

When `nemo_rl` is importable (e.g. on 4-8xA100/H100 or multi-node clusters),
`EnvironmentInterface` and `EnvironmentReturn` are literally imported from
`nemo_rl.environments.interfaces`. On 1-GPU / CPU / CI machines without `nemo_rl`,
this module provides exact signature-compatible types so environments run
identically across both backends.
"""
from __future__ import annotations

import abc
from typing import Any, Generic, NamedTuple, TypeVar

from torch import Tensor

MetadataT = TypeVar("MetadataT")

try:
    from nemo_rl.environments.interfaces import (  # type: ignore[import-not-found]
        EnvironmentInterface,
        EnvironmentReturn,
    )

    NEMO_RL_AVAILABLE = True
except Exception:  # noqa: BLE001
    NEMO_RL_AVAILABLE = False

    class EnvironmentReturn(NamedTuple, Generic[MetadataT]):
        observations: list[dict[str, str]]
        metadata: list[MetadataT]
        next_stop_strings: list[list[str] | None] | list[None]
        rewards: Tensor | dict[str, Tensor]
        terminateds: Tensor
        answers: list[str | None] | None

    class EnvironmentInterface(abc.ABC, Generic[MetadataT]):
        @abc.abstractmethod
        def step(
            self,
            message_log_batch: list[list[dict[str, Any]]],
            metadata: list[MetadataT],
        ) -> EnvironmentReturn[MetadataT]:
            """Execute one batched environment step."""

        @abc.abstractmethod
        def global_post_process_and_metrics(
            self, batch: Any
        ) -> tuple[Any, dict[str, Any]]:
            """Aggregate batch-level metrics after rollout collection."""
