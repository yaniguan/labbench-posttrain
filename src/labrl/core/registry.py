"""Plugin registries for scientific corpora, verifiable environments, and training stages."""
from __future__ import annotations

from typing import Any, Callable

CORPUS_REGISTRY: dict[str, type[Any]] = {}
ENV_REGISTRY: dict[str, Callable[..., Any]] = {}
STAGE_REGISTRY: dict[str, type[Any]] = {}
REWARD_TEMPLATE_REGISTRY: dict[str, type[Any]] = {}


def register_corpus(name: str):
    def _wrap(cls: type[Any]) -> type[Any]:
        cls.name = name
        CORPUS_REGISTRY[name] = cls
        return cls

    return _wrap


def register_env(name: str):
    def _wrap(cls_or_fn: Callable[..., Any]) -> Callable[..., Any]:
        ENV_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return _wrap


def register_stage(name: str):
    def _wrap(cls: type[Any]) -> type[Any]:
        cls.method = name
        STAGE_REGISTRY[name] = cls
        return cls

    return _wrap


def register_reward_template(name: str):
    def _wrap(cls: type[Any]) -> type[Any]:
        cls.template_id = name
        REWARD_TEMPLATE_REGISTRY[name] = cls
        return cls

    return _wrap
