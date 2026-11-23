"""Reproducible run metadata helper for evaluation and benchmark result JSONs."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

_TRACKED_LIBS = ("torch", "numpy", "scipy", "transformers", "pydantic", "httpx", "pyyaml")


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def git_info(repo_root: str | Path = REPO_ROOT) -> tuple[str, bool]:
    root = str(repo_root)
    commit = "unknown"
    dirty = False
    try:
        res = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            commit = res.stdout.strip()
        st = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if st.returncode == 0:
            dirty = bool(st.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    return commit, dirty


def library_versions(names: Iterable[str] = _TRACKED_LIBS) -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version()}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def build_run_meta(
    *,
    data_files: Iterable[str | Path] | None = None,
    prompt_template: str | None = None,
    reward_version: str | None = None,
    sampling: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    commit, dirty = git_info()
    file_hashes: dict[str, str] = {}
    for fp in data_files or []:
        p = Path(fp)
        digest = sha256_file(p)
        if digest is not None:
            try:
                rel = p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            except ValueError:
                rel = p.as_posix()
            file_hashes[rel] = digest

    meta: dict[str, Any] = {
        "git_commit": commit,
        "git_dirty": dirty,
        "utc_timestamp": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "data_sha256": file_hashes,
        "prompt_template_sha256": sha256_text(prompt_template),
        "reward_version": reward_version,
        "sampling": dict(sampling or {}),
        "library_versions": library_versions(),
    }
    if extra:
        meta.update(extra)
    return meta
