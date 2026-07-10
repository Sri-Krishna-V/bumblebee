"""Shared fixtures: region factories and an in-memory Storage fake.

The factories isolate tests from data-model details: when the shape of
Region/RecognizedRegion changes, only this file changes.
"""

import json
import os
import posixpath
from typing import Any

import pytest

from bumblebee.models import RecognizedRegion, Region, Task, TokenUsage


@pytest.fixture(autouse=True)
def _clean_bumblebee_env(monkeypatch):
    """Scrub ambient BUMBLEBEE_* variables so config env fallback is hermetic."""
    for name in list(os.environ):
        if name.startswith("BUMBLEBEE_"):
            monkeypatch.delenv(name)


def make_region(
    page: int,
    idx: int,
    label: str,
    task: str,
    bbox: list[int] | tuple[int, int, int, int],
    *,
    score: float = 0.9,
) -> Region:
    x1, y1, x2, y2 = bbox
    return Region(
        page_index=page,
        region_index=idx,
        label=label,
        task_type=Task(task),
        bbox_2d=(x1, y1, x2, y2),
        score=score,
    )


def make_recognized_region(
    page: int,
    idx: int,
    label: str,
    task: str,
    bbox: list[int] | tuple[int, int, int, int],
    content: str | None,
    *,
    score: float = 0.9,
    usage: dict[str, int] | None = None,
    status: int | None = 200,
    latency: int | None = 12,
) -> RecognizedRegion:
    return RecognizedRegion(
        region=make_region(page, idx, label, task, bbox, score=score),
        content=content,
        usage=TokenUsage(**usage) if usage is not None else None,
        status_code=status,
        latency_ms=latency,
    )


class FakeStorage:
    """In-memory Storage protocol implementation for planning/supervisor tests."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def list_pdfs(self, source: str):  # pragma: no cover - unused in fakes
        raise NotImplementedError

    def read_bytes(self, uri: str) -> bytes:
        return self.files[uri]

    def list_files(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/")
        return sorted(path for path in self.files if path == prefix or path.startswith(prefix + "/"))

    def write_text(self, uri: str, content: str) -> None:
        self.files[uri] = content.encode("utf-8")

    def read_json(self, uri: str) -> Any:
        return json.loads(self.files[uri].decode("utf-8"))

    def write_json(self, uri: str, payload: Any) -> None:
        self.files[uri] = json.dumps(payload).encode("utf-8")

    def exists(self, uri: str) -> bool:
        return uri in self.files

    def join(self, root: str, relative_path: str) -> str:
        return posixpath.join(root.rstrip("/"), relative_path)
