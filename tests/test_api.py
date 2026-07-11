"""Bumblebee API route logic with a fake engine (no GPU, no Modal)."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import bumblebee.api as api_module  # noqa: E402
from bumblebee.api import build_api  # noqa: E402
from bumblebee.models import (  # noqa: E402
    DocumentResult,
    DocumentTimings,
    OcrError,
    ResultSettings,
    TokenUsage,
)

PAGES_JSON = [
    [
        {"label": "text", "native_label": "doc_title", "content": "# Title", "bbox_2d": [1, 2, 3, 4]},
        {"label": "text", "native_label": "text", "content": "Body text.", "bbox_2d": [1, 5, 3, 8], "_ocr_usage": {}},
    ]
]


class FakeEngine:
    def __init__(self, error: OcrError | None = None):
        self.error = error
        self.calls: list[bytes] = []

    async def ocr(self, pdf: bytes) -> DocumentResult:
        self.calls.append(pdf)
        if self.error is not None:
            raise self.error
        return DocumentResult(
            filename="document.pdf",
            page_count=1,
            region_count=2,
            ocr_region_count=2,
            skipped_region_count=0,
            json=PAGES_JSON,
            markdown="# Title\n\nBody text.",
            tokens=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            timings=DocumentTimings(total_seconds=0.5),
            settings=ResultSettings(
                pdf_dpi=100, layout_model="m", layout_backend="fake", ocr_backend="fake", ocr_request_concurrency=1
            ),
        )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("BUMBLEBEE_API_KEY", "sekrit")
    return TestClient(build_api(FakeEngine()))


AUTH = {"Authorization": "Bearer sekrit"}


def test_health_needs_no_auth(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_parse_returns_markdown_layout_and_chunks(client):
    response = client.post("/v1/parse?filename=report.pdf", content=b"%PDF-fake", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "report.pdf"
    assert body["markdown"].startswith("# Title")
    assert body["stats"]["pages"] == 1
    # Private OCR keys are sanitized out of the returned layout.
    assert all("_ocr_usage" not in region for page in body["layout"] for region in page)
    (chunk,) = body["chunks"]
    assert chunk["chunk_id"] == "report#0000"
    assert chunk["section_path"] == ["Title"]


def test_parse_can_skip_chunks(client):
    body = client.post("/v1/parse?chunks=false", content=b"%PDF-fake", headers=AUTH).json()
    assert "chunks" not in body


def test_missing_or_wrong_token_is_rejected(client):
    assert client.post("/v1/parse", content=b"%PDF-fake").status_code == 401
    assert client.post("/v1/parse", content=b"x", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_unconfigured_key_fails_closed(monkeypatch):
    monkeypatch.delenv("BUMBLEBEE_API_KEY", raising=False)
    client = TestClient(build_api(FakeEngine()))
    assert client.post("/v1/parse", content=b"x", headers=AUTH).status_code == 503


def test_empty_body_is_rejected(client):
    assert client.post("/v1/parse", content=b"", headers=AUTH).status_code == 400


def test_oversized_body_is_rejected(client, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_UPLOAD_BYTES", 4)
    assert client.post("/v1/parse", content=b"12345", headers=AUTH).status_code == 413


def test_ocr_failure_maps_to_422(monkeypatch):
    monkeypatch.setenv("BUMBLEBEE_API_KEY", "sekrit")
    client = TestClient(build_api(FakeEngine(error=OcrError(phase="ocr", message="boom"))))
    response = client.post("/v1/parse", content=b"%PDF-fake", headers=AUTH)
    assert response.status_code == 422
    assert "boom" in response.json()["detail"]
