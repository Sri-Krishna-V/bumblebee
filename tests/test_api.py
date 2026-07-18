"""Bumblebee API route logic with a fake engine (no GPU, no Modal)."""

import json

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
        {
            "label": "text",
            "native_label": "text",
            "content": "Body text.",
            "bbox_2d": [1, 5, 3, 8],
            "_ocr_usage": {},
            "_ocr_confidence": 0.87,
            "_ocr_confidence_before": 0.61,
        },
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
            timings=DocumentTimings(total_seconds=0.5, render_seconds=0.1, ocr_seconds=0.3),
            settings=ResultSettings(
                pdf_dpi=100, layout_model="m", layout_backend="fake", ocr_backend="fake", ocr_request_concurrency=1
            ),
        )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("BUMBLEBEE_API_KEYS_JSON", raising=False)
    monkeypatch.setenv("BUMBLEBEE_API_KEY", "sekrit")
    return TestClient(build_api(FakeEngine()))


AUTH = {"Authorization": "Bearer sekrit"}


def test_health_needs_no_auth(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_parse_returns_markdown_layout_and_chunks(client):
    response = client.post("/v1/parse?filename=report.pdf", content=b"%PDF-fake", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-bumblebee-document-retention"] == "none"
    body = response.json()
    assert body["filename"] == "report.pdf"
    assert body["markdown"].startswith("# Title")
    assert body["stats"]["pages"] == 1
    # Stage timings surface only the stages that ran (None fields omitted).
    assert body["stats"]["timings"] == {"render": 0.1, "ocr": 0.3}
    # Private OCR keys are sanitized out of the returned layout — confidence included by default.
    regions = [region for page in body["layout"] for region in page]
    assert all("_ocr_usage" not in region and "_ocr_confidence" not in region for region in regions)
    (chunk,) = body["chunks"]
    assert chunk["chunk_id"] == "report#0000"
    assert chunk["section_path"] == ["Title"]
    assert body["request"]["id"] == response.headers["x-request-id"]
    assert body["request"]["document_retention"] == "none"


def test_include_region_metadata_exposes_confidence_only(client):
    body = client.post("/v1/parse?include_region_metadata=true", content=b"%PDF-fake", headers=AUTH).json()
    regions = [region for page in body["layout"] for region in page]
    assert any(region.get("_ocr_confidence") == 0.87 for region in regions)
    assert any(region.get("_ocr_confidence_before") == 0.61 for region in regions)
    # Usage/latency/status stay private even when confidence is opted in.
    assert all("_ocr_usage" not in region for region in regions)


def test_parse_can_skip_chunks(client):
    body = client.post("/v1/parse?chunks=false", content=b"%PDF-fake", headers=AUTH).json()
    assert "chunks" not in body


def test_missing_or_wrong_token_is_rejected(client):
    assert client.post("/v1/parse", content=b"%PDF-fake").status_code == 401
    assert client.post("/v1/parse", content=b"x", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_unconfigured_key_fails_closed(monkeypatch):
    monkeypatch.delenv("BUMBLEBEE_API_KEY", raising=False)
    monkeypatch.delenv("BUMBLEBEE_API_KEYS_JSON", raising=False)
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


def test_usage_and_audit_are_tenant_scoped_and_never_include_output(monkeypatch):
    monkeypatch.delenv("BUMBLEBEE_API_KEY", raising=False)
    monkeypatch.setenv(
        "BUMBLEBEE_API_KEYS_JSON",
        json.dumps({"alpha": {"key": "alpha-token", "monthly_page_limit": 10}, "beta": "beta-token"}),
    )
    client = TestClient(build_api(FakeEngine()))

    alpha = {"Authorization": "Bearer alpha-token"}
    beta = {"Authorization": "Bearer beta-token"}
    assert client.get("/v1/privacy").json() == {
        "document_retention": "none",
        "ocr_output_retention": "none",
        "audit_metadata_retention_days": 30,
    }
    assert client.post("/v1/parse", content=b"%PDF-fake", headers=alpha).status_code == 200

    alpha_usage = client.get("/v1/usage", headers=alpha).json()
    assert alpha_usage["tenant_id"] == "alpha"
    assert alpha_usage["requests"] == 1
    assert alpha_usage["pages"] == 1
    assert alpha_usage["pages_remaining"] == 9
    assert client.get("/v1/usage", headers=beta).json()["pages"] == 0

    events = client.get("/v1/audit", headers=alpha).json()["events"]
    assert len(events) == 1
    assert set(events[0]) == {
        "request_id",
        "created_at",
        "pages",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
    assert "Body text" not in json.dumps(events)


def test_monthly_page_limit_and_idempotency_protect_a_pilot(monkeypatch):
    monkeypatch.delenv("BUMBLEBEE_API_KEY", raising=False)
    monkeypatch.setenv(
        "BUMBLEBEE_API_KEYS_JSON",
        json.dumps({"trial": {"key": "trial-token", "monthly_page_limit": 1}}),
    )
    engine = FakeEngine()
    client = TestClient(build_api(engine))
    headers = {"Authorization": "Bearer trial-token", "Idempotency-Key": "upload-001"}

    assert client.post("/v1/parse", content=b"%PDF-fake", headers=headers).status_code == 200
    second_headers = {"Authorization": "Bearer trial-token", "Idempotency-Key": "upload-002"}
    assert client.post("/v1/parse", content=b"%PDF-fake", headers=second_headers).status_code == 429
    assert len(engine.calls) == 1


def test_duplicate_idempotency_key_does_not_repeat_ocr(monkeypatch):
    monkeypatch.delenv("BUMBLEBEE_API_KEY", raising=False)
    monkeypatch.setenv("BUMBLEBEE_API_KEYS_JSON", json.dumps({"alpha": "alpha-token"}))
    engine = FakeEngine()
    client = TestClient(build_api(engine))
    headers = {"Authorization": "Bearer alpha-token", "Idempotency-Key": "upload-001"}

    assert client.post("/v1/parse", content=b"%PDF-fake", headers=headers).status_code == 200
    retry = client.post("/v1/parse", content=b"%PDF-fake", headers=headers)
    assert retry.status_code == 409
    assert len(engine.calls) == 1
