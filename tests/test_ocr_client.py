"""VllmOcrClient retry behavior against a fake aiohttp session (no server)."""

import pytest

import bumblebee.ocr as ocr_module
from bumblebee.config import OcrConfig
from bumblebee.models import PreparedRegion
from bumblebee.ocr import VllmOcrClient
from tests.conftest import make_region

OK_DATA = {
    "choices": [{"message": {"content": "hello"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


class FakeResponse:
    def __init__(self, status: int, data=None, text: str = "", exc: Exception | None = None):
        self.status = status
        self._data = data or {}
        self._text = text
        self._exc = exc

    async def json(self):
        return self._data

    async def text(self):
        return self._text

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self

    async def __aexit__(self, *args):
        return False


class FakeSession:
    """Returns one queued FakeResponse per POST, recording calls and payloads."""

    def __init__(self, outcomes: list[FakeResponse]):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.payloads: list[dict] = []

    def post(self, url, *, json, timeout):
        self.calls += 1
        self.payloads.append(json)
        return self.outcomes.pop(0)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(ocr_module, "_RETRY_BACKOFF_SECONDS", 0.0)


def prepared() -> PreparedRegion:
    return PreparedRegion(
        region=make_region(0, 0, "text", "text", [100, 100, 900, 400]),
        image_url="data:image/jpeg;base64,eHg=",
        image_size_bytes=2,
    )


def client(session: FakeSession) -> VllmOcrClient:
    return VllmOcrClient(session=session, base_url="http://127.0.0.1:8000/v1", config=OcrConfig())


async def test_transient_status_is_retried_then_succeeds():
    session = FakeSession([FakeResponse(503, text="busy"), FakeResponse(200, data=OK_DATA)])
    (result,) = await client(session).recognize([prepared()])
    assert result.status_code == 200
    assert result.content == "hello"
    assert session.calls == 2


async def test_exhausted_retries_return_failure():
    session = FakeSession([FakeResponse(503, text="busy")] * ocr_module._RETRY_ATTEMPTS)
    ocr_client = client(session)
    (result,) = await ocr_client.recognize([prepared()])
    assert result.status_code == 503
    assert result.content is None
    assert session.calls == ocr_module._RETRY_ATTEMPTS
    snapshot = ocr_client.metrics_snapshot()
    assert snapshot["requests"]["retried"] == ocr_module._RETRY_ATTEMPTS - 1
    assert snapshot["requests"]["failed"] == 1


async def test_deterministic_client_error_is_not_retried():
    session = FakeSession([FakeResponse(400, text="bad payload")])
    (result,) = await client(session).recognize([prepared()])
    assert result.status_code == 400
    assert session.calls == 1


async def test_network_exception_is_retried():
    session = FakeSession([FakeResponse(0, exc=ConnectionError("reset")), FakeResponse(200, data=OK_DATA)])
    ocr_client = client(session)
    (result,) = await ocr_client.recognize([prepared()])
    assert result.status_code == 200
    assert session.calls == 2
    assert ocr_client.metrics_snapshot()["requests"]["retried"] == 1


async def test_confidence_computed_from_logprobs():
    import math

    data = {
        **OK_DATA,
        "choices": [
            {
                "message": {"content": "hello"},
                "logprobs": {"content": [{"token": "a", "logprob": -0.2}, {"token": "b", "logprob": -0.4}]},
            }
        ],
    }
    session = FakeSession([FakeResponse(200, data=data)])
    (result,) = await client(session).recognize([prepared()])
    assert result.confidence == pytest.approx(math.exp(-0.3))
    assert session.payloads[0]["logprobs"] is True  # requested by default


async def test_confidence_none_without_logprobs_block():
    session = FakeSession([FakeResponse(200, data=OK_DATA)])
    (result,) = await client(session).recognize([prepared()])
    assert result.confidence is None


async def test_logprobs_not_requested_when_disabled():
    session = FakeSession([FakeResponse(200, data=OK_DATA)])
    ocr_client = VllmOcrClient(
        session=session, base_url="http://127.0.0.1:8000/v1", config=OcrConfig(ocr_logprobs=False)
    )
    await ocr_client.recognize([prepared()])
    assert "logprobs" not in session.payloads[0]
