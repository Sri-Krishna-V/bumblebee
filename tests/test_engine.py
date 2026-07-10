"""Engine-level tests with injected fake stages (no GPU, no vLLM).

These drive `process_batch` — the call every run mode funnels through — and the
run layer it owns: output writing, completion markers, payload returns, and
resumability against a real local target directory.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from bumblebee.config import OcrConfig
from bumblebee.engine import DocumentEngine
from bumblebee.models import DocumentInput
from bumblebee.runs import select_active_documents
from bumblebee.storage import LocalStorage
from tests.test_pipeline import FakeLayout, FakeOcr, FakeRender


@pytest.fixture
def engine():
    eng = DocumentEngine()
    eng.render_engine = FakeRender({"a": 2, "b": 1})
    eng.layout_engine = FakeLayout()
    eng._crop_executor = ThreadPoolExecutor(max_workers=2)
    yield eng
    eng._crop_executor.shutdown(wait=False)


class FakeOcrClient(FakeOcr):
    def metrics_snapshot(self):
        return {"requests": {"completed": self.recognized_count}}


@pytest.fixture(autouse=True)
def fake_ocr_client(monkeypatch):
    """Swap the vLLM client for the echo fake; the engine never reaches a server."""
    import bumblebee.engine as engine_module

    monkeypatch.setattr(engine_module, "VllmOcrClient", lambda **kwargs: FakeOcrClient())


def doc(name: str) -> DocumentInput:
    return DocumentInput(uri=name, relative_path=f"{name}.pdf", input_id=f"id-{name}", data=name.encode())


async def test_process_batch_writes_outputs_and_summary(tmp_path, engine):
    target = str(tmp_path / "out")
    config = OcrConfig(max_inflight_pdfs=2, page_chunk_size=2)

    response = await engine.process_batch(
        [doc("a"), doc("b")], config, source="mem://", target=target, batch_id="b-1", write=True
    )

    assert response["batch_id"] == "b-1"
    summary = response["summary"]
    assert summary["documents"] == {"total": 2, "succeeded": 2, "failed": 0, "skipped": 0}
    assert summary["pages"]["processed"] == 3
    assert summary["ocr_client"] == {"requests": {"completed": 3}}

    markdown = (tmp_path / "out" / "a" / "content.md").read_text()
    assert "text p0 r0" in markdown
    layout = json.loads((tmp_path / "out" / "a" / "layout.json").read_text())
    assert len(layout) == 2
    assert all("_ocr_usage" not in region for page in layout for region in page)
    stats = json.loads((tmp_path / "out" / "a" / "stats.json").read_text())
    assert stats["status"] == "succeeded"
    assert stats["output"]["markdown"].replace("\\", "/").endswith("a/content.md")
    assert stats["durations_seconds"]["write"] > 0


async def test_completed_documents_are_skipped_on_rerun(tmp_path, engine):
    target = str(tmp_path / "out")
    config = OcrConfig()
    documents = [doc("a"), doc("b")]
    await engine.process_batch(documents, config, source="mem://", target=target, write=True)

    active, counts = select_active_documents(LocalStorage(), target, documents, config, None)
    assert active == []
    assert counts["completed_documents"] == 2


async def test_return_payloads_without_writing(tmp_path, engine):
    target = str(tmp_path / "out")
    response = await engine.process_batch(
        [doc("b")], OcrConfig(), source="mem://", target=target, write=False, return_payloads=True
    )
    (payload,) = response["documents"]
    assert payload["relative_path"] == "b.pdf"
    assert "text p0 r0" in payload["markdown"]
    assert payload["stats"]["status"] == "succeeded"
    assert not (tmp_path / "out").exists()  # nothing persisted


async def test_unstarted_engine_raises():
    engine = DocumentEngine()
    with pytest.raises(RuntimeError, match="start"):
        async for _ in engine.stream([doc("a")]):
            pass


class FakeProcess:
    """Stub vLLM subprocess whose poll() plays back the given exit codes, then repeats the last."""

    def __init__(self, *codes: int | None):
        self.codes = list(codes)

    def poll(self) -> int | None:
        return self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]


async def test_dead_vllm_server_fails_the_batch_at_entry(tmp_path, engine):
    engine.vllm_process = FakeProcess(137)
    with pytest.raises(RuntimeError, match="exited with code 137"):
        await engine.process_batch([doc("a")], OcrConfig(), source="mem://", target=str(tmp_path / "out"))


async def test_vllm_death_mid_batch_raises_after_stream(tmp_path, engine):
    # Alive at the entry check, dead by the post-stream check.
    engine.vllm_process = FakeProcess(None, 1)
    with pytest.raises(RuntimeError, match="exited with code 1"):
        await engine.process_batch([doc("a")], OcrConfig(), source="mem://", target=str(tmp_path / "out"))


async def test_live_vllm_process_does_not_interfere(tmp_path, engine):
    engine.vllm_process = FakeProcess(None)
    response = await engine.process_batch([doc("a")], OcrConfig(), source="mem://", target=str(tmp_path / "out"))
    assert response["summary"]["documents"]["succeeded"] == 1
