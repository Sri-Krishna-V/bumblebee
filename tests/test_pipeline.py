"""End-to-end pipeline tests with fake render/layout/OCR services (GPU-free)."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from bumblebee.config import OcrConfig
from bumblebee.models import DocumentInput, Page, StageTiming
from bumblebee.pipeline import Pipeline
from tests.conftest import make_region

PAGE_SIZE = (200, 100)


class FakeRender:
    def __init__(self, pages_by_uri: dict[str, int]):
        self.pages_by_uri = pages_by_uri
        self.created_images: list[Image.Image] = []
        self.closed_count = 0

    async def page_count(self, pdf_bytes: bytes, config: OcrConfig) -> int:
        return self.pages_by_uri[pdf_bytes.decode()]

    async def render(self, pdf_bytes, config, page_indices=None):
        indices = list(page_indices) if page_indices is not None else range(self.pages_by_uri[pdf_bytes.decode()])
        pages = []
        for index in indices:
            image = Image.new("RGB", PAGE_SIZE, "white")
            self.created_images.append(image)
            original_close = image.close

            def tracking_close(orig=original_close):
                self.closed_count += 1
                orig()

            image.close = tracking_close
            pages.append(Page(page_index=index, width=PAGE_SIZE[0], height=PAGE_SIZE[1], image=image))
        return pages, StageTiming(queue_seconds=0.0, exec_seconds=0.01, wall_seconds=0.01)


class FakeLayout:
    backend = "fake"

    async def detect(self, pages):
        regions = {
            page.page_index: [
                make_region(page.page_index, 0, "text", "text", [100, 100, 900, 400]),
                make_region(page.page_index, 1, "image", "skip", [100, 500, 900, 900]),
            ]
            for page in pages
        }
        return regions, StageTiming(queue_seconds=0.0, exec_seconds=0.01, wall_seconds=0.01)


class FakeOcr:
    def __init__(self):
        self.recognized_count = 0

    async def recognize(self, prepared):
        from bumblebee.models import RecognizedRegion, TokenUsage

        self.recognized_count += len(prepared)
        return [
            RecognizedRegion(
                region=item.region,
                content=f"text p{item.region.page_index} r{item.region.region_index}",
                usage=TokenUsage(input_tokens=5, output_tokens=3, total_tokens=8),
                status_code=200,
                latency_ms=7,
            )
            for item in prepared
        ]


def doc(name: str, data: bytes | None = None) -> DocumentInput:
    return DocumentInput(uri=name, relative_path=f"{name}.pdf", input_id=f"id-{name}", data=data)


@pytest.fixture
def crop_executor():
    executor = ThreadPoolExecutor(max_workers=2)
    yield executor
    executor.shutdown(wait=False)


def build_pipeline(render, reader=None, config: OcrConfig | None = None, crop_executor=None, ocr=None):
    return Pipeline(
        config=config or OcrConfig(page_chunk_size=2, max_inflight_pdfs=4),
        render=render,
        layout=FakeLayout(),
        ocr=ocr or FakeOcr(),
        crop_executor=crop_executor,
        read=reader or (lambda document: document.uri.encode()),
        batch_id="batch-test",
    )


async def collect(pipeline, documents):
    return [processed async for processed in pipeline.stream(documents)]


async def test_documents_flow_end_to_end(crop_executor):
    render = FakeRender({"a": 3, "b": 1})
    pipeline = build_pipeline(render, crop_executor=crop_executor)
    processed = await collect(pipeline, [doc("a"), doc("b")])

    assert {p.document.input_id for p in processed} == {"id-a", "id-b"}
    by_id = {p.document.input_id: p for p in processed}
    result_a = by_id["id-a"].result
    assert result_a is not None
    assert result_a.page_count == 3
    assert result_a.ocr_region_count == 3  # one text region per page
    assert result_a.skipped_region_count == 3  # one skip region per page
    assert "text p0 r0" in result_a.markdown
    assert "<!-- page:start 3 -->" in result_a.markdown
    assert result_a.tokens.total_tokens == 24
    assert result_a.settings.layout_backend == "fake"

    stats = by_id["id-a"].stats
    assert stats["status"] == "succeeded"
    assert stats["pages"] == {"processed": 3}
    assert stats["regions"]["ocr"] == 3
    assert stats["ocr_requests"]["requests"] == 3
    assert stats["durations_seconds"]["render"] > 0
    assert stats["bytes"]["input_pdf"] == 1


async def test_page_images_are_closed(crop_executor):
    render = FakeRender({"a": 5})
    pipeline = build_pipeline(render, crop_executor=crop_executor)
    await collect(pipeline, [doc("a")])
    assert render.closed_count == len(render.created_images) == 5


async def test_page_chunking_respects_chunk_size(crop_executor):
    render = FakeRender({"a": 5})
    calls: list[list[int]] = []
    original_render = render.render

    async def spy_render(pdf_bytes, config, page_indices=None):
        calls.append(list(page_indices))
        return await original_render(pdf_bytes, config, page_indices)

    render.render = spy_render
    pipeline = build_pipeline(render, crop_executor=crop_executor)  # page_chunk_size=2
    await collect(pipeline, [doc("a")])
    assert calls == [[0, 1], [2, 3], [4]]


async def test_payload_documents_bypass_reader(crop_executor):
    render = FakeRender({"a": 1})
    reader_calls: list[str] = []

    def reader(document):
        reader_calls.append(document.uri)
        return document.uri.encode()

    pipeline = build_pipeline(render, reader=reader, crop_executor=crop_executor)
    processed = await collect(pipeline, [doc("a", data=b"a")])
    assert reader_calls == []
    assert processed[0].result is not None


async def test_empty_pdf_succeeds_with_empty_result(crop_executor):
    render = FakeRender({"a": 0})
    pipeline = build_pipeline(render, crop_executor=crop_executor)
    processed = await collect(pipeline, [doc("a")])
    result = processed[0].result
    assert result is not None
    assert result.page_count == 0
    assert result.markdown == ""
    assert processed[0].stats["status"] == "succeeded"


async def test_failed_document_yields_failure_stats_not_exception(crop_executor):
    class ExplodingRender(FakeRender):
        async def render(self, pdf_bytes, config, page_indices=None):
            if pdf_bytes == b"bad":
                raise ValueError("corrupt pdf")
            return await super().render(pdf_bytes, config, page_indices)

    render = ExplodingRender({"good": 1, "bad": 2})
    pipeline = build_pipeline(render, crop_executor=crop_executor)
    processed = await collect(pipeline, [doc("good"), doc("bad")])
    by_id = {p.document.input_id: p for p in processed}

    assert by_id["id-good"].result is not None
    failed = by_id["id-bad"]
    assert failed.result is None
    assert failed.stats["status"] == "failed"
    assert failed.stats["error"]["phase"] == "render"
    assert failed.stats["error"]["type"] == "ValueError"


async def test_failed_ocr_requests_fail_the_document(crop_executor):
    class FlakyOcr(FakeOcr):
        """Fails every region beyond page 0, so only multi-page docs fail."""

        async def recognize(self, prepared):
            from dataclasses import replace

            regions = await super().recognize(prepared)
            return [replace(r, content=None, status_code=503) if r.region.page_index >= 1 else r for r in regions]

    render = FakeRender({"good": 1, "bad": 3})
    pipeline = build_pipeline(render, crop_executor=crop_executor, ocr=FlakyOcr())
    processed = await collect(pipeline, [doc("good"), doc("bad")])
    by_id = {p.document.input_id: p for p in processed}

    assert by_id["id-good"].result is not None
    assert by_id["id-good"].stats["status"] == "succeeded"
    failed = by_id["id-bad"]
    assert failed.result is None
    assert failed.stats["status"] == "failed"
    assert failed.stats["error"]["phase"] == "ocr"
    assert failed.stats["error"]["type"] == "OcrError"
    assert "2/3 OCR requests failed" in failed.stats["error"]["message"]
    assert "503" in failed.stats["error"]["message"]


class ConfidenceOcr(FakeOcr):
    """Low confidence for the first ``main_regions`` regions, high afterwards (retry pass)."""

    def __init__(self, main_regions: int, low: float = 0.5, high: float = 0.95):
        super().__init__()
        self.main_regions = main_regions
        self.low = low
        self.high = high
        self.call_sizes: list[int] = []

    async def recognize(self, prepared):
        from dataclasses import replace

        self.call_sizes.append(len(prepared))
        regions = await super().recognize(prepared)
        confidence = self.low if self.recognized_count <= self.main_regions else self.high
        return [replace(r, confidence=confidence) for r in regions]


async def test_adaptive_retry_reocrs_lowest_confidence_within_budget(crop_executor):
    render = FakeRender({"a": 10})  # 10 pages -> 10 OCR text regions
    dpi_calls: list[tuple[int, list[int]]] = []
    original_render = render.render

    async def spy_render(pdf_bytes, config, page_indices=None):
        dpi_calls.append((config.pdf_dpi, list(page_indices) if page_indices is not None else []))
        return await original_render(pdf_bytes, config, page_indices)

    render.render = spy_render
    ocr = ConfidenceOcr(main_regions=10)
    pipeline = build_pipeline(render, crop_executor=crop_executor, ocr=ocr)
    (processed,) = await collect(pipeline, [doc("a")])

    # Budget: max(1, 10 // 10) = 1 region retried, on a page re-rendered at 2x DPI.
    assert ocr.call_sizes[-1] == 1
    assert sum(ocr.call_sizes) == 11
    retry_renders = [call for call in dpi_calls if call[0] == 200]
    assert len(retry_renders) == 1 and len(retry_renders[0][1]) == 1

    confidence = processed.stats["ocr_requests"]["confidence"]
    assert confidence["low"] == 9  # one region improved to 0.95, nine still at 0.5
    assert confidence["min"] == 0.5


async def test_adaptive_retry_keeps_the_better_result(crop_executor):
    render = FakeRender({"a": 1})
    ocr = ConfidenceOcr(main_regions=1)
    pipeline = build_pipeline(render, crop_executor=crop_executor, ocr=ocr)
    (processed,) = await collect(pipeline, [doc("a")])
    result = processed.result
    assert result is not None
    (text_region,) = [r for r in result.json[0] if r.get("_ocr_confidence") is not None]
    assert text_region["_ocr_confidence"] == 0.95
    assert text_region["_ocr_confidence_before"] == 0.5


async def test_adaptive_retry_disabled_by_config(crop_executor):
    render = FakeRender({"a": 1})
    ocr = ConfidenceOcr(main_regions=1)
    config = OcrConfig(page_chunk_size=2, max_inflight_pdfs=4, adaptive_retry=False)
    pipeline = build_pipeline(render, crop_executor=crop_executor, ocr=ocr, config=config)
    await collect(pipeline, [doc("a")])
    assert ocr.call_sizes == [1]


async def test_no_retry_when_confidence_is_high(crop_executor):
    render = FakeRender({"a": 2})
    ocr = ConfidenceOcr(main_regions=0)  # every call returns high confidence
    pipeline = build_pipeline(render, crop_executor=crop_executor, ocr=ocr)
    (processed,) = await collect(pipeline, [doc("a")])
    assert sum(ocr.call_sizes) == 2  # no retry pass
    assert processed.stats["ocr_requests"]["confidence"]["low"] == 0


async def test_reader_errors_are_failures(crop_executor):
    def reader(document):
        raise FileNotFoundError(document.uri)

    render = FakeRender({"a": 1})
    pipeline = build_pipeline(render, reader=reader, crop_executor=crop_executor)
    processed = await collect(pipeline, [doc("a")])
    assert processed[0].stats["status"] == "failed"
    assert processed[0].stats["error"]["phase"] == "read_input"
