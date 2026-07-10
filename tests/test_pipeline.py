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


def build_pipeline(render, reader=None, config: OcrConfig | None = None, crop_executor=None):
    return Pipeline(
        config=config or OcrConfig(page_chunk_size=2, max_inflight_pdfs=4),
        render=render,
        layout=FakeLayout(),
        ocr=FakeOcr(),
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


async def test_reader_errors_are_failures(crop_executor):
    def reader(document):
        raise FileNotFoundError(document.uri)

    render = FakeRender({"a": 1})
    pipeline = build_pipeline(render, reader=reader, crop_executor=crop_executor)
    processed = await collect(pipeline, [doc("a")])
    assert processed[0].stats["status"] == "failed"
    assert processed[0].stats["error"]["phase"] == "read_input"
