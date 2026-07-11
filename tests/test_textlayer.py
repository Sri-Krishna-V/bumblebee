"""Born-digital text-layer extraction and the pipeline hybrid (GPU-free)."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from bumblebee.config import OcrConfig
from bumblebee.models import Task
from bumblebee.textlayer import extract_region_texts, satisfy_regions_from_text_layer
from tests.conftest import make_region, make_text_pdf
from tests.test_pipeline import FakeOcr, FakeRender, build_pipeline, collect, doc

# 612x792pt page. Text baselines at y=720 (top) and y=100 (bottom), bottom-up.
PDF = make_text_pdf([(72, 720, "Hello embedded text"), (72, 100, "Bottom line text")])
# Normalized 0..1000 top-down boxes over those lines.
TOP_BOX = (40, 40, 950, 130)
BOTTOM_BOX = (40, 820, 950, 920)
EMPTY_BOX = (40, 400, 950, 500)


def test_extract_region_texts_maps_topdown_boxes_to_pdf_points():
    regions = [
        make_region(0, 0, "text", "text", list(TOP_BOX)),
        make_region(0, 1, "text", "text", list(BOTTOM_BOX)),
        make_region(0, 2, "text", "text", list(EMPTY_BOX)),
    ]
    texts = extract_region_texts(PDF, {0: regions})
    assert texts == {0: ["Hello embedded text", "Bottom line text", ""]}


def test_satisfy_uses_layer_when_coverage_is_high():
    text_a = make_region(0, 0, "text", "text", list(TOP_BOX))
    text_b = make_region(0, 1, "text", "text", list(BOTTOM_BOX))
    table = make_region(0, 2, "table", "table", list(EMPTY_BOX))
    satisfied, remaining = satisfy_regions_from_text_layer({0: ["Hello", "World"]}, {0: [text_a, text_b, table]})
    assert [r.content for r in satisfied] == ["Hello", "World"]
    assert all(r.status_code is None for r in satisfied)
    assert remaining == {0: [table]}  # tables always go to OCR


def test_satisfy_falls_back_when_coverage_is_low():
    regions = [make_region(0, i, "text", "text", list(TOP_BOX)) for i in range(4)]
    satisfied, remaining = satisfy_regions_from_text_layer({0: ["only one", "", "", ""]}, {0: regions})
    assert satisfied == []
    assert remaining == {0: regions}


def test_satisfy_ignores_pages_without_extraction():
    region = make_region(0, 0, "text", "text", list(TOP_BOX))
    satisfied, remaining = satisfy_regions_from_text_layer({}, {0: [region]})
    assert satisfied == []
    assert remaining == {0: [region]}


class TextLayerRender(FakeRender):
    """FakeRender that serves embedded text for every eligible region."""

    async def extract_region_texts(self, pdf_bytes, regions_by_page):
        return {
            page: [f"embedded p{region.page_index}" for region in regions] for page, regions in regions_by_page.items()
        }


class ExplodingTextLayerRender(FakeRender):
    async def extract_region_texts(self, pdf_bytes, regions_by_page):
        raise ValueError("broken text layer")


@pytest.fixture
def crop_executor():
    executor = ThreadPoolExecutor(max_workers=2)
    yield executor
    executor.shutdown(wait=False)


async def test_pipeline_serves_text_regions_from_layer(crop_executor):
    render = TextLayerRender({"a": 2})
    ocr = FakeOcr()
    pipeline = build_pipeline(render, crop_executor=crop_executor, ocr=ocr)
    (processed,) = await collect(pipeline, [doc("a")])

    assert ocr.recognized_count == 0  # every text region satisfied without OCR
    result = processed.result
    assert result is not None
    assert "embedded p0" in result.markdown and "embedded p1" in result.markdown
    assert processed.stats["text_layer"] == {"regions": 2}
    assert processed.stats["status"] == "succeeded"


async def test_pipeline_text_layer_off_uses_ocr(crop_executor):
    render = TextLayerRender({"a": 1})
    ocr = FakeOcr()
    config = OcrConfig(page_chunk_size=2, max_inflight_pdfs=4, text_layer="off")
    pipeline = build_pipeline(render, crop_executor=crop_executor, ocr=ocr, config=config)
    (processed,) = await collect(pipeline, [doc("a")])
    assert ocr.recognized_count == 1
    assert processed.stats["text_layer"] == {"regions": 0}


async def test_pipeline_falls_back_to_ocr_on_extraction_error(crop_executor):
    render = ExplodingTextLayerRender({"a": 1})
    ocr = FakeOcr()
    pipeline = build_pipeline(render, crop_executor=crop_executor, ocr=ocr)
    (processed,) = await collect(pipeline, [doc("a")])
    assert processed.stats["status"] == "succeeded"
    assert ocr.recognized_count == 1


def test_text_layer_config_is_validated():
    with pytest.raises(ValueError, match="text_layer"):
        OcrConfig(text_layer="sometimes")
    assert OcrConfig(text_layer="off").text_layer == "off"


def test_task_text_is_the_only_eligible_task():
    formula = make_region(0, 0, "display_formula", "formula", list(TOP_BOX))
    satisfied, remaining = satisfy_regions_from_text_layer({0: []}, {0: [formula]})
    assert satisfied == []
    assert remaining == {0: [formula]}
    assert formula.task_type == Task.FORMULA
