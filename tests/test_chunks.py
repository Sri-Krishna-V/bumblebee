"""Chunk building from formatted layout JSON, and chunks.jsonl output writing."""

import json
import os

from bumblebee.chunks import build_chunks, chunks_to_jsonl
from bumblebee.models import (
    DocumentInput,
    DocumentResult,
    DocumentTimings,
    ProcessedDoc,
    ResultSettings,
    TokenUsage,
)
from bumblebee.runs import output_paths_for_stem, write_outputs
from tests.conftest import FakeStorage


def region(label: str, native: str, content, bbox=(10, 10, 900, 100)):
    return {"label": label, "native_label": native, "content": content, "bbox_2d": list(bbox)}


def build(pages, max_tokens=512):
    return build_chunks(pages, doc_path="doc.pdf", output_stem="doc", max_tokens=max_tokens)


def test_section_hierarchy_and_ids():
    pages = [
        [
            region("text", "doc_title", "# Annual Report"),
            region("text", "text", "Intro paragraph."),
            region("text", "paragraph_title", "## Risk Factors"),
            region("text", "text", "Risky business."),
        ]
    ]
    chunks = build(pages)
    assert [c["chunk_id"] for c in chunks] == ["doc#0000", "doc#0001"]
    assert chunks[0]["section_path"] == ["Annual Report"]
    assert "# Annual Report" in chunks[0]["text"] and "Intro paragraph." in chunks[0]["text"]
    assert chunks[1]["section_path"] == ["Annual Report", "Risk Factors"]
    assert chunks[1]["text"].startswith("## Risk Factors")


def test_packing_respects_token_budget_but_not_mid_region():
    # ~10 tokens per region (40 chars); budget of 15 tokens fits only one per chunk.
    text = "x" * 40
    pages = [[region("text", "text", text), region("text", "text", text), region("text", "text", text)]]
    chunks = build(pages, max_tokens=15)
    assert len(chunks) == 3
    # A single oversized region still becomes one chunk (regions are never split).
    oversized = build([[region("text", "text", "y" * 400)]], max_tokens=15)
    assert len(oversized) == 1
    assert oversized[0]["token_estimate"] == 100


def test_headings_flush_the_buffer():
    pages = [
        [
            region("text", "text", "Before heading."),
            region("text", "paragraph_title", "## Section A"),
            region("text", "text", "After heading."),
        ]
    ]
    chunks = build(pages)
    assert len(chunks) == 2
    assert chunks[0]["section_path"] == []
    assert chunks[1]["section_path"] == ["Section A"]


def test_tables_and_formulas_are_atomic():
    pages = [
        [
            region("text", "text", "Lead-in text."),
            region("table", "table", "<table><tr><td>1</td></tr></table>"),
            region("formula", "display_formula", "$$\nE = mc^2\n$$"),
            region("text", "text", "Trailing text."),
        ]
    ]
    chunks = build(pages)
    assert [c["kind"] for c in chunks] == ["text", "table", "formula", "text"]
    assert chunks[1]["text"].startswith("<table")


def test_pages_and_bboxes_are_recorded_one_based():
    pages = [
        [region("text", "text", "Page one.", bbox=(1, 2, 3, 4))],
        [region("text", "text", "Page two.", bbox=(5, 6, 7, 8))],
    ]
    chunks = build(pages, max_tokens=512)
    assert len(chunks) == 1
    assert chunks[0]["pages"] == [1, 2]
    assert chunks[0]["bboxes"] == [{"page": 1, "bbox": [1, 2, 3, 4]}, {"page": 2, "bbox": [5, 6, 7, 8]}]


def test_null_content_and_empty_pages_are_skipped():
    pages = [[], [region("image", "image", None)], [region("text", "text", "   ")]]
    assert build(pages) == []


def test_jsonl_roundtrip():
    chunks = build([[region("text", "text", "Hello.")]])
    lines = chunks_to_jsonl(chunks).strip().splitlines()
    assert [json.loads(line) for line in lines] == chunks


def _processed_doc(pages_json) -> ProcessedDoc:
    document = DocumentInput(uri="doc.pdf", relative_path="doc.pdf", input_id="id-doc")
    result = DocumentResult(
        filename="doc.pdf",
        page_count=len(pages_json),
        region_count=1,
        ocr_region_count=1,
        skipped_region_count=0,
        json=pages_json,
        markdown="Hello.",
        tokens=TokenUsage(),
        timings=DocumentTimings(total_seconds=0.1),
        settings=ResultSettings(
            pdf_dpi=100, layout_model="m", layout_backend="fake", ocr_backend="fake", ocr_request_concurrency=1
        ),
    )
    stats = {"status": "succeeded", "durations_seconds": {"write": 0.0, "total": 0.1}}
    return ProcessedDoc(document=document, stats=stats, result=result)


def test_write_outputs_writes_chunks_jsonl_when_enabled():
    storage = FakeStorage()
    pages_json = [[region("text", "text", "Hello.")]]
    paths = output_paths_for_stem(storage, "out", "doc", emit_chunks=True)
    assert paths.chunks == "out/doc/chunks.jsonl"
    write_outputs(storage, _processed_doc(pages_json), paths)
    record = json.loads(storage.files["out/doc/chunks.jsonl"].decode("utf-8").strip())
    assert record["chunk_id"] == "doc#0000"
    assert record["doc"] == "doc.pdf"


def test_write_outputs_skips_chunks_when_disabled():
    storage = FakeStorage()
    paths = output_paths_for_stem(storage, "out", "doc")
    assert paths.chunks is None
    write_outputs(storage, _processed_doc([[region("text", "text", "Hello.")]]), paths)
    assert "out/doc/chunks.jsonl" not in storage.files
    assert "out/doc/content.md" in storage.files


def test_bumblebee_entry_point_defaults_chunks_on(monkeypatch):
    import bumblebee.bumblebee as bumblebee

    calls: list[str] = []
    monkeypatch.setattr(bumblebee, "app", lambda: calls.append("ran"))
    try:
        bumblebee.main()
        assert os.environ["BUMBLEBEE_EMIT_CHUNKS"] == "1"
        assert calls == ["ran"]
    finally:
        os.environ.pop("BUMBLEBEE_EMIT_CHUNKS", None)


def test_bumblebee_respects_preset_environment(monkeypatch):
    import bumblebee.bumblebee as bumblebee

    monkeypatch.setenv("BUMBLEBEE_EMIT_CHUNKS", "0")
    monkeypatch.setattr(bumblebee, "app", lambda: None)
    bumblebee.main()
    assert os.environ["BUMBLEBEE_EMIT_CHUNKS"] == "0"
