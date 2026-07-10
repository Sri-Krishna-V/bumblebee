"""Stats builders characterization: the persisted schema that resumability relies on."""

from bumblebee.models import (
    DocumentInput,
    DocumentResult,
    DocumentTimings,
    ResultSettings,
    TokenUsage,
)
from bumblebee.stats import build_failed_stats, build_run_summary, build_success_stats

DOC = DocumentInput(uri="/in/a.pdf", relative_path="a.pdf", input_id="id-a", metadata={"size_bytes": 10})


def result() -> DocumentResult:
    return DocumentResult(
        filename="a.pdf",
        page_count=2,
        region_count=5,
        ocr_region_count=4,
        skipped_region_count=1,
        json=[[{"label": "text"}], [{"label": "table"}]],
        markdown="# hi",
        tokens=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        timings=DocumentTimings(
            total_seconds=4.0,
            wait_seconds=0.1,
            read_seconds=0.2,
            render_seconds=0.5,
            layout_seconds=0.7,
            crop_seconds=0.3,
            ocr_seconds=2.0,
            format_seconds=0.2,
        ),
        settings=ResultSettings(
            pdf_dpi=100,
            layout_model="PaddlePaddle/PP-DocLayoutV3_safetensors",
            layout_backend="onnx",
            ocr_backend="vllm_online_server",
            ocr_request_concurrency=1024,
        ),
        region_counts_by_page={"0": 3, "1": 2},
    )


def test_success_stats_schema():
    stats = build_success_stats(document=DOC, result=result(), total_seconds=4.2)
    assert stats["input_id"] == "id-a"
    assert stats["status"] == "succeeded"
    assert stats["input"]["relative_path"] == "a.pdf"
    assert stats["input"]["metadata"] == {"size_bytes": 10}
    assert "data" not in stats["input"]
    assert stats["pages"] == {"processed": 2}
    assert stats["regions"]["detected"] == 5
    assert stats["regions"]["ocr"] == 4
    assert stats["regions"]["skipped"] == 1
    assert stats["regions"]["by_label"] == {"table": 1, "text": 1}
    assert stats["tokens"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    durations = stats["durations_seconds"]
    assert durations["total"] == 4.2
    assert durations["render"] == 0.5
    assert durations["layout"] == 0.7
    assert durations["crop"] == 0.3
    assert durations["ocr"] == 2.0
    assert "created_at" in stats


def test_payload_bytes_never_reach_stats():
    document = DocumentInput(uri="mem://a", relative_path="a.pdf", input_id="id-a", data=b"\x00binary")
    stats = build_success_stats(document=document, result=result(), total_seconds=1.0)
    assert "data" not in stats["input"]


def test_failed_stats_schema():
    stats = build_failed_stats(document=DOC, phase="render", error=ValueError("bad pdf"), total_seconds=1.5)
    assert stats["status"] == "failed"
    assert stats["error"]["phase"] == "render"
    assert stats["error"]["type"] == "ValueError"
    assert stats["error"]["message"] == "bad pdf"
    assert "traceback_preview" in stats["error"]
    assert stats["durations_seconds"]["total"] == 1.5


def test_run_summary_totals():
    results = [
        {
            "status": "succeeded",
            "pages": {"processed": 3},
            "regions": {"ocr": 6},
            "tokens": {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
        },
        {"status": "failed"},
    ]
    summary = build_run_summary(results=results, source="/in", target="/out", wall_seconds=2.0)
    assert summary["status"] == "completed_with_failures"
    assert summary["documents"] == {"total": 2, "succeeded": 1, "failed": 1, "skipped": 0}
    assert summary["pages"]["processed"] == 3
    assert summary["tokens"]["total_tokens"] == 42
    assert summary["throughput"]["pages_per_second"] == 1.5
