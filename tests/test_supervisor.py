"""Batch supervisor characterization: retries, refiltering, failure markers, summary."""

from typing import Any

from bumblebee.batches import BatchPolicy, supervise_batches
from bumblebee.models import DocumentInput


def doc(name: str) -> DocumentInput:
    return DocumentInput(uri=f"/in/{name}", relative_path=name, input_id=f"id-{name}", metadata={})


def success_stats(document: DocumentInput) -> dict[str, Any]:
    return {
        "input_id": document.input_id,
        "status": "succeeded",
        "pages": {"processed": 2},
        "regions": {"ocr": 3},
        "tokens": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "durations_seconds": {"total": 1.0},
    }


def batch_response(documents: list[DocumentInput]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stats = [success_stats(d) for d in documents]
    return {"durations_seconds": {"wall": 1.0}}, stats


class RecordingRunTarget:
    """RunTarget test double recording refilters, failure markers, and summaries."""

    def __init__(self, refilter_result=None):
        self.refilter_calls: list[list[str]] = []
        self.failed_marker_docs: list[str] = []
        self.summaries: list[dict[str, Any]] = []
        self._refilter_result = refilter_result

    def refilter(self, documents):
        self.refilter_calls.append([d.relative_path for d in documents])
        if self._refilter_result is not None:
            return [d for d in documents if d.relative_path in self._refilter_result]
        return documents

    def write_failed_markers(self, documents, error):
        self.failed_marker_docs.extend(d.relative_path for d in documents)
        return [{"input_id": d.input_id, "status": "failed"} for d in documents]

    def write_summary(self, summary):
        self.summaries.append(summary)


POLICY = BatchPolicy(max_docs=2, retries=1, retry_backoff_seconds=0.0)


def counts_for(documents: list[DocumentInput]) -> dict[str, int]:
    return {
        "total_documents": len(documents),
        "pending_documents": len(documents),
        "completed_documents": 0,
        "remaining_pending_documents": 0,
    }


async def test_all_batches_succeed():
    documents = [doc("a.pdf"), doc("b.pdf"), doc("c.pdf")]
    calls: list[list[str]] = []
    run_target = RecordingRunTarget()

    async def run_batch(batch_documents, batch):
        calls.append([d.relative_path for d in batch_documents])
        return batch_response(batch_documents)

    summary = await supervise_batches(
        source="/in",
        target="/out",
        policy=POLICY,
        documents=documents,
        counts=counts_for(documents),
        run_batch=run_batch,
        run_target=run_target,
    )
    assert calls == [["a.pdf", "b.pdf"], ["c.pdf"]]
    assert summary["documents"] == {"total": 3, "succeeded": 3, "failed": 0, "skipped": 0}
    assert summary["pages"]["processed"] == 6
    assert summary["status"] == "completed"
    assert len(run_target.summaries) == 1
    assert len(summary["batches"]["items"]) == 2
    assert all(item["status"] == "completed" for item in summary["batches"]["items"])
    assert summary["durations_seconds"]["processing_wall"] == 2.0


async def test_retry_then_success_with_refilter():
    documents = [doc("a.pdf"), doc("b.pdf")]
    attempts = 0
    # Simulate: a.pdf finished before the preemption, so the retry drops it.
    run_target = RecordingRunTarget(refilter_result={"b.pdf"})

    async def run_batch(batch_documents, batch):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("preempted")
        return batch_response(batch_documents)

    summary = await supervise_batches(
        source="/in",
        target="/out",
        policy=POLICY,
        documents=documents,
        counts=counts_for(documents),
        run_batch=run_batch,
        run_target=run_target,
    )
    assert attempts == 2
    assert run_target.refilter_calls == [["a.pdf", "b.pdf"]]
    result_ids = {r["input_id"] for r in summary["results"]}
    assert result_ids == {"id-b.pdf"}


async def test_exhausted_retries_marks_failed():
    documents = [doc("a.pdf")]
    run_target = RecordingRunTarget()

    async def run_batch(batch_documents, batch):
        raise RuntimeError("boom")

    summary = await supervise_batches(
        source="/in",
        target="/out",
        policy=BatchPolicy(retries=1, retry_backoff_seconds=0.0),
        documents=documents,
        counts=counts_for(documents),
        run_batch=run_batch,
        run_target=run_target,
    )
    assert run_target.failed_marker_docs == ["a.pdf"]
    assert summary["status"] == "completed_with_failures"
    assert summary["documents"]["failed"] == 1
    (item,) = summary["batches"]["items"]
    assert item["status"] == "failed"
    assert "RuntimeError" in item["error"]


async def test_completed_documents_fold_into_skipped():
    documents = [doc("a.pdf")]
    run_target = RecordingRunTarget()

    async def run_batch(batch_documents, batch):
        return batch_response(batch_documents)

    counts = counts_for(documents)
    counts["completed_documents"] = 4

    summary = await supervise_batches(
        source="/in",
        target="/out",
        policy=POLICY,
        documents=documents,
        counts=counts,
        run_batch=run_batch,
        run_target=run_target,
    )
    assert summary["documents"]["total"] == 5
    assert summary["documents"]["skipped"] == 4
