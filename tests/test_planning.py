"""Batch planning and resumability filtering characterization."""

from bumblebee.batches import BatchPolicy, plan_document_batches
from bumblebee.config import OcrConfig
from bumblebee.models import DocumentInput
from bumblebee.runs import output_paths_for_document, select_active_documents
from tests.conftest import FakeStorage

TARGET = "/out"


def doc(name: str, size: int | None = None, pages: int | None = None) -> DocumentInput:
    metadata = {}
    if size is not None:
        metadata["size_bytes"] = size
    if pages is not None:
        metadata["pages"] = pages
    return DocumentInput(uri=f"/in/{name}", relative_path=name, input_id=f"id-{name}", metadata=metadata)


def config(**kwargs) -> OcrConfig:
    return OcrConfig(**kwargs)


class TestPlanBatches:
    def test_max_docs_splits(self):
        batches = plan_document_batches([doc(f"{i}.pdf") for i in range(5)], BatchPolicy(max_docs=2))
        assert [len(b.documents) for b in batches] == [2, 2, 1]

    def test_max_bytes_splits(self):
        documents = [doc("a.pdf", size=600), doc("b.pdf", size=600), doc("c.pdf", size=100)]
        batches = plan_document_batches(documents, BatchPolicy(max_docs=10, max_bytes=1000))
        assert [len(b.documents) for b in batches] == [1, 2]
        assert batches[1].bytes_estimate == 700

    def test_max_pages_splits(self):
        documents = [doc("a.pdf", pages=30), doc("b.pdf", pages=30), doc("c.pdf", pages=30)]
        batches = plan_document_batches(documents, BatchPolicy(max_docs=10, max_pages=50))
        assert [len(b.documents) for b in batches] == [1, 1, 1]

    def test_batch_ids_are_deterministic(self):
        documents = [doc(f"{i}.pdf") for i in range(3)]
        first = plan_document_batches(documents, BatchPolicy(max_docs=2))
        second = plan_document_batches(documents, BatchPolicy(max_docs=2))
        assert [b.batch_id for b in first] == [b.batch_id for b in second]
        assert first[0].batch_id.startswith("batch-00001-")

    def test_oversized_single_document_still_batched(self):
        batches = plan_document_batches([doc("big.pdf", size=10_000)], BatchPolicy(max_bytes=100))
        assert len(batches) == 1

    def test_payload_bytes_count_toward_size(self):
        documents = [
            DocumentInput(uri=f"mem://{i}", relative_path=f"{i}.pdf", input_id=f"id-{i}", data=b"x" * 600)
            for i in range(2)
        ]
        batches = plan_document_batches(documents, BatchPolicy(max_docs=10, max_bytes=1000))
        assert [len(b.documents) for b in batches] == [1, 1]


def write_complete_outputs(storage: FakeStorage, document: DocumentInput, cfg: OcrConfig, status="succeeded"):
    paths = output_paths_for_document(storage, TARGET, document)
    storage.write_text(paths.markdown, "# md")
    storage.write_json(paths.json, [])
    storage.write_json(paths.stats, {"status": status})
    return paths


class TestSelectActive:
    def test_empty_target_all_active(self):
        storage = FakeStorage()
        documents = [doc("a.pdf"), doc("b.pdf")]
        active, counts = select_active_documents(storage, TARGET, documents, config(), None)
        assert active == documents
        assert counts["completed_documents"] == 0
        assert counts["pending_documents"] == 2

    def test_completed_document_skipped(self):
        storage = FakeStorage()
        documents = [doc("a.pdf"), doc("b.pdf")]
        write_complete_outputs(storage, documents[0], config())
        active, counts = select_active_documents(storage, TARGET, documents, config(), None)
        assert active == [documents[1]]
        assert counts["completed_documents"] == 1

    def test_failed_stats_stay_active(self):
        storage = FakeStorage()
        documents = [doc("a.pdf")]
        write_complete_outputs(storage, documents[0], config(), status="failed")
        active, _ = select_active_documents(storage, TARGET, documents, config(), None)
        assert active == documents

    def test_corrupt_stats_stay_active(self):
        storage = FakeStorage()
        documents = [doc("a.pdf")]
        paths = write_complete_outputs(storage, documents[0], config())
        storage.files[paths.stats] = b"{not json"
        active, _ = select_active_documents(storage, TARGET, documents, config(), None)
        assert active == documents

    def test_missing_markdown_stays_active(self):
        storage = FakeStorage()
        documents = [doc("a.pdf")]
        paths = write_complete_outputs(storage, documents[0], config())
        del storage.files[paths.markdown]
        active, _ = select_active_documents(storage, TARGET, documents, config(), None)
        assert active == documents

    def test_force_reprocesses_everything(self):
        storage = FakeStorage()
        documents = [doc("a.pdf")]
        write_complete_outputs(storage, documents[0], config(force=True))
        active, _ = select_active_documents(storage, TARGET, documents, config(force=True), None)
        assert active == documents

    def test_limit_applies_after_completion_filtering(self):
        storage = FakeStorage()
        documents = [doc("a.pdf"), doc("b.pdf"), doc("c.pdf")]
        write_complete_outputs(storage, documents[0], config())
        active, counts = select_active_documents(storage, TARGET, documents, config(), 1)
        assert active == [documents[1]]
        assert counts["remaining_pending_documents"] == 1
