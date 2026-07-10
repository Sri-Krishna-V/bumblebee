# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUntypedClassDecorator=false, reportUntypedFunctionDecorator=false
"""Modal workers: a thin GpuWorker delegating to DocumentEngine, plus a storage helper.

The GPU container loads the layout model first, then starts the vLLM OCR server
(both on one GPU), then serves many documents concurrently through one shared
:class:`bumblebee.engine.DocumentEngine`.
"""

import asyncio
import logging
from typing import Any

import modal

from bumblebee.batches import BatchPolicy, DocumentBatch, batch_summary, summary_results, supervise_batches
from bumblebee.config import OcrConfig
from bumblebee.engine import DocumentEngine
from bumblebee.logging import configure_logging
from bumblebee.modal.app import (
    CPU_CORES,
    MODAL,
    MODAL_SECRETS,
    app,
    hf_cache_vol,
    image,
    storage_image,
    vllm_cache_vol,
)
from bumblebee.models import DocumentInput
from bumblebee.runs import RunTarget, select_active_documents
from bumblebee.storage import resolve_storage

logger = logging.getLogger(__name__)


@app.cls(
    image=image,
    gpu=MODAL.gpu,
    cpu=CPU_CORES,
    timeout=MODAL.timeout_seconds,
    startup_timeout=MODAL.startup_timeout_seconds,
    scaledown_window=MODAL.scaledown_window_seconds,
    max_containers=MODAL.max_containers,
    secrets=MODAL_SECRETS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class GpuWorker:
    """One GPU container running both the layout and OCR models."""

    @modal.enter()
    def start(self) -> None:
        """Load the layout model, then start and warm the vLLM OCR server."""
        configure_logging()
        self.engine = DocumentEngine().start()

    @modal.method()
    async def process_batch(
        self,
        documents: list[DocumentInput],
        config: OcrConfig,
        source: str,
        target: str,
        batch_id: str | None = None,
        write: bool = True,
        return_payloads: bool = False,
    ) -> dict[str, Any]:
        """Process one pre-filtered batch of documents (bytes payloads or cloud refs).

        ``write`` persists outputs from the worker (cloud targets the worker can
        reach); ``return_payloads`` ships outputs back to the caller (local
        targets it cannot).
        """
        return await self.engine.process_batch(
            documents,
            config,
            source=source,
            target=target,
            batch_id=batch_id,
            write=write,
            return_payloads=return_payloads,
        )

    @modal.exit()
    def stop(self) -> None:
        """Tear down the vLLM server and worker threads."""
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.stop()


@app.cls(
    image=storage_image,
    cpu=1.0,
    timeout=MODAL.timeout_seconds,
    startup_timeout=MODAL.startup_timeout_seconds,
    scaledown_window=MODAL.scaledown_window_seconds,
    secrets=MODAL_SECRETS,
    nonpreemptible=True,
)
class StorageWorker:
    """Small Modal-side storage helper for cloud listing/filtering/writes."""

    @modal.enter()
    def start(self) -> None:
        """Configure logging for storage helper calls."""
        configure_logging()

    @modal.method()
    def list_pdfs(self, source: str) -> list[DocumentInput]:
        """List source PDFs using Modal-side cloud credentials."""
        return resolve_storage(source).list_pdfs(source)

    @modal.method()
    def filter_active(
        self,
        documents: list[DocumentInput],
        target: str,
        config: OcrConfig,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Filter documents against a target using Modal-side cloud credentials."""
        target_storage = resolve_storage(target)
        active, counts = select_active_documents(target_storage, target, documents, config, limit)
        return {"documents": active, "counts": counts}

    @modal.method()
    def write_run_summary(self, target: str, summary: dict[str, Any]) -> None:
        """Write the run summary using Modal-side cloud credentials."""
        storage = resolve_storage(target)
        storage.write_json(storage.join(target, "_run_summary.json"), summary)

    @modal.method()
    def run_detached(
        self,
        source: str,
        target: str,
        config: OcrConfig,
        limit: int | None = None,
        batch_policy: BatchPolicy | None = None,
    ) -> dict[str, Any]:
        """Non-preemptible CPU supervisor that pulls cloud batches to GPU workers.

        Lives on a cheap, non-preemptible container so the run survives GPU
        preemption: each batch is a separate ``process_batch`` call on a fresh
        GPU container, and a preempted batch is retried (re-filtering the target
        first so finished documents are not redone).
        """
        policy = batch_policy or BatchPolicy()
        source_storage = resolve_storage(source)
        target_storage = resolve_storage(target)
        documents = source_storage.list_pdfs(source)
        active, counts = select_active_documents(target_storage, target, documents, config, limit)
        run_target = RunTarget(target_storage, target, config, failed_phase="modal_cloud_batch")
        gpu_worker = GpuWorker()

        async def run_batch(
            batch_documents: list[DocumentInput], batch: DocumentBatch
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            response = await asyncio.to_thread(
                gpu_worker.process_batch.remote,
                batch_documents,
                config,
                source,
                target,
                batch.batch_id,
                True,
                False,
            )
            summary = batch_summary(response)
            return summary, summary_results(summary)

        return asyncio.run(
            supervise_batches(
                source=source,
                target=target,
                policy=policy,
                documents=active,
                counts=counts,
                run_batch=run_batch,
                run_target=run_target,
            )
        )
