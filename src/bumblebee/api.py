"""Privacy-first, engine-agnostic FastAPI application for Bumblebee.

The ASGI app is built around an object exposing ``async ocr(pdf_bytes)`` so
route logic stays testable with a fake engine. It accepts raw PDF bytes and
never writes them, rendered pages, Markdown, layout JSON, or chunks to its
control-plane store. The Modal deployment wrapper lives in
``bumblebee.modal.api``; this module only needs FastAPI in development and in
the hosted image.
"""

import time
import uuid
from collections.abc import Callable
from typing import Any, Protocol

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from bumblebee.chunks import DEFAULT_CHUNK_MAX_TOKENS, build_chunks
from bumblebee.models import DocumentResult, OcrError, output_stem_for
from bumblebee.pilot import ApiKeyRegistry, ApiPrincipal, PilotConfigError, PilotUsageStore
from bumblebee.stats import PRIVATE_JSON_KEYS, sanitize_layout_json

# Confidence is opt-in public metadata; the other _ocr_* keys stay private.
_METADATA_STRIP = PRIVATE_JSON_KEYS - {"_ocr_confidence", "_ocr_confidence_before"}

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # Single-request cap; batch runs handle bigger corpora.


class OcrEngine(Protocol):
    """The one engine method the API needs."""

    async def ocr(self, pdf: bytes) -> DocumentResult:
        """OCR one in-memory PDF."""
        ...


def build_api(
    engine: OcrEngine,
    *,
    key_registry: ApiKeyRegistry | None = None,
    usage_store: PilotUsageStore | None = None,
    usage_checkpoint: Callable[[], object] | None = None,
) -> FastAPI:
    """Build Bumblebee's ASGI app around one started engine.

    ``key_registry`` and ``usage_store`` are injectable to keep API tests
    GPU-free. ``usage_checkpoint`` lets Modal commit the metadata-only ledger
    after a successful parse. Production derives the first two from environment
    variables and fails closed when their configuration is invalid.
    """
    config_error: str | None = None
    try:
        registry = key_registry or ApiKeyRegistry.from_env()
        store = usage_store or PilotUsageStore.from_env()
    except PilotConfigError as exc:
        registry = None
        store = None
        config_error = str(exc)

    api = FastAPI(title="bumblebee", description="PDF in, layout-aware markdown + RAG chunks out.")

    def require_principal(authorization: str | None) -> ApiPrincipal:
        """Authenticate a design partner or return a safe operational error."""
        if config_error is not None:
            raise HTTPException(status_code=503, detail="Bumblebee API configuration is invalid")
        if registry is None or not registry.configured:
            raise HTTPException(status_code=503, detail="Bumblebee API authentication is not configured")
        principal = registry.authenticate(authorization)
        if principal is None:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")
        return principal

    @api.get("/health")
    async def health() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction] - FastAPI route
        return {"status": "ok"}

    @api.get("/v1/privacy")
    async def privacy() -> dict[str, str | int]:  # pyright: ignore[reportUnusedFunction] - FastAPI route
        """Explain document-retention behavior without requiring a key."""
        retention_days = store.retention_days if store is not None else 30
        return {
            "document_retention": "none",
            "ocr_output_retention": "none",
            "audit_metadata_retention_days": retention_days,
        }

    @api.get("/v1/usage")
    async def usage(  # pyright: ignore[reportUnusedFunction] - FastAPI route
        authorization: str | None = Header(default=None),
    ) -> dict[str, int | str | None]:
        """Show only the authenticated tenant's current-month totals."""
        principal = require_principal(authorization)
        assert store is not None
        return store.usage(principal).as_dict()

    @api.get("/v1/audit")
    async def audit(  # pyright: ignore[reportUnusedFunction] - FastAPI route
        limit: int = 20,
        authorization: str | None = Header(default=None),
    ) -> dict[str, list[dict[str, int | str]]]:
        """Return recent audit metadata, never documents or OCR output."""
        principal = require_principal(authorization)
        assert store is not None
        return {"events": store.audit(principal, limit=limit)}

    @api.post("/v1/parse")
    async def parse(  # pyright: ignore[reportUnusedFunction] - FastAPI route
        request: Request,
        chunks: bool = True,
        chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
        filename: str = "document.pdf",
        include_region_metadata: bool = False,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        """Parse one PDF and return an unpersisted RAG-ready result."""
        principal = require_principal(authorization)
        assert store is not None
        pdf = await request.body()
        if not pdf:
            raise HTTPException(status_code=400, detail="empty body; send raw PDF bytes")
        if len(pdf) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"PDF exceeds {MAX_UPLOAD_BYTES} bytes")

        before = store.usage(principal)
        if before.pages_remaining == 0:
            raise HTTPException(status_code=429, detail="monthly page limit reached; contact Bumblebee to continue")

        request_id = uuid.uuid4().hex
        try:
            claimed = store.claim_idempotency_key(principal, idempotency_key, request_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not claimed:
            raise HTTPException(status_code=409, detail="duplicate Idempotency-Key; the original request is retained")

        started = time.perf_counter()
        recorded = False
        try:
            result = await engine.ocr(pdf)
            duration_ms = round((time.perf_counter() - started) * 1000)
            after = store.record(
                principal,
                request_id=request_id,
                pages=result.page_count,
                duration_ms=duration_ms,
                input_tokens=result.tokens.input_tokens,
                output_tokens=result.tokens.output_tokens,
                total_tokens=result.tokens.total_tokens,
            )
            recorded = True
            if usage_checkpoint is not None:
                usage_checkpoint()
            if after.monthly_page_limit is not None and after.pages > after.monthly_page_limit:
                # Page count is only known after parsing. Meter the request but
                # withhold its output once it crosses the cap, preventing a
                # pilot from silently doing unbounded work.
                raise HTTPException(
                    status_code=429,
                    detail="monthly page limit reached while processing; contact Bumblebee to continue",
                )

            payload: dict[str, Any] = {
                "filename": filename,
                "markdown": result.markdown,
                "layout": sanitize_layout_json(
                    result.json, strip=_METADATA_STRIP if include_region_metadata else PRIVATE_JSON_KEYS
                ),
                "stats": {
                    "pages": result.page_count,
                    "regions": result.region_count,
                    "ocr_regions": result.ocr_region_count,
                    "tokens": {
                        "input_tokens": result.tokens.input_tokens,
                        "output_tokens": result.tokens.output_tokens,
                        "total_tokens": result.tokens.total_tokens,
                    },
                    "seconds": round(duration_ms / 1000, 3),
                    "timings": {
                        stage: round(value, 3)
                        for stage, value in {
                            "wait": result.timings.wait_seconds,
                            "read": result.timings.read_seconds,
                            "render": result.timings.render_seconds,
                            "layout": result.timings.layout_seconds,
                            "crop": result.timings.crop_seconds,
                            "ocr": result.timings.ocr_seconds,
                            "format": result.timings.format_seconds,
                        }.items()
                        if value is not None
                    },
                },
                "request": {
                    "id": request_id,
                    "document_retention": "none",
                    "audit_metadata_retention_days": store.retention_days,
                },
            }
            if chunks:
                payload["chunks"] = build_chunks(
                    result.json,
                    doc_path=filename,
                    output_stem=output_stem_for(filename),
                    max_tokens=chunk_max_tokens,
                )
            return JSONResponse(
                content=payload,
                headers={
                    "Cache-Control": "no-store",
                    "X-Bumblebee-Document-Retention": "none",
                    "X-Request-ID": request_id,
                },
            )
        except OcrError as exc:
            store.release_idempotency_key(principal, idempotency_key)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            if not recorded:
                store.release_idempotency_key(principal, idempotency_key)
            raise
        except Exception:
            if not recorded:
                store.release_idempotency_key(principal, idempotency_key)
            raise

    return api
