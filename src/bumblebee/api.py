"""Bumblebee hosted-API application (FastAPI), engine-agnostic and GPU-free.

The ASGI app is built by :func:`build_api` around any engine exposing
``async ocr(pdf_bytes) -> DocumentResult``, so route logic is unit-testable
with a fake engine. The Modal deployment wrapper lives in
:mod:`bumblebee.modal.api`; this module only needs ``fastapi`` (dev/image
dependency, not a core requirement).

Requests send raw PDF bytes as the body (``curl --data-binary @doc.pdf``) —
no multipart, no extra parser dependency. Auth is a single bearer token from
the ``BUMBLEBEE_API_KEY`` environment variable; requests are rejected when it
is unset (fail closed — the deployed URL is public).
"""

import os
import time
from typing import Any, Protocol

from fastapi import FastAPI, Header, HTTPException, Request

from bumblebee.chunks import DEFAULT_CHUNK_MAX_TOKENS, build_chunks
from bumblebee.models import DocumentResult, OcrError, output_stem_for
from bumblebee.stats import sanitize_layout_json

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # ponytail: single-request cap; batch runs handle bigger corpora


class OcrEngine(Protocol):
    """The one engine method the API needs."""

    async def ocr(self, pdf: bytes) -> DocumentResult:
        """OCR one in-memory PDF."""
        ...


def _require_api_key(authorization: str | None) -> None:
    expected = os.environ.get("BUMBLEBEE_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="BUMBLEBEE_API_KEY is not configured on the server")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


def build_api(engine: OcrEngine) -> FastAPI:
    """Build the bumblebee ASGI app around one started engine."""
    api = FastAPI(title="bumblebee", description="PDF in, layout-aware markdown + RAG chunks out.")

    @api.get("/health")
    async def health() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction] - FastAPI route
        return {"status": "ok"}

    @api.post("/v1/parse")
    async def parse(  # pyright: ignore[reportUnusedFunction] - FastAPI route
        request: Request,
        chunks: bool = True,
        chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
        filename: str = "document.pdf",
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_api_key(authorization)
        pdf = await request.body()
        if not pdf:
            raise HTTPException(status_code=400, detail="empty body; send raw PDF bytes")
        if len(pdf) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"PDF exceeds {MAX_UPLOAD_BYTES} bytes")

        started = time.perf_counter()
        try:
            result = await engine.ocr(pdf)
        except OcrError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        payload: dict[str, Any] = {
            "filename": filename,
            "markdown": result.markdown,
            "layout": sanitize_layout_json(result.json),
            "stats": {
                "pages": result.page_count,
                "regions": result.region_count,
                "ocr_regions": result.ocr_region_count,
                "tokens": {
                    "input_tokens": result.tokens.input_tokens,
                    "output_tokens": result.tokens.output_tokens,
                    "total_tokens": result.tokens.total_tokens,
                },
                "seconds": round(time.perf_counter() - started, 3),
            },
        }
        if chunks:
            payload["chunks"] = build_chunks(
                result.json,
                doc_path=filename,
                output_stem=output_stem_for(filename),
                max_tokens=chunk_max_tokens,
            )
        return payload

    return api
