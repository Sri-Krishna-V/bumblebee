"""Bumblebee Studio app state: upload -> parse -> results.

Two parse engines, picked at runtime:

- **Hosted** — POST raw PDF bytes to the Modal-deployed bumblebee API
  (``BUMBLEBEE_API_URL`` + ``BUMBLEBEE_API_KEY``). Full layout + OCR pipeline.
- **Local demo** — born-digital text-layer extraction via pypdfium2, chunked
  with the same :func:`bumblebee.chunks.build_chunks` the product ships. No
  GPU, so scanned PDFs are rejected with a clear message instead of garbage.
"""

import asyncio
import os
import time
from typing import Any

import aiohttp
import reflex as rx

from bumblebee.chunks import build_chunks, chunks_to_jsonl

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_MARKDOWN_CHARS = 80_000  # keep the websocket payload sane for huge docs


def _local_parse(pdf: bytes, filename: str) -> dict[str, Any]:
    """Text-layer demo parse: perfect for born-digital PDFs, no GPU needed."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf)
    try:
        pages_json: list[list[dict[str, Any]]] = []
        md_pages: list[str] = []
        for page in doc:
            text = page.get_textpage().get_text_bounded() or ""
            lines = [ln.strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
            paragraphs = [ln for ln in lines if ln]
            pages_json.append(
                [
                    {"label": "text", "native_label": "paragraph", "content": p, "bbox_2d": [0, 0, 1000, 1000]}
                    for p in paragraphs
                ]
            )
            md_pages.append("\n\n".join(paragraphs))
        page_count = len(pages_json)
    finally:
        doc.close()

    region_count = sum(len(p) for p in pages_json)
    if region_count == 0:
        raise RuntimeError(
            "This PDF has no embedded text layer (it is likely scanned). "
            "Connect the hosted engine (BUMBLEBEE_API_URL + BUMBLEBEE_API_KEY) for full OCR."
        )
    stem = filename.rsplit(".", 1)[0] or "document"
    return {
        "markdown": "\n\n".join(md_pages),
        "chunks": build_chunks(pages_json, doc_path=filename, output_stem=stem),
        "stats": {"pages": page_count, "regions": region_count, "ocr_regions": 0},
    }


async def _hosted_parse(pdf: bytes, filename: str) -> dict[str, Any]:
    """Send raw PDF bytes to the deployed bumblebee API."""
    url = os.environ["BUMBLEBEE_API_URL"].rstrip("/") + "/v1/parse"
    headers = {"Authorization": f"Bearer {os.environ['BUMBLEBEE_API_KEY']}"}
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, params={"filename": filename}, data=pdf, headers=headers) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:300]
                raise RuntimeError(f"Parse API returned {resp.status}: {detail}")
            return await resp.json()


def hosted_configured() -> bool:
    return bool(os.environ.get("BUMBLEBEE_API_URL") and os.environ.get("BUMBLEBEE_API_KEY"))


class ParseState(rx.State):
    """One document's journey through the studio."""

    status: str = "idle"  # idle | parsing | done | error
    filename: str = ""
    error: str = ""
    engine: str = ""
    markdown: str = ""
    markdown_truncated: bool = False
    chunks: list[dict[str, str]] = []
    stats: list[dict[str, str]] = []
    elapsed: str = ""

    _pdf_path: str = ""  # backend-only: uploaded file on disk
    _chunks_jsonl: str = ""  # backend-only: full chunk records for download

    @rx.var
    def is_parsing(self) -> bool:
        return self.status == "parsing"

    @rx.var
    def chunk_count(self) -> int:
        return len(self.chunks)

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        file = files[0]
        name = file.name or "document.pdf"
        data = await file.read()
        if not name.lower().endswith(".pdf"):
            self.status, self.error = "error", "Only PDF files are supported. Drop a .pdf to parse it."
            return
        if len(data) > MAX_UPLOAD_BYTES:
            self.status, self.error = "error", "That PDF is over the 50 MB single-request limit."
            return
        if not data:
            self.status, self.error = "error", "The uploaded file is empty."
            return

        upload_dir = rx.get_upload_dir()
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"{time.time_ns()}_{name}"
        path.write_bytes(data)

        self._pdf_path = str(path)
        self.filename = name
        self.error = ""
        self.status = "parsing"
        return ParseState.run_parse

    @rx.event(background=True)
    async def run_parse(self):
        async with self:
            path, name = self._pdf_path, self.filename
        pdf = await asyncio.to_thread(lambda: open(path, "rb").read())

        started = time.perf_counter()
        try:
            if hosted_configured():
                engine = "hosted GPU pipeline"
                payload = await _hosted_parse(pdf, name)
            else:
                engine = "local text-layer demo"
                payload = await asyncio.to_thread(_local_parse, pdf, name)
        except Exception as exc:  # surfaced in the UI, never a blank screen
            async with self:
                self.status, self.error = "error", str(exc)
            return
        elapsed = time.perf_counter() - started

        markdown = payload.get("markdown", "")
        raw_chunks: list[dict[str, Any]] = payload.get("chunks") or []
        stats: dict[str, Any] = payload.get("stats") or {}

        async with self:
            self.engine = engine
            self.markdown_truncated = len(markdown) > MAX_MARKDOWN_CHARS
            self.markdown = markdown[:MAX_MARKDOWN_CHARS]
            self._chunks_jsonl = chunks_to_jsonl(raw_chunks)
            self.chunks = [
                {
                    "id": str(c.get("chunk_id", "")),
                    "kind": str(c.get("kind", "text")),
                    "section": " / ".join(c.get("section_path") or []) or "—",
                    "pages": ", ".join(str(p) for p in (c.get("pages") or [])),
                    "tokens": str(c.get("token_estimate", "")),
                    "text": str(c.get("text", ""))[:1200],
                }
                for c in raw_chunks
            ]
            api_seconds = stats.get("seconds")
            self.elapsed = f"{api_seconds if api_seconds is not None else round(elapsed, 2)}s"
            self.stats = [
                {"label": "pages", "value": str(stats.get("pages", "—"))},
                {"label": "regions", "value": str(stats.get("regions", "—"))},
                {"label": "sent to OCR", "value": str(stats.get("ocr_regions", "—"))},
                {"label": "chunks", "value": str(len(raw_chunks))},
                {"label": "parse time", "value": self.elapsed},
            ]
            self.status = "done"

    @rx.event
    def download_chunks(self):
        stem = self.filename.rsplit(".", 1)[0] or "document"
        return rx.download(data=self._chunks_jsonl, filename=f"{stem}.chunks.jsonl")

    @rx.event
    def download_markdown(self):
        stem = self.filename.rsplit(".", 1)[0] or "document"
        return rx.download(data=self.markdown, filename=f"{stem}.md")

    @rx.event
    def reset_studio(self):
        self.reset()
