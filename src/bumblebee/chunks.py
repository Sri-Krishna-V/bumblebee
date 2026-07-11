"""RAG chunk building from formatted layout JSON (the bumblebee product surface).

Chunks are derived purely from the per-page layout JSON that the formatter
already produces (``label``, ``native_label``, ``content``, ``bbox_2d``), so
they can be built anywhere that JSON exists — worker-side writes, local payload
writes, or the hosted API — without changing any transport shape.

Chunking policy: consecutive text regions are packed up to a token budget and
never across heading boundaries; tables and formulas are atomic chunks; image
regions (null content) are skipped. Heading text starts the next chunk (so
chunks read self-contained) and also feeds ``section_path`` metadata.
"""

import json
import re
from typing import Any

# ponytail: token counts are estimated as len(text)//4 (no tokenizer dependency);
# swap in a real tokenizer if chunk sizing ever needs to be exact.
_CHARS_PER_TOKEN = 4

DEFAULT_CHUNK_MAX_TOKENS = 512


def _token_estimate(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _heading_title(content: str) -> str:
    """Strip the formatter's markdown heading prefix for section_path entries."""
    return re.sub(r"^#+\s*", "", content).strip()


class _ChunkBuilder:
    """Accumulate regions into chunk records for one document."""

    def __init__(self, *, doc_path: str, output_stem: str, max_tokens: int) -> None:
        self.doc_path = doc_path
        self.output_stem = output_stem
        self.max_tokens = max(1, int(max_tokens))
        self.chunks: list[dict[str, Any]] = []
        self.section: list[str] = []
        self._texts: list[str] = []
        self._boxes: list[dict[str, Any]] = []
        self._tokens = 0

    def flush(self) -> None:
        if not self._texts:
            return
        self._emit("text", "\n\n".join(self._texts), self._boxes)
        self._texts = []
        self._boxes = []
        self._tokens = 0

    def add_text(self, content: str, page: int, bbox: list[int]) -> None:
        estimate = _token_estimate(content)
        if self._texts and self._tokens + estimate > self.max_tokens:
            self.flush()
        self._texts.append(content)
        self._boxes.append({"page": page, "bbox": bbox})
        self._tokens += estimate

    def add_atomic(self, kind: str, content: str, page: int, bbox: list[int]) -> None:
        self.flush()
        self._emit(kind, content, [{"page": page, "bbox": bbox}])

    def set_section(self, level: int, title: str) -> None:
        """Update the heading stack; level 1 = doc_title, level 2 = paragraph_title."""
        self.flush()
        if level == 1:
            self.section = [title]
        else:
            self.section = [*self.section[:1], title]

    def _emit(self, kind: str, text: str, boxes: list[dict[str, Any]]) -> None:
        self.chunks.append(
            {
                "chunk_id": f"{self.output_stem}#{len(self.chunks):04d}",
                "doc": self.doc_path,
                "section_path": list(self.section),
                "pages": sorted({box["page"] for box in boxes}),
                "bboxes": boxes,
                "kind": kind,
                "text": text,
                "token_estimate": _token_estimate(text),
            }
        )


def build_chunks(
    pages: list[list[dict[str, Any]]],
    *,
    doc_path: str,
    output_stem: str,
    max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Build RAG chunk records from a document's formatted layout JSON.

    ``pages`` is the layout JSON shape written to ``layout.json`` (sanitized or
    not — only ``label``/``native_label``/``content``/``bbox_2d`` are read).
    Page numbers in the output are 1-based, matching the markdown page markers.
    """
    builder = _ChunkBuilder(doc_path=doc_path, output_stem=output_stem, max_tokens=max_tokens)
    for page_idx, regions in enumerate(pages):
        page_number = page_idx + 1
        for region in regions:
            content = region.get("content")
            if not isinstance(content, str) or not content.strip():
                continue  # image/skip regions carry null content
            label = str(region.get("label", "text"))
            native_label = str(region.get("native_label", ""))
            bbox = [int(v) for v in region.get("bbox_2d", [])]

            if native_label == "doc_title":
                builder.set_section(1, _heading_title(content))
                builder.add_text(content, page_number, bbox)
            elif native_label == "paragraph_title":
                builder.set_section(2, _heading_title(content))
                builder.add_text(content, page_number, bbox)
            elif label in ("table", "formula"):
                builder.add_atomic(label, content, page_number, bbox)
            else:
                builder.add_text(content, page_number, bbox)
    builder.flush()
    return builder.chunks


def chunks_to_jsonl(chunks: list[dict[str, Any]]) -> str:
    """Serialize chunk records as JSON Lines (one chunk per line)."""
    return "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
