"""Born-digital text-layer extraction (the OCR hybrid's fast path).

Many PDFs carry an embedded text layer. For those pages, reading the text
directly is both perfectly accurate and free — no GPU request. Only ``TEXT``
regions are eligible: tables and formulas always go to the VLM because their
*structure* is the output, not just the characters.

PDFium is not thread-safe, so :func:`extract_region_texts` must run on the
same serialized worker thread as rendering (``RenderEngine`` wraps it). The
per-page policy in :func:`satisfy_regions_from_text_layer` guards against
broken text layers (bad encodings, CID fonts): a page's embedded text is used
only when at least ``coverage_threshold`` of its eligible regions yield text —
otherwise the whole page falls back to OCR.
"""

import logging
from collections.abc import Mapping, Sequence

from bumblebee.models import RecognizedRegion, Region, Task
from bumblebee.pdf import pdfium_module

logger = logging.getLogger(__name__)

COVERAGE_THRESHOLD = 0.8


def _region_pdf_box(region: Region, width_pt: float, height_pt: float) -> tuple[float, float, float, float]:
    """Map a normalized top-down bbox (0..1000) to PDF points (bottom-up): (l, b, r, t)."""
    x1n, y1n, x2n, y2n = region.bbox_2d
    left = x1n * width_pt / 1000
    right = x2n * width_pt / 1000
    top = height_pt - (y1n * height_pt / 1000)
    bottom = height_pt - (y2n * height_pt / 1000)
    return left, bottom, right, top


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_region_texts(
    pdf_bytes: bytes,
    regions_by_page: Mapping[int, Sequence[Region]],
) -> dict[int, list[str]]:
    """Extract embedded text for regions, aligned with each page's input order.

    Returns ``""`` for regions without embedded text. Must run on the
    serialized PDFium worker thread (see module docstring).
    """
    pdfium = pdfium_module()
    pdf = pdfium.PdfDocument(pdf_bytes)
    out: dict[int, list[str]] = {}
    try:
        for page_index, regions in regions_by_page.items():
            page = pdf[page_index]
            try:
                width_pt, height_pt = (float(v) for v in page.get_size())
                textpage = page.get_textpage()
                try:
                    texts: list[str] = []
                    for region in regions:
                        left, bottom, right, top = _region_pdf_box(region, width_pt, height_pt)
                        bounded = textpage.get_text_bounded(left=left, bottom=bottom, right=right, top=top)
                        texts.append(_normalize(bounded))
                    out[page_index] = texts
                finally:
                    textpage.close()
            finally:
                page.close()
    finally:
        pdf.close()
    return out


def satisfy_regions_from_text_layer(
    region_texts: Mapping[int, Sequence[str]],
    chunk_regions: dict[int, list[Region]],
    *,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> tuple[list[RecognizedRegion], dict[int, list[Region]]]:
    """Split regions into text-layer-satisfied results and regions still needing OCR.

    ``region_texts`` maps page index to texts aligned with that page's eligible
    (``TEXT``-task) regions in order, as returned by :func:`extract_region_texts`.
    """
    satisfied: list[RecognizedRegion] = []
    remaining: dict[int, list[Region]] = {}
    for page_index, regions in chunk_regions.items():
        eligible = [region for region in regions if region.task_type == Task.TEXT]
        texts = list(region_texts.get(page_index, []))
        nonempty = sum(1 for text in texts if text)
        usable = bool(eligible) and len(texts) == len(eligible) and nonempty / len(eligible) >= coverage_threshold
        if not usable:
            remaining[page_index] = list(regions)
            continue
        text_for = {id(region): text for region, text in zip(eligible, texts, strict=True)}
        keep: list[Region] = []
        for region in regions:
            text = text_for.get(id(region), "")
            if region.task_type == Task.TEXT and text:
                satisfied.append(RecognizedRegion(region=region, content=text))
            else:
                keep.append(region)
        remaining[page_index] = keep
    return satisfied, remaining
