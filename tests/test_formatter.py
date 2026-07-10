"""Golden tests for the GLM-OCR-compatible formatter.

The golden file was generated from the pre-refactor formatter; the formatter's
numerics and output bytes must never change. Regenerate only for a deliberate,
reviewed output-format change.
"""

import json
from pathlib import Path

from bumblebee.format import format_document
from tests.conftest import make_recognized_region as region

GOLDEN = json.loads((Path(__file__).parent / "goldens" / "formatter_golden.json").read_text())

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

REGIONS = [
    # page 0: doc title, hyphen text-block merge, formula + trailing formula number
    region(0, 0, "doc_title", "text", [100, 10, 900, 60], "## The Great Doc", usage=USAGE),
    region(
        0, 1, "text", "text", [100, 100, 900, 200], "This paragraph ends with a hyphen and the word docu-", usage=USAGE
    ),
    region(0, 2, "text", "text", [100, 210, 900, 300], "ment continues in the next block.", usage=USAGE),
    region(0, 3, "display_formula", "formula", [100, 320, 800, 400], "$$E = mc^2$$", usage=USAGE),
    region(0, 4, "formula_number", "text", [820, 330, 900, 390], "(1)", usage=USAGE),
    # page 1: bullet alignment, paragraph title, skip region, whitespace-only content dropped
    region(1, 0, "paragraph_title", "text", [100, 10, 900, 50], "- Methods", usage=USAGE),
    region(1, 1, "text", "text", [100, 60, 900, 100], "- first bullet", usage=USAGE),
    region(1, 2, "text", "text", [102, 110, 900, 150], "middle bullet without dash", usage=USAGE),
    region(1, 3, "text", "text", [101, 160, 900, 200], "- third bullet", usage=USAGE),
    region(1, 4, "image", "skip", [100, 300, 500, 600], None, status=None, latency=None, usage=USAGE),
    region(1, 5, "text", "text", [100, 700, 900, 750], "   ", usage=USAGE),
    # page 2: table passthrough, numbered list normalization, single-newline doubling
    region(2, 0, "table", "table", [50, 50, 950, 500], "<table><tr><td>x</td></tr></table>", usage=USAGE),
    region(2, 1, "text", "text", [100, 520, 900, 560], "（2）second item", usage=USAGE),
    region(2, 2, "text", "text", [100, 570, 900, 640], "line one\nline two", usage=USAGE),
    region(2, 3, "text", "text", [100, 650, 900, 700], "• bullet via dot", usage=USAGE),
    region(2, 4, "text", "text", [100, 710, 900, 760], "1）numbered", usage=USAGE),
    # page 3: empty page (no regions)
    # page 4: repeated-content truncation + failed OCR region (None content dropped)
    region(4, 0, "text", "text", [100, 100, 900, 400], ("spam and eggs! " * 40).strip(), usage=USAGE),
    region(4, 1, "text", "text", [100, 500, 900, 560], None, status=500, usage=None),
    region(4, 2, "inline_formula", "formula", [100, 600, 900, 650], "\\[x + y\\]", usage=USAGE),
]


def test_formatter_golden_output():
    json_result, markdown = format_document(5, REGIONS)
    assert markdown == GOLDEN["markdown"]
    assert json_result == GOLDEN["json"]


def test_empty_document():
    json_result, markdown = format_document(0, [])
    assert json_result == []
    assert markdown == ""


def test_page_markers_for_empty_pages():
    json_result, markdown = format_document(2, [])
    assert json_result == [[], []]
    assert "<!-- page:start 1 -->" in markdown
    assert "<!-- page:end 2 -->" in markdown
