"""Bumblebee Studio — the product face of the bumblebee parse API.

Single page: hero (animated hive) -> studio (upload -> parse -> results)
-> pipeline -> API. All styling lives in assets/style.css; components carry
class names, not style soup.
"""

import reflex as rx

from bumblebee_studio.state import ParseState, hosted_configured

GITHUB_URL = "https://github.com/Sri-Krishna-V/bumblebee"

FONTS = (
    "https://fonts.googleapis.com/css2"
    "?family=Bricolage+Grotesque:opsz,wght@12..96,300..800"
    "&family=Inter:wght@400;500;600"
    "&family=JetBrains+Mono:wght@400;500"
    "&display=swap"
)


# ── Shared bits ──────────────────────────────────────────────────


def nav() -> rx.Component:
    return rx.el.nav(
        rx.link(rx.box(class_name="mark"), rx.text("bumblebee"), href="#", class_name="wordmark"),
        rx.box(
            rx.link("Studio", href="#studio", class_name="nav-link"),
            rx.link("Pipeline", href="#pipeline", class_name="nav-link"),
            rx.link("API", href="#api", class_name="nav-link"),
            rx.link("GitHub", href=GITHUB_URL, is_external=True, class_name="btn btn-ghost btn-small"),
            class_name="nav-links",
        ),
        class_name="nav",
    )


def hive_field() -> rx.Component:
    """The signature: a honeycomb that wakes in a radial wave from center."""
    cells = []
    hex_w, hex_h, gap = 96, 108, 8
    for row in range(7):
        for col in range(14):
            x = col * (hex_w + gap) + (0 if row % 2 == 0 else (hex_w + gap) / 2) - 60
            y = row * (hex_h * 0.75 + gap) - 40
            dist = ((x - 620) ** 2 + (y - 320) ** 2) ** 0.5
            filled = (row * 13 + col * 7) % 19 == 0
            cells.append(
                rx.box(
                    class_name="hex hex-fill" if filled else "hex",
                    style={
                        "left": f"{x:.0f}px",
                        "top": f"{y:.0f}px",
                        "animation_delay": f"{dist / 210:.2f}s",
                    },
                )
            )
    return rx.box(*cells, class_name="hive", aria_hidden="true")


# ── Hero ─────────────────────────────────────────────────────────


def hero() -> rx.Component:
    return rx.el.header(
        hive_field(),
        rx.box(
            rx.text("Bumblebee · Document Intelligence", class_name="eyebrow rise rise-1"),
            rx.el.h1(
                "Every PDF, ",
                rx.el.span("retrieval-ready.", class_name="honey"),
                class_name="hero-title rise rise-2",
            ),
            rx.text(
                "Bumblebee reads the layout, runs OCR only where it must, and returns "
                "markdown, layout JSON, and RAG-ready chunks — one call, one GPU.",
                class_name="hero-sub rise rise-3",
            ),
            rx.box(
                rx.link("Parse a document", href="#studio", class_name="btn btn-primary"),
                rx.link("Read the API", href="#api", class_name="btn btn-ghost"),
                style={"display": "flex", "gap": "14px", "flex_wrap": "wrap", "justify_content": "center"},
                class_name="rise rise-4",
            ),
            rx.text(
                "Apache-2.0 · Open source · PP-DocLayoutV3 + GLM-OCR",
                class_name="trust-line rise rise-5",
            ),
            class_name="hero-inner",
        ),
        class_name="hero",
    )


# ── Studio ───────────────────────────────────────────────────────


def engine_pill() -> rx.Component:
    if hosted_configured():
        return rx.box(
            rx.box(class_name="dot"),
            rx.text("hosted GPU engine connected"),
            class_name="engine-pill",
        )
    return rx.box(
        rx.box(class_name="dot dot-demo"),
        rx.text("demo mode — text-layer engine · set BUMBLEBEE_API_URL for full OCR"),
        class_name="engine-pill",
    )


def dropzone() -> rx.Component:
    return rx.upload.root(
        rx.box(
            rx.icon("file-text", size=40, color="#FFB224"),
            rx.text("Drop a PDF here", class_name="dropzone-title"),
            rx.text("or click to choose · up to 50 MB", class_name="dropzone-hint"),
            class_name="dropzone",
        ),
        id="doc",
        accept={"application/pdf": [".pdf"]},
        max_files=1,
        on_drop=ParseState.handle_upload(rx.upload_files(upload_id="doc")),
        width="100%",
    )


def parsing_view() -> rx.Component:
    return rx.box(
        rx.box(
            rx.box(*[rx.box(class_name="cell") for _ in range(7)], class_name="hexloader"),
            rx.text("Parsing " + " ", ParseState.filename, class_name="parsing-label"),
            rx.text(
                "layout → OCR → chunks",
                style={"color": "var(--taupe)", "font_family": "var(--font-mono)", "font_size": "0.78rem"},
            ),
            class_name="parsing-stage",
        ),
        class_name="scanning",
    )


def error_view() -> rx.Component:
    return rx.box(
        rx.text("Parse failed", class_name="error-title"),
        rx.text(ParseState.error, style={"color": "var(--taupe)", "line_height": "1.6"}),
        rx.button("Try another PDF", on_click=ParseState.reset_studio, class_name="btn btn-primary btn-small"),
        class_name="error-card",
    )


def stat_tile(stat: dict[str, str]) -> rx.Component:
    return rx.box(
        rx.text(stat["value"], class_name="stat-value"),
        rx.text(stat["label"], class_name="stat-label"),
        class_name="stat-tile",
    )


def chunk_card(chunk: dict[str, str]) -> rx.Component:
    return rx.box(
        rx.box(
            rx.text(
                chunk["kind"],
                class_name=rx.cond(chunk["kind"] == "text", "kind-badge", "kind-badge kind-badge-table"),
            ),
            rx.text("p. ", chunk["pages"], class_name="chunk-section"),
            rx.text("~", chunk["tokens"], " tok", class_name="chunk-section"),
            class_name="chunk-meta",
        ),
        rx.text(chunk["section"], class_name="chunk-section"),
        rx.text(chunk["text"], class_name="chunk-text"),
        class_name="chunk-card",
    )


def results_view() -> rx.Component:
    return rx.box(
        rx.box(
            rx.box(
                rx.text(ParseState.filename, style={"font_family": "var(--font-display)", "font_weight": "600", "font_size": "1.15rem"}),
                rx.text(ParseState.engine, class_name="chunk-section"),
                style={"display": "flex", "flex_direction": "column", "gap": "4px"},
            ),
            rx.box(
                rx.button("Download chunks.jsonl", on_click=ParseState.download_chunks, class_name="btn btn-ghost btn-small"),
                rx.button("Download markdown", on_click=ParseState.download_markdown, class_name="btn btn-ghost btn-small"),
                rx.button("Parse another", on_click=ParseState.reset_studio, class_name="btn btn-primary btn-small"),
                style={"display": "flex", "gap": "10px", "flex_wrap": "wrap"},
            ),
            style={"display": "flex", "justify_content": "space-between", "align_items": "center", "flex_wrap": "wrap", "gap": "16px"},
        ),
        rx.box(rx.foreach(ParseState.stats, stat_tile), class_name="stats-row"),
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Chunks", value="chunks"),
                rx.tabs.trigger("Markdown", value="markdown"),
            ),
            rx.tabs.content(
                rx.box(rx.foreach(ParseState.chunks, chunk_card), class_name="chunk-grid", margin_top="18px"),
                value="chunks",
            ),
            rx.tabs.content(
                rx.box(
                    rx.cond(
                        ParseState.markdown_truncated,
                        rx.text("Preview truncated — download the full markdown above.", class_name="chunk-section"),
                    ),
                    rx.markdown(ParseState.markdown),
                    class_name="md-body",
                    margin_top="18px",
                ),
                value="markdown",
            ),
            default_value="chunks",
            color_scheme="amber",
        ),
        style={"display": "flex", "flex_direction": "column", "gap": "8px"},
    )


def studio() -> rx.Component:
    return rx.el.section(
        rx.box(
            rx.text("Studio", class_name="eyebrow"),
            rx.el.h2("Drop a PDF. Watch the hive work.", class_name="section-title"),
            rx.box(engine_pill()),
            class_name="section-head",
        ),
        rx.box(
            rx.match(
                ParseState.status,
                ("parsing", parsing_view()),
                ("error", error_view()),
                ("done", results_view()),
                dropzone(),
            ),
            class_name="studio-card",
        ),
        class_name="section",
        id="studio",
    )


# ── Pipeline ─────────────────────────────────────────────────────


def pipe_card(step: str, title: str, body: str) -> rx.Component:
    return rx.box(
        rx.text(step, class_name="pipe-step"),
        rx.el.h3(title, class_name="pipe-title"),
        rx.text(body, class_name="pipe-body"),
        class_name="pipe-card",
    )


def pipeline() -> rx.Component:
    return rx.el.section(
        rx.box(
            rx.text("Pipeline", class_name="eyebrow"),
            rx.el.h2("OCR only where it earns its keep.", class_name="section-title"),
            rx.text(
                "Most parsers OCR every pixel. Bumblebee detects layout first, reads the "
                "embedded text layer for free where it exists, and spends GPU time only on "
                "regions that need it.",
                class_name="section-sub",
            ),
            class_name="section-head",
        ),
        rx.box(
            pipe_card(
                "01 / LAYOUT",
                "See the page first",
                "PP-DocLayoutV3 maps every region — titles, paragraphs, tables, formulas, "
                "figures — before a single character is read.",
            ),
            pipe_card(
                "02 / HYBRID OCR",
                "Read the cheap way when possible",
                "Born-digital text comes straight from the PDF's text layer, perfectly and "
                "for free. GLM-OCR on vLLM handles only the regions that truly need vision.",
            ),
            pipe_card(
                "03 / CHUNKS",
                "Leave retrieval-ready",
                "Heading-aware chunks packed to a token budget, each with section path, page "
                "numbers, and bounding boxes. Straight into your vector store.",
            ),
            class_name="pipeline-grid",
        ),
        class_name="section",
        id="pipeline",
    )


# ── API ──────────────────────────────────────────────────────────

CURL = """curl -X POST "$BUMBLEBEE_API_URL/v1/parse?filename=paper.pdf" \\
  -H "Authorization: Bearer $BUMBLEBEE_API_KEY" \\
  --data-binary @paper.pdf

# -> { "markdown": ..., "layout": ..., "chunks": [...], "stats": {...} }"""


def api_section() -> rx.Component:
    return rx.el.section(
        rx.box(
            rx.text("API", class_name="eyebrow"),
            rx.el.h2("One call. Everything out.", class_name="section-title"),
            rx.text(
                "Raw PDF bytes in the body, bearer token in the header. No multipart, no SDK "
                "required, no second request for chunks.",
                class_name="section-sub",
            ),
            class_name="section-head",
        ),
        rx.el.pre(CURL, class_name="code-block"),
        class_name="section",
        id="api",
    )


def footer() -> rx.Component:
    return rx.el.footer(
        rx.box(
            rx.box(class_name="mark", style={"width": "16px", "height": "18px"}),
            rx.text("bumblebee — cheap, fast, single-GPU document parsing"),
            style={"display": "flex", "align_items": "center", "gap": "10px"},
        ),
        rx.link("Source on GitHub", href=GITHUB_URL, is_external=True, class_name="nav-link"),
        class_name="footer",
    )


# ── Page + app ───────────────────────────────────────────────────


def index() -> rx.Component:
    return rx.box(nav(), hero(), studio(), pipeline(), api_section(), footer())


app = rx.App(
    stylesheets=[FONTS, "style.css"],
    style={"background": "#0B0908"},
)
app.add_page(index, route="/", title="Bumblebee — every PDF, retrieval-ready")
