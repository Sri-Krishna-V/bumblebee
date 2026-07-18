"""Materialize Bumblebee's batch output in the official olmOCR-bench layout.

The official runner expects one Markdown file per PDF at
``<candidate>/<category>/<stem>_pg1_repeat1.md``. Bumblebee's Modal batch
writer instead produces ``<output>/<category>/<stem>/content.md``. This small,
strict adapter copies only Markdown and verifies full page coverage before a
score can be presented as a complete benchmark result.

Example:

    python -m evals.olmocr_bench.adapter \
      --bench-data evals/data/olmocr-bench-20260718/bench_data \
      --source evals/data/olmocr-bench-20260718/bumblebee-raw \
      --candidate bumblebee
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class Coverage:
    """A completeness check for an official benchmark candidate."""

    expected_pages: int
    copied_pages: int
    missing_pages: list[str]
    unexpected_outputs: list[str]


def candidate_path(candidate_root: Path, relative_pdf: Path, repeat: int) -> Path:
    """Return the official runner path corresponding to one benchmark PDF."""
    stem = relative_pdf.with_suffix("")
    return candidate_root / stem.parent / f"{stem.name}_pg1_repeat{repeat}.md"


def materialize_candidate(
    *,
    bench_data: Path,
    source: Path,
    candidate_root: Path,
    repeat: int = 1,
    overwrite: bool = False,
) -> Coverage:
    """Copy completed Bumblebee Markdown and verify benchmark-page coverage."""
    if repeat < 1:
        raise ValueError("repeat must be at least one")
    pdf_root = bench_data / "pdfs"
    if not pdf_root.is_dir():
        raise FileNotFoundError(f"missing official benchmark PDFs at {pdf_root}")
    if not source.is_dir():
        raise FileNotFoundError(f"missing Bumblebee batch output at {source}")

    expected = {pdf.relative_to(pdf_root) for pdf in pdf_root.rglob("*.pdf")}
    copied: set[Path] = set()
    unexpected: list[str] = []
    for markdown in source.rglob("content.md"):
        relative_stem = markdown.parent.relative_to(source)
        relative_pdf = relative_stem.with_suffix(".pdf")
        if relative_pdf not in expected:
            unexpected.append(relative_stem.as_posix())
            continue
        destination = candidate_path(candidate_root, relative_pdf, repeat)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {destination}; pass --overwrite for a deliberate rerun")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(markdown, destination)
        copied.add(relative_pdf)

    missing = sorted((path.as_posix() for path in expected - copied))
    return Coverage(
        expected_pages=len(expected),
        copied_pages=len(copied),
        missing_pages=missing,
        unexpected_outputs=sorted(unexpected),
    )


def repository_revision() -> str | None:
    """Return the checked-out Bumblebee revision when Git is available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    """Expose strict candidate materialization as a command-line operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-data", type=Path, required=True, help="Directory containing official JSONL files and pdfs/.")
    parser.add_argument("--source", type=Path, required=True, help="Bumblebee Modal batch-output directory.")
    parser.add_argument("--candidate", default="bumblebee", help="Candidate folder name under --bench-data.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing candidate Markdown deliberately.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a manifest without failing when the Modal run is intentionally a smoke subset.",
    )
    args = parser.parse_args()

    candidate_root = args.bench_data / args.candidate
    coverage = materialize_candidate(
        bench_data=args.bench_data,
        source=args.source,
        candidate_root=candidate_root,
        repeat=args.repeat,
        overwrite=args.overwrite,
    )
    manifest = {
        "candidate": args.candidate,
        "created_at": datetime.now(UTC).isoformat(),
        "bumblebee_repository_revision": repository_revision(),
        "source": str(args.source),
        "coverage": asdict(coverage),
    }
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "bumblebee-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    if coverage.missing_pages and not args.allow_incomplete:
        raise SystemExit(f"refusing to score incomplete candidate: {len(coverage.missing_pages)} benchmark PDFs are missing")


if __name__ == "__main__":
    main()
