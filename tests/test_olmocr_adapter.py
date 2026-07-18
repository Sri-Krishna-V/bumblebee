from pathlib import Path

from evals.olmocr_bench.adapter import candidate_path, materialize_candidate


def test_materialize_candidate_matches_official_layout(tmp_path: Path):
    bench_data = tmp_path / "bench_data"
    (bench_data / "pdfs" / "tables").mkdir(parents=True)
    (bench_data / "pdfs" / "tables" / "ledger.pdf").write_bytes(b"%PDF-fake")
    source = tmp_path / "batch-output"
    (source / "tables" / "ledger").mkdir(parents=True)
    (source / "tables" / "ledger" / "content.md").write_text("# Ledger", encoding="utf-8")
    candidate = bench_data / "bumblebee"

    coverage = materialize_candidate(bench_data=bench_data, source=source, candidate_root=candidate)

    assert coverage.expected_pages == 1
    assert coverage.copied_pages == 1
    assert coverage.missing_pages == []
    assert candidate_path(candidate, Path("tables/ledger.pdf"), 1).read_text(encoding="utf-8") == "# Ledger"


def test_materialize_candidate_reports_missing_pages_for_a_smoke_subset(tmp_path: Path):
    bench_data = tmp_path / "bench_data"
    pdfs = bench_data / "pdfs" / "tables"
    pdfs.mkdir(parents=True)
    (pdfs / "first.pdf").write_bytes(b"%PDF-first")
    (pdfs / "second.pdf").write_bytes(b"%PDF-second")
    source = tmp_path / "batch-output"
    (source / "tables" / "first").mkdir(parents=True)
    (source / "tables" / "first" / "content.md").write_text("first", encoding="utf-8")

    coverage = materialize_candidate(bench_data=bench_data, source=source, candidate_root=bench_data / "bumblebee")

    assert coverage.copied_pages == 1
    assert coverage.missing_pages == ["tables/second.pdf"]
