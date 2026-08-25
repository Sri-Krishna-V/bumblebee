"""Prepare the DocVQA validation split for the parse-then-QA benchmark.

Downloads the split from HuggingFace, writes each page image as a single-page
PDF (one per unique document id), and a questions.jsonl mapping questions to
documents. Requires the ``evals`` dependency group::

    uv run --group evals python -m evals.docvqa.prepare --limit 20   # subset dry-run
    uv run --group evals python -m evals.docvqa.prepare              # full split (~1,300 docs)

Then parse the PDFs with bumblebee (Modal or local GPU)::

    bumblebee modal --source evals/data/docvqa/pdfs --target evals/data/docvqa/parsed
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evals/data/docvqa"))
    parser.add_argument("--limit", type=int, default=None, help="Maximum unique documents to prepare.")
    parser.add_argument("--dataset", default="lmms-lab/DocVQA")
    parser.add_argument("--config", dest="config_name", default="DocVQA")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    from datasets import load_dataset  # heavy import; evals group only

    dataset = load_dataset(args.dataset, args.config_name, split=args.split)

    pdf_dir = args.out / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    questions_path = args.out / "questions.jsonl"

    written_docs: set[str] = set()
    questions = 0
    with questions_path.open("w", encoding="utf-8") as questions_file:
        for row in dataset:
            doc_id = str(row.get("docId"))
            if doc_id not in written_docs and args.limit is not None and len(written_docs) >= args.limit:
                continue
            if doc_id not in written_docs:
                image = row["image"].convert("RGB")
                image.save(pdf_dir / f"{doc_id}.pdf", format="PDF")
                written_docs.add(doc_id)
            record = {
                "question_id": row.get("questionId"),
                "question": row.get("question"),
                "answers": list(row.get("answers") or []),
                "doc": doc_id,
            }
            questions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            questions += 1

    print(f"prepared documents={len(written_docs)} questions={questions} out={args.out}")


if __name__ == "__main__":
    main()
