"""Merge ANLS scores and run throughput into one benchmark report (markdown).

Usage:
    uv run python -m evals.report \
        --scores evals/data/docvqa/scores.json \
        --run-summary evals/data/docvqa/parsed/_run_summary.json \
        --gpu-hourly-usd 2.00
"""

import argparse
import json
from pathlib import Path
from typing import Any


def build_report(scores: dict[str, Any], summary: dict[str, Any], gpu_hourly_usd: float, model: str) -> str:
    throughput = summary.get("throughput") or {}
    pages_per_second = throughput.get("pages_per_second")
    pages = (summary.get("pages") or {}).get("processed")
    cost_per_1k = gpu_hourly_usd / (pages_per_second * 3.6) if pages_per_second else None

    lines = [
        "# bumblebee DocVQA benchmark (parse-then-QA)",
        "",
        "The LLM answers questions from bumblebee's parsed markdown only — no image access —",
        "so the score measures how much answerable information the parse retains.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| ANLS | {scores.get('anls')} |",
        f"| Questions scored | {scores.get('questions')} |",
        f"| Pages parsed | {pages} |",
        f"| Pages/second | {round(pages_per_second, 2) if pages_per_second else 'n/a'} |",
        f"| GPU $/hour (input) | {gpu_hourly_usd:.2f} |",
        f"| Parse cost / 1k pages | {'$' + format(cost_per_1k, '.3f') if cost_per_1k else 'n/a'} |",
        f"| QA model | {model} |",
        "",
        "## Comparison caveats",
        "",
        "Landing AI reports 99.16% on DocVQA validation with the same parse-then-QA setup",
        "(their blog: 5,286/5,331, figure-dependent questions excluded, guided prompting,",
        "unspecified QA LLM). Numbers here use a different QA model and no question",
        "filtering, so treat any comparison as directional, not head-to-head.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("evals/data/docvqa/scores.json"))
    parser.add_argument("--run-summary", type=Path, default=Path("evals/data/docvqa/parsed/_run_summary.json"))
    parser.add_argument("--gpu-hourly-usd", type=float, default=2.00)
    parser.add_argument("--model", default="(see answers.jsonl)")
    parser.add_argument("--out", type=Path, default=Path("evals/results/docvqa_report.md"))
    args = parser.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    report = build_report(scores, summary, args.gpu_hourly_usd, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
