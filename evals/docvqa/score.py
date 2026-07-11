"""ANLS scoring for DocVQA answers (the official metric, implemented inline).

ANLS = mean over questions of max-over-references similarity, where similarity
is ``1 - normalized_levenshtein`` and scores below the 0.5 threshold count as 0.

Usage:
    uv run python -m evals.docvqa.score --answers evals/data/docvqa/answers.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

ANLS_THRESHOLD = 0.5


def levenshtein(a: str, b: str) -> int:
    """Edit distance via the classic two-row DP."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def anls_score(prediction: str, references: list[str], threshold: float = ANLS_THRESHOLD) -> float:
    """Score one prediction against its reference answers (case-insensitive)."""
    prediction = prediction.strip().lower()
    best = 0.0
    for reference in references:
        reference = reference.strip().lower()
        longest = max(len(prediction), len(reference))
        if longest == 0:
            similarity = 1.0 if prediction == reference else 0.0
        else:
            similarity = 1.0 - levenshtein(prediction, reference) / longest
        best = max(best, similarity)
    return best if best >= threshold else 0.0


def score_answers(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ANLS over answer records ({"prediction": str, "answers": [str, ...]})."""
    scores = [
        anls_score(str(item.get("prediction") or ""), [str(a) for a in item.get("answers", [])]) for item in answers
    ]
    count = len(scores)
    return {
        "questions": count,
        "anls": round(sum(scores) / count, 4) if count else None,
        "exact_or_close": sum(1 for s in scores if s > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=Path("evals/data/docvqa/answers.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("evals/data/docvqa/scores.json"))
    args = parser.parse_args()

    answers = [json.loads(line) for line in args.answers.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = score_answers(answers)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
