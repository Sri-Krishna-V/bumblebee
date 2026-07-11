"""Answer DocVQA questions from parsed markdown only (parse-then-QA), via OpenRouter.

Mirrors the ADE benchmark methodology: the LLM sees ONLY bumblebee's parsed
``content.md`` for the document — never the page image — so the score measures
how much answerable information the parse retained.

Resumable at question level: answers are appended to answers.jsonl and already
answered questions are skipped, so rate-limited free models can be run in
several sittings. Requires ``OPENROUTER_API_KEY``::

    uv run --group evals python -m evals.docvqa.qa --model google/gemma-3-27b-it:free

Free model ids churn — check https://openrouter.ai/models?q=free for a current one.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-3-27b-it:free"
MAX_CONTENT_CHARS = 30_000  # keep prompts inside small-model context windows

PROMPT = """You are answering a question using ONLY the parsed content of a scanned document, \
provided below as markdown.

Rules:
- Answer with the shortest exact span from the document that answers the question (a value, name, date, or phrase).
- Copy the answer text exactly as it appears; do not rephrase, explain, or add punctuation.
- If the document does not contain the answer, reply exactly: unanswerable

Document content:
---
{content}
---

Question: {question}
Answer:"""


def ask(client: Any, model: str, content: str, question: str) -> str:
    prompt = PROMPT.format(content=content[:MAX_CONTENT_CHARS], question=question)
    response = client.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 64,
        },
        timeout=120,
    )
    if response.status_code == 429:
        raise RateLimited(response.headers.get("Retry-After"))
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"] or "").strip()


class RateLimited(Exception):
    def __init__(self, retry_after: str | None):
        super().__init__("rate limited")
        try:
            self.retry_after = float(retry_after) if retry_after else 30.0
        except ValueError:
            self.retry_after = 30.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("evals/data/docvqa"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between requests (free tiers are limited).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum questions to answer this run.")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set")

    import httpx

    questions = load_jsonl(args.data / "questions.jsonl")
    answers_path = args.data / "answers.jsonl"
    answered = {item["question_id"] for item in load_jsonl(answers_path)}
    pending = [q for q in questions if q["question_id"] not in answered]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"questions={len(questions)} answered={len(answered)} pending={len(pending)} model={args.model}")

    content_cache: dict[str, str | None] = {}
    done = 0
    with httpx.Client() as client, answers_path.open("a", encoding="utf-8") as out:
        for item in pending:
            doc = str(item["doc"])
            if doc not in content_cache:
                markdown_path = args.data / "parsed" / doc / "content.md"
                content_cache[doc] = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else None
            content = content_cache[doc]
            if content is None:
                print(f"skip question_id={item['question_id']} (no parsed content.md for doc {doc})")
                continue
            while True:
                try:
                    prediction = ask(client, args.model, content, str(item["question"]))
                    break
                except RateLimited as limited:
                    print(f"rate limited; sleeping {limited.retry_after:.0f}s")
                    time.sleep(limited.retry_after)
            record = {
                "question_id": item["question_id"],
                "doc": doc,
                "prediction": prediction,
                "answers": item["answers"],
                "model": args.model,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            done += 1
            if done % 25 == 0:
                print(f"answered {done}/{len(pending)}")
            time.sleep(args.sleep)

    print(f"done: answered {done} questions this run -> {answers_path}")


if __name__ == "__main__":
    main()
