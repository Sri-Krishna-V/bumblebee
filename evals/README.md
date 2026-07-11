# bumblebee benchmarks

DocVQA parse-then-QA benchmark: parse the DocVQA validation pages with
bumblebee, then have an LLM answer the questions **from the parsed markdown
only** (no image access). This mirrors Landing AI's ADE methodology, so the
result is directly interpretable: how much answerable information does the
parse retain?

## Prerequisites

- `uv sync --frozen --group evals` (adds `datasets` + `httpx`)
- Modal account for the parse step (or a local GPU box)
- `OPENROUTER_API_KEY` for the QA step (a free model works; expect rate limits)

## Steps

```bash
# 1. Prepare (start with a 20-doc subset; drop --limit for the full ~1,300 docs)
uv run --group evals python -m evals.docvqa.prepare --limit 20

# 2. Parse on Modal (~$1-2 for the full split, startup dominates)
bumblebee modal --source evals/data/docvqa/pdfs --target evals/data/docvqa/parsed

# 3. Answer questions from parsed markdown (resumable; re-run to continue)
uv run --group evals python -m evals.docvqa.qa --model google/gemma-3-27b-it:free

# 4. Score (ANLS, the official DocVQA metric)
uv run python -m evals.docvqa.score

# 5. Report (accuracy + throughput + $/1k pages in one markdown table)
uv run python -m evals.report --gpu-hourly-usd 2.00
```

Free OpenRouter model ids churn — check <https://openrouter.ai/models?q=free>
and pass `--model`. The QA step checkpoints every answer to `answers.jsonl`,
so interruptions and rate limits only pause progress, never lose it.

## Budget notes ($30 Modal credits)

- Subset smoke run (20 docs): well under $1.
- Full validation split (~1,300 single-page PDFs): $1–2 including model-load startup.
- Re-run after accuracy changes: same again. Everything else (QA on a free model,
  scoring, reporting) is $0.

## Honest-comparison rules

Publish the QA model, prompt, and question filtering alongside any number.
Landing AI's 99.16% excluded figure-dependent questions and used guided
prompting with an unspecified LLM — treat comparisons as directional.
