# Bumblebee

Internal document-ingestion tool. PDFs in, layout-aware Markdown and RAG-ready
chunks out, on a single GPU.

One process co-locates two models on one device: PP-DocLayoutV3 region detection
(`PaddlePaddle/PP-DocLayoutV3_safetensors`) and the GLM-OCR recognizer
(`zai-org/GLM-OCR`), served by an in-process vLLM OpenAI-compatible server. An
async pipeline streams each document through render → layout → crop → OCR →
format in page chunks, so OCR requests for a document's first pages are already
in flight while its later pages are still queued for render and layout. vLLM's
continuous batcher does the rest.

There are three entry points onto the same engine:

| Entry point | Use for |
| --- | --- |
| `bumblebee` (CLI) | Batch runs on a GPU box you already have. Folder or bucket in, folder or bucket out, resumable. |
| `bumblebee modal` | The same batch workflow with the GPU rented from Modal per run. No local GPU needed. |
| `bumblebee deploy-api` | A persistent Modal web endpoint that parses one PDF per request and never persists it. `web/` is its front end. |

## How a document is processed

1. **Read** — PDF bytes come from local disk, `s3://`, `gs://`, or `az://`
   (fsspec backends), or in memory for the hosted API.
2. **Render** — PDFium renders pages at `pdf_dpi` (default 100), capping the
   longest side at `pdf_max_side` (default 3500 px). PDFium is not thread-safe,
   so rendering is serialized on one worker thread.
3. **Layout** — PP-DocLayoutV3 detects regions over a 25-class label space and
   predicts reading order. Each label maps to a task: `text`, `table`,
   `formula`, or `skip` (skipped regions never reach OCR).
4. **Text-layer hybrid** — with `text_layer=auto` (the default), `TEXT` regions
   on born-digital pages are served straight from the PDF's embedded text layer:
   exact, and free of a GPU request. A page's embedded text is only trusted when
   at least 80% of its eligible regions yield text (this guards against broken
   encodings and CID fonts); otherwise the whole page falls back to OCR. Tables
   and formulas always go to the VLM, because their structure is the output.
   `text_layer=off` always OCRs.
5. **Crop and OCR** — remaining regions are cropped, resized to GLM-OCR's vision
   geometry, JPEG-encoded at `jpeg_quality`, and sent concurrently to the local
   vLLM server. Token budgets differ per task (`max_tokens_text`,
   `max_tokens_formula`, `max_tokens_table`).
6. **Confidence and adaptive retry** — when `ocr_logprobs` is on (default), each
   region gets a confidence score of `exp(mean token logprob)`. With
   `adaptive_retry` on (default), a document's lowest-confidence regions below
   `confidence_threshold` (default 0.80) are re-rendered and re-OCR'd once at 2x
   DPI, capped at 10% of that document's OCR regions.
7. **Format** — regions are assembled into Markdown with `<!-- page:start N -->`
   / `<!-- page:end N -->` markers, plus a per-page layout JSON.
8. **Chunk** (optional) — `chunks.jsonl` is built from the layout JSON. See
   [Chunk output](#chunk-output).

## Repository layout

| Path | What lives there |
| --- | --- |
| `src/bumblebee/cli.py` | Typer CLI: default local-GPU command, `modal`, `deploy-api`. |
| `src/bumblebee/engine.py` | `DocumentEngine` — loads layout, boots vLLM, owns `ocr`/`stream`/`run`. |
| `src/bumblebee/pipeline.py` | Provider-neutral async pipeline. Never touches storage. |
| `src/bumblebee/config.py` | `OcrConfig`, `EngineConfig`, `ModalConfig` and the `BUMBLEBEE_*` fallback machinery. |
| `src/bumblebee/chunks.py` | RAG chunk building and JSONL serialization. |
| `src/bumblebee/textlayer.py` | Born-digital text-layer extraction and the per-page trust policy. |
| `src/bumblebee/runs.py` | Output paths, resumability, persistence. |
| `src/bumblebee/batches.py` | `BatchPolicy` and the batch supervisor. |
| `src/bumblebee/storage.py` | Local and fsspec (S3/GCS/Azure) storage backends. |
| `src/bumblebee/api.py` | The hosted FastAPI parse API (engine-agnostic). |
| `src/bumblebee/pilot.py` | API key registry and the SQLite usage/audit ledger. |
| `src/bumblebee/modal/` | Modal app, image, workers, batch entrypoint, API deployment. |
| `evals/` | DocVQA parse-then-QA harness and the olmOCR-bench candidate adapter. |
| `web/` | Next.js studio (`bumblebee-evidence-engine`) in front of the hosted API. |
| `tests/` | GPU-free unit tests. |

## Install

Requires Python 3.12–3.14 (`.python-version` pins 3.13) and [uv](https://docs.astral.sh/uv/).

```bash
# CPU checkout: package + dev tooling from the lockfile. Enough for tests,
# lint, types, the CLI's --help, and reading config.
make install          # == uv sync --frozen --group dev
```

That does not install the GPU stack. On a GPU box:

```bash
uv sync --frozen --extra gpu --extra trt               # inference + fast layout
uv sync --frozen --extra gpu --extra trt --extra s3    # ... plus s3:// URIs
```

Extras:

| Extra | Contents |
| --- | --- |
| `gpu` | vLLM (Linux only), transformers, accelerate, sentencepiece, huggingface-hub. Required for any real OCR run. |
| `trt` | onnx, onnxruntime-gpu, tensorrt (Linux only). Powers the default `onnx` layout backend; the PyTorch detector is the fallback when unavailable. |
| `modal` | The `modal` CLI/SDK, needed by `bumblebee modal` and `bumblebee deploy-api`. |
| `s3` / `gcs` / `azure` | fsspec plus the matching filesystem for `s3://`, `gs://`, `az://` URIs. |

Dependency groups:

| Group | Contents |
| --- | --- |
| `dev` | ruff, pyright, pytest, pytest-asyncio, fastapi, httpx. |
| `evals` | `datasets` + `httpx` for the benchmark harness. Declared as conflicting with the cloud-storage extras in `pyproject.toml`; install it in its own environment. |

TensorRT/CUDA wheels come from NVIDIA's package index, already configured under
`[[tool.uv.index]]`.

### Docker (bare-metal GPU boxes)

```bash
make build-image      # == docker build -t bumblebee .
docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface \
  bumblebee --source s3://bucket/in --target s3://bucket/out
```

The image installs the `gpu`, `trt`, `azure`, `s3`, and `gcs` extras from
`uv.lock`. Mount the Hugging Face cache — the first run downloads both models
and builds the TensorRT layout engine, which is slow.

## CLI

The single console script is `bumblebee`, wired to `bumblebee.cli:rag_main`.
That entry point sets `BUMBLEBEE_EMIT_CHUNKS=1` before dispatching, so **running
`bumblebee` emits `chunks.jsonl` by default**. An explicit `--no-chunks`, or a
pre-set `BUMBLEBEE_EMIT_CHUNKS`, still wins. The plain entry point
(`bumblebee.cli:main`, reachable as `python -m bumblebee.cli`) is identical
except that chunk emission defaults to off.

Every settings flag defaults to unset. An unset flag falls back to its
`BUMBLEBEE_*` environment variable, then to the built-in default. Explicit flags
always win.

```bash
# Local GPU, local folders. Startup takes minutes (model load, vLLM boot,
# CUDA graph capture); use --limit for a smoke run.
bumblebee --source ./input --target ./output --limit 1

# Cloud in, cloud out. Install the matching storage extra.
bumblebee --source s3://bucket/in --target s3://bucket/out

# Reprocess everything, including already-complete outputs.
bumblebee --source ./input --target ./output --force

# Markdown only, no chunks.
bumblebee --source ./input --target ./output --no-chunks
```

Runs are resumable by default: a document whose `stats.json` reads
`status: "succeeded"` (with `content.md` and `layout.json` present) is skipped.
The command prints the run summary as JSON on stdout.

### Run flags (`OcrConfig`)

Accepted by both the default command and `bumblebee modal`.

| Flag | Default | Notes |
| --- | --- | --- |
| `--source` | `./input` | Local folder/file or cloud URI containing PDFs. |
| `--target` | `./output` | Local folder or cloud URI for outputs. |
| `--limit` | unlimited | Maximum number of *incomplete* PDFs to process. |
| `--force` / `--no-force` | off | Reprocess documents whose outputs are already complete. |
| `--pdf-dpi` | 100 | Render DPI. Raise it when documents need more OCR accuracy. |
| `--pdf-max-side` | 3500 | Longest rendered page side, in pixels. |
| `--page-chunk-size` | 8 | Pages streamed through render→layout→crop→OCR as one unit. Keep it a multiple of `--layout-batch-size`. |
| `--jpeg-quality` | 90 | JPEG quality for OCR crops. |
| `--text-layer` | `auto` | `auto` serves text regions from born-digital PDFs; `off` always OCRs. |
| `--max-tokens-text` | 2048 | Max generated tokens for text regions. |
| `--max-tokens-formula` | 2048 | Max generated tokens for formulae. |
| `--max-tokens-table` | 4096 | Max generated tokens for tables. |
| `--temperature` | 0.0 | Generation temperature. |
| `--top-p` | 0.00001 | Generation top_p. Must be in (0, 1]. |
| `--chunks` / `--no-chunks` | on under `bumblebee` | Write `chunks.jsonl` beside each document. |
| `--chunk-max-tokens` | 512 | Approximate token budget per packed text chunk. |
| `--max-inflight-pdfs` | 16 (32 under `modal`) | Documents rendered/cropped concurrently. |
| `--ocr-request-concurrency` | 1024 | Concurrent OCR requests to vLLM. Keep near `--max-num-seqs`. |
| `--storage-check-workers` | 64 | Threads reading completion markers when resuming against cloud targets. |
| `--log-level` | `INFO` | DEBUG/INFO/WARNING/... |

### Engine flags (`EngineConfig`, startup-scoped)

Changing any of these requires an engine restart.

| Flag | Default | Notes |
| --- | --- | --- |
| `--vllm-port` | 8000 | Local vLLM server port. |
| `--vllm-health-timeout` | 900 | Seconds to wait for vLLM to become healthy. |
| `--gpu-memory-utilization` | 0.60 | vLLM's share of the GPU; the rest is headroom for the co-located layout model. |
| `--max-model-len` | 8192 | vLLM max model length. |
| `--max-num-seqs` | 1024 | vLLM max concurrent sequences. |
| `--max-num-batched-tokens` | 16384 | vLLM batched-token budget. |
| `--api-server-count` | 4 | vLLM API server processes. Scale with vCPUs; oversubscribing the CPU hurts throughput. |
| `--speculative-config` | GLM MTP, 3 tokens | JSON passed to vLLM `--speculative-config`; an explicit empty string disables it. |
| `--vllm-extra-args` | none | Extra args appended to `vllm serve`. |
| `--layout-backend` | `onnx` | `onnx` (ONNX Runtime + TensorRT) or `transformers` (PyTorch). |
| `--layout-batch-size` | 4 | Small batches interleave best with OCR decode on the shared GPU. |
| `--layout-threshold` | backend-calibrated (0.5 onnx / 0.3 transformers) | Detection score threshold. |
| `--trt-layout-cache` | `/root/.cache/vllm/trt_layout` | TensorRT layout engine cache dir. |
| `--trt-builder-opt-level` | 1 | TensorRT builder optimization level. |
| `--crop-encode-workers` | 8 | Threads for crop/resize/JPEG preparation. |

### Batch flags (`BatchPolicy`)

| Flag | Default | Notes |
| --- | --- | --- |
| `--batch-docs` | 64 | Maximum PDFs per GPU batch. |
| `--batch-bytes-mb` | 512 | Approximate maximum input MB per GPU batch. |
| `--batch-pages` | unlimited | Maximum PDF pages per GPU batch. |
| `--batch-retries` | 3 | Retries for a failed or preempted batch. |
| `--batch-retry-backoff-seconds` | 10 | Initial retry backoff, in seconds. |

## Outputs

Each document gets a directory under the target, named after its input path with
the `.pdf` suffix dropped:

```text
<target>/
  <stem>/
    content.md       # layout-aware Markdown with page markers
    layout.json      # per-page region list (private _ocr_* keys stripped)
    chunks.jsonl     # only when chunk emission is on
    stats.json       # written LAST; the completion marker
  _run_summary.json  # run-level totals and throughput
```

`stats.json` is the resumability contract: it is written after every other
output, and only `status: "succeeded"` counts as complete, so partial or corrupt
outputs are retried on the next run. `chunks.jsonl` deliberately does **not**
count toward completeness, so outputs written before chunks existed stay valid.

`layout.json` is a list of pages, each a list of regions:

```json
{ "index": 3, "label": "text", "native_label": "paragraph_title",
  "content": "## Methods", "bbox_2d": [102, 331, 894, 372] }
```

`label` is the coarse task label (`text`, `table`, `formula`, `image`);
`native_label` is the original PP-DocLayoutV3 class (`doc_title`,
`paragraph_title`, `footnote`, ...). `bbox_2d` is `[x1, y1, x2, y2]` normalized
to 0–1000. Persisted layout JSON is sanitized: the private `_ocr_usage`,
`_ocr_latency_ms`, `_ocr_status_code`, `_ocr_confidence`, and
`_ocr_confidence_before` keys are stripped.

`_run_summary.json` carries `documents` (total/succeeded/failed/skipped),
`pages.processed`, `regions.ocr`, `tokens`, `durations_seconds.wall`, and a
`throughput` block (`pages_per_second`, `ocr_regions_per_second`, token rates).

## Chunk output

`chunks.jsonl` is one JSON object per line, built purely from the layout JSON —
so the same builder serves batch runs and the hosted API.

Policy: consecutive text regions are packed up to `chunk_max_tokens` and never
across a heading boundary; tables and formulas are atomic chunks; image regions
(null content) are skipped. Heading text starts the next chunk, so chunks read
self-contained, and also feeds `section_path`. Token counts are estimated as
`len(text) // 4` — there is no tokenizer dependency.

| Field | Type | Meaning |
| --- | --- | --- |
| `chunk_id` | string | `"<output_stem>#0000"`, zero-padded, ordered within the document. |
| `doc` | string | The document's relative input path (the API uses `filename`). |
| `section_path` | string[] | Heading stack: `[doc_title]`, or `[doc_title, paragraph_title]`. |
| `pages` | int[] | Sorted 1-based page numbers the chunk spans. |
| `bboxes` | object[] | `{"page": <1-based>, "bbox": [x1, y1, x2, y2]}` per source region, 0–1000 normalized. |
| `kind` | string | `text`, `table`, or `formula`. |
| `text` | string | Chunk content; packed text regions are joined with a blank line. |
| `token_estimate` | int | Estimated tokens for `text`. |

```json
{"chunk_id":"reports/q3#0007","doc":"reports/q3.pdf","section_path":["Q3 Report","Revenue"],"pages":[4],"bboxes":[{"page":4,"bbox":[96,210,902,455]}],"kind":"text","text":"...","token_estimate":118}
```

## Hosted API

`src/bumblebee/api.py` builds a FastAPI app around any object exposing
`async ocr(pdf_bytes) -> DocumentResult`, which is what keeps route tests
GPU-free. It accepts raw PDF bytes and never writes the PDF, rendered pages,
Markdown, layout JSON, or chunks to its control-plane store.

| Route | Auth | Purpose |
| --- | --- | --- |
| `GET /health` | none | Liveness. |
| `GET /v1/privacy` | none | Retention posture: `document_retention`, `ocr_output_retention`, `audit_metadata_retention_days`. |
| `GET /v1/usage` | bearer | The calling tenant's current UTC-month totals. |
| `GET /v1/audit?limit=20` | bearer | Recent request metadata for the calling tenant (limit clamped to 1–100). |
| `POST /v1/parse` | bearer | Parse one PDF. |

`POST /v1/parse` takes the PDF as the raw request body (max 50 MB) and these
query parameters:

| Parameter | Default | Effect |
| --- | --- | --- |
| `chunks` | `true` | Include a `chunks` array in the response. |
| `chunk_max_tokens` | 512 | Token budget per packed text chunk. |
| `filename` | `document.pdf` | Used for `doc` / `chunk_id`, and echoed back. |
| `include_region_metadata` | `false` | Keep `_ocr_confidence` / `_ocr_confidence_before` on layout regions. The other `_ocr_*` keys are always stripped. |

Headers: `Authorization: Bearer <token>` (required) and an optional
`Idempotency-Key` (1–200 characters, scoped per tenant).

```bash
curl -sS -X POST "$BUMBLEBEE_API_URL/v1/parse?filename=report.pdf&chunk_max_tokens=384" \
  -H "Authorization: Bearer $BUMBLEBEE_API_KEY" \
  -H "Content-Type: application/pdf" \
  -H "Idempotency-Key: 7f3c1a9e-..." \
  --data-binary @report.pdf
```

The response is JSON with `filename`, `markdown`, `layout` (sanitized),
`chunks` (when requested), `stats` (`pages`, `regions`, `ocr_regions`, `tokens`,
`seconds`, and per-stage `timings`), and `request` (`id`, `document_retention`,
`audit_metadata_retention_days`). Responses carry `Cache-Control: no-store`,
`X-Bumblebee-Document-Retention: none`, and `X-Request-ID`.

Status codes worth knowing: `400` empty body or malformed idempotency key,
`401` bad or missing token, `409` duplicate `Idempotency-Key`, `413` over 50 MB,
`422` the document failed to parse, `429` monthly page limit reached, `503` the
key configuration is missing or invalid (the API fails closed). The page limit
is checked before parsing and again after — page count is only known once a
document is parsed, so a request that crosses the cap is metered but has its
output withheld.

### Keys and the usage ledger

`src/bumblebee/pilot.py` holds tenant-scoped keys and a SQLite metadata ledger.
Tokens are never stored — only SHA-256 digests, compared in constant time.

```bash
# Multi-tenant, with optional per-tenant page caps
BUMBLEBEE_API_KEYS_JSON='{"acme":{"key":"bb_live_...","monthly_page_limit":50000},"demo":"bb_demo_..."}'

# Single-tenant fallback
BUMBLEBEE_API_KEY=bb_live_...
BUMBLEBEE_DEFAULT_TENANT=default
```

The ledger stores request id, tenant, timestamp, page count, duration, and token
counts — never documents or OCR output. It is single-writer under a thread lock
and sized for a one-container deployment, not for multi-region billing. Rows
older than `BUMBLEBEE_AUDIT_RETENTION_DAYS` (default 30) are pruned on write.

## Modal deployment

Install the extra (`uv sync --extra modal`) and authenticate the `modal` CLI.

### Batch runs

```bash
bumblebee modal --source ./input --target ./output --limit 5
bumblebee modal --source s3://bucket/in --target s3://bucket/out --detach
```

This shells out to `modal run -m bumblebee.modal.run::run`. The run config and
batch policy cross as strict JSON blobs, so the CLI's flag list cannot drift
from the entrypoint. Engine and Modal resource flags cross as `BUMBLEBEE_*`
environment variables instead, because Modal evaluates its `@app.cls` resource
decorators and image at import time. Only explicitly-set flags are forwarded, so
the child process's own environment fallback still applies.

Local-source PDFs are uploaded batch by batch; cloud-source PDFs are listed and
read inside Modal via Modal secrets, so they never bounce through the caller.
`--detach` submits a non-preemptible in-cloud supervisor and returns
immediately — it requires **both** source and target to be cloud URIs.

Modal resource flags (`ModalConfig`), for the `modal` subcommand only:

| Flag | Default |
| --- | --- |
| `--app-name` | `bumblebee` |
| `--gpu` | `A100-40GB` |
| `--cpu-cores` | 16 |
| `--cpu-limit` | none |
| `--scaledown-window` | 900 s |
| `--startup-timeout` | 1200 s |
| `--timeout` | 21600 s |
| `--max-containers` | 1 |
| `--modal-secrets` | none (comma-separated Modal secret names mounted into the workers) |
| `--detach` | off |

The image pins `vllm==0.21.0` on a CUDA 12.9 base and ships the package source
via `PYTHONPATH`. Three Modal volumes are reused across runs:
`huggingface-cache`, `vllm-cache` (weights and compiled kernels, so later
container starts skip the TensorRT build), and `bumblebee-pilot-data` (metadata
ledger only).

### Deploying the hosted API

```bash
modal secret create bumblebee-api BUMBLEBEE_API_KEY=<token>
bumblebee deploy-api        # == modal deploy -m bumblebee.modal.api
```

Deployment requires a Modal secret named `bumblebee-api` carrying either
`BUMBLEBEE_API_KEY` or `BUMBLEBEE_API_KEYS_JSON`. One GPU container holds a
started `DocumentEngine` and serves the FastAPI app; `max_containers` is pinned
to 1 and the idle window defaults to 120 s (`BUMBLEBEE_API_SCALEDOWN_SECONDS`)
so an idle deployment does not burn GPU credit. The usage DB lives at
`/data/usage.sqlite3` on the pilot volume, which is committed after each
successful parse. Cold starts take minutes (model load + vLLM boot) — warm it
before a live demo.

## Web studio

`web/` is a Next.js 16 app, package name `bumblebee-evidence-engine`. The
browser posts a PDF to the same-origin route `POST /api/parse`, which proxies it
to the hosted API with the server-side key attached; the browser never sees the
key, and neither hop persists the PDF. Limits mirror the API (PDF only, 50 MB).
The route generates an `Idempotency-Key` when the client does not send one, and
passes `X-Request-ID` back.

```bash
cd web
cp .env.example .env.local     # then fill in the two values
npm install
npm run dev                    # http://localhost:3000
```

```ini
BUMBLEBEE_API_URL=https://<your-modal-endpoint>.modal.run
BUMBLEBEE_API_KEY=<a design-partner key>
```

Both are server-only — never prefix either with `NEXT_PUBLIC_`. Other scripts:
`npm run build` (standalone build), `npm run start`, `npm run lint`,
`npm run typecheck`.

## Configuration reference

Every config field resolves the same way: explicit value (CLI flag or
constructor argument) → `BUMBLEBEE_*` environment variable → built-in default.
The environment is read when a config object is instantiated, not at import
time. Unknown keys are rejected when a config is built from a mapping — that is
what parses the Modal entrypoint's JSON, where a silently dropped typo would run
a whole batch with the wrong settings.

Most variables mirror a CLI flag one for one: `--pdf-dpi` is
`BUMBLEBEE_PDF_DPI`, `--layout-batch-size` is `BUMBLEBEE_LAYOUT_BATCH_SIZE`, and
so on. The settings below have **no CLI flag** and can only be set through the
environment:

| Variable | Default | Effect |
| --- | --- | --- |
| `BUMBLEBEE_OCR_LOGPROBS` | `1` | Ask vLLM for token logprobs so every region gets a confidence score. |
| `BUMBLEBEE_ADAPTIVE_RETRY` | `1` | Re-render and re-OCR the lowest-confidence regions once at 2x DPI. |
| `BUMBLEBEE_CONFIDENCE_THRESHOLD` | `0.80` | Confidence below which a region becomes a retry candidate. |
| `BUMBLEBEE_LOG_LEVEL` | `INFO` | Level for the `bumblebee` logger (`--log-level` overrides it). |
| `BUMBLEBEE_API_KEYS_JSON` | unset | Tenant → key (or `{"key": ..., "monthly_page_limit": ...}`) mapping. |
| `BUMBLEBEE_API_KEY` | unset | Single-tenant fallback key. |
| `BUMBLEBEE_DEFAULT_TENANT` | `default` | Tenant id for the single-key fallback. |
| `BUMBLEBEE_USAGE_DB` | `:memory:` | Path to the SQLite usage ledger. |
| `BUMBLEBEE_AUDIT_RETENTION_DAYS` | `30` | Metadata retention window, in days. |
| `BUMBLEBEE_API_SCALEDOWN_SECONDS` | `120` | Idle window for the Modal API container. |

Booleans accept `1`, `true`, or `yes`. An empty value counts as unset —
`BUMBLEBEE_SPECULATIVE_CONFIG` is the deliberate exception, where an explicit
empty string disables speculative decoding.

### Cloud credentials

Storage backends dispatch on the URI scheme: `s3://`, `gs://` / `gcs://`, and
`az://` / `abfs://` / `abfss://`.

- **S3** — `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are read by s3fs. Point
  at an S3-compatible endpoint with `BUMBLEBEE_S3_ENDPOINT_URL` (or
  `AWS_ENDPOINT_URL_S3`) and `BUMBLEBEE_S3_REGION` (or `AWS_REGION`).
- **GCS** — gcsfs reads `GOOGLE_APPLICATION_CREDENTIALS` itself.
- **Azure** — `BUMBLEBEE_AZURE_CONNECTION_STRING` (or
  `AZURE_STORAGE_CONNECTION_STRING`); or `BUMBLEBEE_AZURE_ACCOUNT_NAME` plus a
  SAS token or URL (`BUMBLEBEE_AZURE_SAS_URL`, `BUMBLEBEE_AZURE_SAS_TOKEN`, or
  the `AZURE_STORAGE_*` equivalents); or service-principal `AZURE_CLIENT_ID` /
  `AZURE_CLIENT_SECRET` / `AZURE_TENANT_ID`.

For Modal runs, put these in a Modal secret and name it in `--modal-secrets` /
`BUMBLEBEE_MODAL_SECRET_NAMES`.

## Using the engine from Python

```python
from bumblebee import DocumentEngine, EngineConfig, OcrConfig

engine = DocumentEngine(EngineConfig(layout_batch_size=4)).start()   # minutes
try:
    result = await engine.ocr(pdf_bytes)                    # one in-memory PDF
    async for doc in engine.stream(documents, OcrConfig()): # many, as they finish
        ...
    summary = await engine.run(source, target, OcrConfig()) # storage-to-storage
finally:
    engine.stop()
```

`start()` is expensive and idempotent, and there is deliberately no context
manager: create one engine per process and keep it alive. Long-lived workers
normally never call `stop()`. `import bumblebee` and reading `OcrConfig` work
without torch/vLLM installed — `DocumentEngine` is imported lazily on first
attribute access.

## Evals

The harness in `evals/` is not shipped with the package. Install its group in a
separate environment (`uv sync --frozen --group evals`); it conflicts with the
cloud-storage extras.

### DocVQA (parse-then-QA)

Parse the DocVQA validation pages, then have an LLM answer the questions from
the parsed Markdown only — no image access — so the score reflects how much
answerable information the parse retained. The QA step needs
`OPENROUTER_API_KEY`.

```bash
# 1. Prepare (start with a subset; drop --limit for the full split)
uv run --group evals python -m evals.docvqa.prepare --limit 20
#    also: --out, --dataset, --config, --split

# 2. Parse (Modal here; a local GPU run works the same way)
bumblebee modal --source evals/data/docvqa/pdfs --target evals/data/docvqa/parsed

# 3. Answer from the parsed markdown (resumable; checkpoints every answer)
uv run --group evals python -m evals.docvqa.qa --model google/gemma-3-27b-it:free
#    also: --data, --sleep, --limit

# 4. Score with ANLS, the official DocVQA metric
uv run python -m evals.docvqa.score          # also: --answers, --out

# 5. Report accuracy + throughput + cost per 1k pages
uv run python -m evals.report --gpu-hourly-usd 2.00
#    also: --scores, --run-summary, --model, --out
```

Free OpenRouter model ids churn; pass a current one with `--model`. The QA step
checkpoints to `answers.jsonl`, so a rate limit pauses progress rather than
losing it. Publish the QA model, prompt, and any question filtering alongside a
score, or the number is not interpretable.

### olmOCR bench

The official runner expects one Markdown file per PDF at
`<candidate>/<category>/<stem>_pg1_repeat1.md`, while a batch run produces
`<output>/<category>/<stem>/content.md`. The adapter copies only Markdown and
verifies full page coverage before a score can be presented as a complete
benchmark result.

```bash
python -m evals.olmocr_bench.adapter \
  --bench-data evals/data/<bench>/bench_data \
  --source evals/data/<bench>/bumblebee-raw \
  --candidate bumblebee
#    also: --repeat, --overwrite, --allow-incomplete
```

It writes a manifest recording the candidate, timestamp, repository revision,
source, and coverage.

### Measuring throughput

Do not quote throughput figures that were not measured on the hardware in
question. Every run writes `_run_summary.json` at the target with
`pages.processed`, `durations_seconds.wall`, and a `throughput` block; that is
the number to cite, alongside the GPU type, `--pdf-dpi`, `--layout-backend`,
`--api-server-count`, and the corpus. `evals/report.py` turns
`throughput.pages_per_second` plus a `--gpu-hourly-usd` input into a cost per
1,000 pages. Per-document `stats.json` carries the per-stage timing split
(render / layout / crop / OCR, each with its queue-versus-execute breakdown),
which is where to look when a run is slower than expected.

## Development

```bash
make install        # uv sync --frozen --group dev
make format         # ruff import fixes + formatter (alias: make fix)
make format-check   # verify formatting without mutating the tree
make lint           # ruff check
make types          # pyright, strict, with the modal extra (alias: make typecheck)
make unit           # pytest -q, GPU-free (alias: make test)
make quality        # lint + format-check + types
make check          # quality + tests — what CI runs
```

The test suite never needs a GPU: the engine, OCR client, and API are exercised
against fakes. `tests/smoke_test.py` is a release check against an installed
distribution and is not part of `make unit`.

Conventions the toolchain enforces: 120-column lines, Google-style docstrings,
and absolute imports only — relative imports are banned, so always write
`from bumblebee.config import ...`. Pyright runs in strict mode over
`src/bumblebee`. CI runs lint, format-check, types, and tests on pushes to
`main` and on pull requests.

## License

Proprietary and confidential; see `LICENSE`. Internal use only.
