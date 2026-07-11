# bumblebee (built on bumblebee)

**bumblebee** is a RAG-ingestion product on the bumblebee engine: PDF buckets in, layout-aware
markdown + RAG-ready `chunks.jsonl` out — via CLI or a hosted API. The `bumblebee` command is
the same CLI as `bumblebee` with chunk emission on by default; see
[Bumblebee: RAG ingestion](#bumblebee-rag-ingestion) below.

`bumblebee` was born from the need to OCR a large corpus of PDF documents with complex layouts quickly and at low cost. It converts PDF documents to markdown and json, including documents with complex layouts and tables. It uses  `pp-doclayout-v3` for layout detection and GLM-OCR to recognize text. We measured throughput at more than 14 pages per second on our benchmark. For other datasets, we reached up to 20 pages per second. It runs on a single L40S GPU (or similar).

Read [our blog](https://github.com/Sri-Krishna-V/bumblebee) to learn all about the optimizations that we apply.

<img src="assets/readme-example.png" alt="A PDF page with bumblebee layout bounding boxes next to the rendered Markdown output." width="900">

## Benchmarks

We use 100 PDF documents from the [European Medicines Agency](https://www.ema.europa.eu/en/medicines/download-medicine-data) to benchmark throughput. The documents are typical business documents with a mix of tables, text, and some figures. All metrics are reported end-to-end including download and upload. Warm-up (model loading and initialization) is excluded from our measurements. We compare `bumblebee` against the official GLM-OCR SDK. The benchmarks ran on a single L40S for both solutions.

| Metric | GLM-OCR SDK | bumblebee |
| --- | ---: | ---: |
| Pages/sec | 5.29 | 14.26 |
| Output tokens/sec | 4,316 | 11,801 |
| Page regions/sec | 75 | 205.4 |


## Features

| Feature | Description |
| --- | --- |
| Fast & cheap | Fastest and cheapest VLM-based OCR that we know of. |
| VLM-based OCR | State-of-the-art text recognition that handles difficult text extraction well. |
| Layout detection | Recognizes bounding boxes for headings, paragraphs, images, tables, and more. |
| OCR from cloud storage | Processes PDFs from and writes outputs to S3 buckets, Azure storage containers, or Google Cloud buckets. |
| One-line deployment | Runs the full pipeline on Modal with a single CLI command. |
| Bare-metal deployment | Runs on any machine with an NVIDIA GPU. |


## Bumblebee: RAG ingestion

The `bumblebee` console script runs the same commands as `bumblebee`, with chunk emission on
by default: every document gets a `chunks.jsonl` beside its markdown, ready to load into any
vector store or search index.

```bash
bumblebee --source ./input --target ./output          # local GPU
bumblebee modal --source s3://in --target s3://out    # Modal
bumblebee --chunks --source ./input --target ./output # same thing, explicit flag
```

Each line of `chunks.jsonl` is one retrieval chunk:

```json
{"chunk_id": "report#0007", "doc": "report.pdf", "section_path": ["Annual Report", "4. Risk Factors"],
 "pages": [12], "bboxes": [{"page": 12, "bbox": [112, 340, 890, 610]}],
 "kind": "text", "text": "...", "token_estimate": 412}
```

Chunking is structure-aware: consecutive text regions pack up to `--chunk-max-tokens` and never
cross heading boundaries, headings feed `section_path`, tables and formulas stay atomic
(`kind: "table" | "formula"`), and every chunk carries page numbers + bounding boxes so a RAG
answer can highlight its exact source on the original page.

### Hosted API

Deploy a persistent parse endpoint on Modal (one GPU container, bearer-token auth):

```bash
modal secret create bumblebee-api BUMBLEBEE_API_KEY=<token>
bumblebee deploy-api

curl -X POST "https://<your-modal-url>/v1/parse?filename=report.pdf" \
  -H "Authorization: Bearer <token>" \
  --data-binary @report.pdf
```

The response carries `markdown`, `layout`, `chunks`, and `stats` (`?chunks=false` to skip
chunking). Cost note: the API container scales down after 120s idle
(`BUMBLEBEE_API_SCALEDOWN_SECONDS`); a warm GPU costs ~$2/hour and a cold start takes minutes
(model load), so warm it up before a live demo.

### Accuracy features

- **Born-digital text layer** (`--text-layer auto`, default): text regions on pages with a
  trustworthy embedded text layer are read directly from the PDF — exact and GPU-free; tables
  and formulas always go to the VLM. `off` disables.
- **Confidence scores** (`BUMBLEBEE_OCR_LOGPROBS`, default on): every OCR region gets
  `exp(mean token logprob)`; per-document min/mean and a low-confidence count land in
  `stats.json` under `ocr_requests.confidence`.
- **Adaptive retry** (`BUMBLEBEE_ADAPTIVE_RETRY`, default on): a document's lowest-confidence
  regions (below `BUMBLEBEE_CONFIDENCE_THRESHOLD`, default 0.80; at most 10% of its regions)
  are re-rendered at 2x DPI and re-OCRed once, keeping the better result.

### Benchmark harness

`evals/` holds the DocVQA parse-then-QA benchmark harness (accuracy + throughput + $/1k pages
in one report). See [evals/README.md](evals/README.md).

## Use on Modal (easiest, no local GPU required)

Log into Modal or register a new account. If you are already logged in, skip this step.

```bash
uvx --python 3.12 modal login
```

Then run `bumblebee` directly with `uvx`; no project files or persistent install are needed. The local command only needs the `modal` extra. The GPU stack is installed in the remote Modal image.

```bash
# run with local files
# startup takes a few minutes because models need to be loaded and CUDA graphs compiled
# use --limit 1 for a smoke test; omit it to process all incomplete PDFs
uvx --python 3.12 --from 'bumblebee[modal]' bumblebee modal \
  --source input_folder/with_pdfs/ \
  --target output_folder/for_results/ \
  --limit 1

# cloud storage to cloud storage can detach from the process
uvx --python 3.12 --from 'bumblebee[modal]' bumblebee modal \
  --source "s3://in" \
  --target "s3://out" \
  --detach
```

## Use on bare metal GPU

Follow these instructions, to run `bumblebee` on any GPU machine.

Clone the repository.

```bash
git clone https://github.com/Sri-Krishna-V/bumblebee.git
cd bumblebee
```

Now build the docker image to run it or directly install the project into a virtual environment.

```bash
# docker
docker build -t bumblebee .

# Local in/out: mount the folders and reuse the cached weights.
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$PWD/input:/data/in" -v "$PWD/output:/data/out" \
  bumblebee --source /data/in --target /data/out

# Cloud in/out: no mounts — pass URIs plus credentials as env (-e).
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  bumblebee --source s3://bucket/in --target s3://bucket/out --limit 100


# Virtual environment (uv creates .venv automatically if needed)
uv sync --python 3.12 --frozen --no-dev --extra gpu --extra trt --extra s3   # add --extra gcs/azure as needed

uv run bumblebee --source ./input --target ./output
uv run bumblebee --source s3://bucket/in --target s3://bucket/out --limit 100
```

## Use in code

```python
from bumblebee import EngineConfig, OcrConfig, DocumentEngine

engine = DocumentEngine().start()          # loads layout model, boots vLLM

# Startup knobs via EngineConfig, per-run knobs via OcrConfig (or per call);
# unset fields fall back to BUMBLEBEE_* env vars, then the built-in defaults.
engine = DocumentEngine(
    EngineConfig(gpu_memory_utilization=0.7),
    run_config=OcrConfig(pdf_dpi=150),
).start()

result = await engine.ocr(pdf_bytes)            # one PDF, in memory
print(result.markdown, result.tokens.total_tokens)

async for doc in engine.stream(documents, config):   # many, as each finishes
    ...

summary = await engine.run("s3://in", "s3://out")    # storage-to-storage, resumable
```

## Outputs

For each `foo.pdf` under the source, bumblebee writes one directory named after
the relative stem in the target:

- `foo/content.md` — GLM-OCR-formatted Markdown
- `foo/layout.json` — per-page region JSON
- `foo/stats.json` — per-document timings, token usage, region counts and completion marker
- `foo/chunks.jsonl` — RAG-ready chunks (only with `--chunks` / the `bumblebee` CLI)

A run-level `_run_summary.json` is written too, with document/page/token totals,
throughput, and per-batch status.

## Configuration

Most settings resolve the same way: **explicit value → `BUMBLEBEE_*` environment
variable → built-in default**. "Explicit" means a CLI flag (`--pdf-dpi 150`,
`--max-num-seqs 512`, `--gpu H100`) or a constructor argument in code
(`DocumentEngine(EngineConfig(...))`, `OcrConfig(...)`). Plain CLI routing flags are
called out below. OCR crops are always sent to vLLM as inline base64 data URLs.

The tables below list every operational flag on `bumblebee` and
`bumblebee modal` (except `--help`, which prints the command help and exits).

**Common CLI flags** (both commands):

| Flag | Default | Explanation |
| --- | --- | --- |
| `--source` | `./input` | Local folder/file or cloud URI containing PDFs. Plain CLI flag. |
| `--target` | `./output` | Local folder or cloud URI for OCR outputs. Plain CLI flag. |
| `--limit` | all incomplete PDFs | Maximum number of incomplete PDFs to process in this run. Plain CLI flag. |
| `--log-level` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, ...); also reads `BUMBLEBEE_LOG_LEVEL`. |

**Run settings** (`OcrConfig`, both commands):

| Flag | Default | Explanation |
| --- | --- | --- |
| `--force/--no-force` | off | Reprocess documents whose outputs are already complete. |
| `--pdf-dpi` | `100` | PDF render DPI. 100 is throughput-tuned; raise for accuracy. |
| `--pdf-max-side` | `3500` | Longest rendered page side in pixels; larger pages are downscaled. |
| `--page-chunk-size` | `8` | Pages streamed through render → layout → crop → OCR as one unit. |
| `--jpeg-quality` | `90` | JPEG quality for OCR crops before base64 encoding. |
| `--max-tokens-text` | `2048` | Maximum generated tokens for text regions. |
| `--max-tokens-formula` | `2048` | Maximum generated tokens for formula regions. |
| `--max-tokens-table` | `4096` | Maximum generated tokens for table regions. |
| `--temperature` | `0.0` | OCR generation temperature. |
| `--top-p` | `0.00001` | OCR generation nucleus-sampling `top_p`. |
| `--chunks/--no-chunks` | off (`bumblebee`: on) | Write RAG-ready `chunks.jsonl` beside each document. |
| `--chunk-max-tokens` | `512` | Approximate token budget per packed text chunk. |
| `--text-layer` | `auto` | Serve text regions from born-digital PDFs' embedded text (`auto`) or always OCR (`off`). |
| `--max-inflight-pdfs` | `16` local, `32` on Modal | Documents rendered, laid out, and cropped concurrently. Modal uses 32 only when neither flag nor env var is set. |
| `--ocr-request-concurrency` | `1024` | Concurrent OCR requests to the vLLM server; keep near `--max-num-seqs`. |
| `--storage-check-workers` | `64` | Threads reading completion markers when resuming against cloud targets. |

Environment-only run settings (no CLI flag): `BUMBLEBEE_OCR_LOGPROBS` (default `1`; per-region
confidence scores), `BUMBLEBEE_ADAPTIVE_RETRY` (default `1`; re-OCR low-confidence regions at
2x DPI), `BUMBLEBEE_CONFIDENCE_THRESHOLD` (default `0.80`).

**Engine settings** (`EngineConfig`, both commands; startup-scoped):

| Flag | Default | Explanation |
| --- | --- | --- |
| `--vllm-port` | `8000` | Local port for the OpenAI-compatible vLLM server. |
| `--vllm-health-timeout` | `900` | Seconds to wait for the vLLM server to become healthy. |
| `--gpu-memory-utilization` | `0.60` | vLLM's share of GPU memory; remaining headroom is for the co-located layout model. |
| `--max-model-len` | `8192` | vLLM maximum model context length. |
| `--max-num-seqs` | `1024` | vLLM maximum concurrent sequences. OOM-safe to raise within the reserved KV pool. |
| `--max-num-batched-tokens` | `16384` | vLLM batched-token budget. |
| `--api-server-count` | `4` | vLLM API server processes; scale with available CPU cores. |
| `--speculative-config` | `{"method": "mtp", "num_speculative_tokens": 3}` | JSON passed to vLLM `--speculative-config`; an empty string disables it. |
| `--vllm-extra-args` | empty | Extra shell-style arguments appended to `vllm serve`. |
| `--layout-backend` | `onnx` | Layout detector backend: `onnx` (ONNX Runtime + TensorRT FP16) or `transformers` (PyTorch). |
| `--layout-batch-size` | `4` | Layout detection batch size; small batches interleave best with OCR. |
| `--layout-threshold` | backend default | Layout detection score threshold: `0.5` for `onnx`, `0.3` for `transformers`. |
| `--trt-layout-cache` | `/root/.cache/vllm/trt_layout` | TensorRT layout engine cache directory. |
| `--trt-builder-opt-level` | `1` | TensorRT builder optimization level. |
| `--crop-encode-workers` | `8` | Threads for crop/resize/JPEG/base64 preparation. |

**Batch policy** (`BatchPolicy`, both commands):

| Flag | Default | Explanation |
| --- | --- | --- |
| `--batch-docs` | `64` | Maximum PDFs per GPU batch. |
| `--batch-bytes-mb` | `512` | Approximate maximum input size per GPU batch, in MB (`BUMBLEBEE_BATCH_MAX_BYTES` is bytes). |
| `--batch-pages` | unlimited | Maximum PDF pages per GPU batch. |
| `--batch-retries` | `3` | Retries for a failed or preempted GPU batch. |
| `--batch-retry-backoff-seconds` | `10` | Initial retry backoff seconds for failed batches. |

**Modal-only flags** (`bumblebee modal`):

| Flag | Default | Explanation |
| --- | --- | --- |
| `--detach/--no-detach` | off | Submit a detached cloud run and return immediately. Requires cloud source and cloud target. Plain CLI flag. |
| `--app-name` | `bumblebee` | Modal app name. |
| `--gpu` | `A100-40GB` | Modal GPU type. |
| `--cpu-cores` | `16` | Modal CPU request. |
| `--cpu-limit` | none | Optional Modal CPU limit. |
| `--scaledown-window` | `900` | Modal container scaledown window in seconds. |
| `--startup-timeout` | `1200` | Modal container startup timeout in seconds. |
| `--timeout` | `21600` | Modal call timeout in seconds. |
| `--max-containers` | `1` | Maximum Modal GPU containers. |
| `--modal-secrets` | none | Comma-separated Modal secret names mounted into workers (`BUMBLEBEE_MODAL_SECRET_NAMES`). |


## Development

Install the development environment with uv:

```bash
uv sync --frozen --group dev
```

Common checks are available through the `Makefile`:

```bash
make lint          # Ruff lint checks
make format-check  # verify Ruff formatting
make types         # Pyright, including the Modal extra
make unit          # GPU-free test suite
make check         # lint, formatting, types, and tests
```

## Acknowledgements

This work is only a thin layer on strong foundations:

- [GLM OCR SDK](https://github.com/zai-org/GLM-OCR)
- [PaddlePaddle/PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
- [bndos/pp-doclayout-v3-trt](https://huggingface.co/bndos/pp-doclayout-v3-trt)

## License

Apache-2.0.
