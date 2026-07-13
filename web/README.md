# Bumblebee Studio

The web face of bumblebee: drop a PDF, watch it become markdown + RAG-ready
chunks. Built with [Reflex](https://reflex.dev) — pure Python, full stack.

## Run

```bash
uv sync --group web
cd web
../.venv/Scripts/reflex run --env prod --single-port   # Windows
# ../.venv/bin/reflex run --env prod --single-port     # macOS/Linux
```

Open http://localhost:3000.

## Engines

| Mode | When | What |
| --- | --- | --- |
| Hosted GPU | `BUMBLEBEE_API_URL` + `BUMBLEBEE_API_KEY` set | Full layout + hybrid OCR via the Modal-deployed `/v1/parse` API |
| Local demo | otherwise | pypdfium2 text-layer extraction + the same `build_chunks` the product ships; scanned PDFs are rejected with a clear message |

## Production

`reflex run --env prod --single-port` serves frontend + backend on one port
(3000) — put it behind any reverse proxy. Uploads are capped at 50 MB and the
markdown preview at 80k chars; full outputs are always downloadable.
