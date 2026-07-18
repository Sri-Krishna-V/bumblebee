# Bumblebee Evidence Engine

The Next.js product surface for Bumblebee's private, source-grounded PDF
ingestion pilot. It sends a selected PDF to the same-origin server route, which
proxies it to the hosted Bumblebee API. The browser never receives the upstream
API key, and neither the browser route nor Bumblebee's API persists the PDF.

## Configure

Copy `.env.example` to `.env.local` and set the server-only values:

```bash
BUMBLEBEE_API_URL=https://your-modal-endpoint.modal.run
BUMBLEBEE_API_KEY=your-design-partner-key
```

Do not use `NEXT_PUBLIC_` for either value.

## Run

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000. The Studio supports PDFs up to 50 MB, previews the
local source only in the browser, and offers Markdown, chunks, and a
privacy/usage receipt after a parse.

## Production

`npm run build` creates a standalone Next.js build. Deploy the frontend where
its server environment can reach the Modal API, then set the two values above
in that deployment's private environment settings.
