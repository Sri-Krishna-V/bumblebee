import { randomUUID } from "node:crypto";

import { NextResponse } from "next/server";

export const runtime = "nodejs";

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

function failure(detail: string, status: number) {
  return NextResponse.json(
    { detail },
    { status, headers: { "Cache-Control": "no-store", "X-Bumblebee-Document-Retention": "none" } },
  );
}

function upstreamUrl(): URL | null {
  const base = process.env.BUMBLEBEE_API_URL?.replace(/\/$/, "");
  if (!base) return null;
  try {
    return new URL(`${base}/v1/parse`);
  } catch {
    return null;
  }
}

export async function POST(request: Request) {
  const endpoint = upstreamUrl();
  const apiKey = process.env.BUMBLEBEE_API_KEY;
  if (!endpoint || !apiKey) {
    return failure("The Evidence Engine is not connected to a Bumblebee API yet.", 503);
  }

  const form = await request.formData().catch(() => null);
  const document = form?.get("document");
  if (!(document instanceof File)) return failure("Choose a PDF document to parse.", 400);
  if (!document.name.toLowerCase().endsWith(".pdf")) return failure("Bumblebee accepts PDF documents only.", 415);
  if (document.size === 0) return failure("The selected PDF is empty.", 400);
  if (document.size > MAX_UPLOAD_BYTES) return failure("That PDF exceeds the 50 MB pilot limit.", 413);

  endpoint.searchParams.set("filename", document.name);
  const idempotencyKey = request.headers.get("idempotency-key") ?? randomUUID();

  let upstream: Response;
  try {
    upstream = await fetch(endpoint, {
      method: "POST",
      body: await document.arrayBuffer(),
      cache: "no-store",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/pdf",
        "Idempotency-Key": idempotencyKey,
      },
    });
  } catch {
    return failure("Bumblebee could not reach the parse worker. Your file was not retained.", 502);
  }

  const body = await upstream.arrayBuffer();
  const requestId = upstream.headers.get("x-request-id");
  return new NextResponse(body, {
    status: upstream.status,
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": upstream.headers.get("content-type") ?? "application/json; charset=utf-8",
      "X-Bumblebee-Document-Retention": "none",
      ...(requestId ? { "X-Request-ID": requestId } : {}),
    },
  });
}
