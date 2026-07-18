import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  const configured = Boolean(process.env.BUMBLEBEE_API_URL && process.env.BUMBLEBEE_API_KEY);
  return NextResponse.json(
    {
      status: configured ? "ready" : "configuration_required",
      document_retention: "none",
      ocr_output_retention: "none",
      max_upload_mb: 50,
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
