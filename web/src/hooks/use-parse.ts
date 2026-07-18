"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { type ApiError, isParseResponse, type ParseResponse } from "@/lib/contracts";

export type ParseStatus = "idle" | "ready" | "parsing" | "complete" | "error";

export type ParseState = {
  file: File | null;
  previewUrl: string | null;
  result: ParseResponse | null;
  error: string | null;
  status: ParseStatus;
};

const initialState: ParseState = {
  file: null,
  previewUrl: null,
  result: null,
  error: null,
  status: "idle",
};

function readableError(value: unknown): string {
  if (value && typeof value === "object" && "detail" in value) {
    const detail = (value as ApiError).detail;
    if (typeof detail === "string") return detail;
  }
  return "The parse could not be completed. Your file was not retained.";
}

export function useParse() {
  const [state, setState] = useState<ParseState>(initialState);
  const abortRef = useRef<AbortController | null>(null);
  const previewRef = useRef<string | null>(null);

  const revokePreview = useCallback(() => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = null;
  }, []);

  useEffect(() => () => revokePreview(), [revokePreview]);

  const clear = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    revokePreview();
    setState(initialState);
  }, [revokePreview]);

  const selectFile = useCallback(
    (file: File) => {
      abortRef.current?.abort();
      revokePreview();
      const previewUrl = URL.createObjectURL(file);
      previewRef.current = previewUrl;
      setState({ file, previewUrl, result: null, error: null, status: "ready" });
    },
    [revokePreview],
  );

  const parse = useCallback(async () => {
    if (!state.file || state.status === "parsing") return;
    const controller = new AbortController();
    abortRef.current = controller;
    setState((current) => ({ ...current, error: null, status: "parsing" }));

    try {
      const form = new FormData();
      form.set("document", state.file, state.file.name);
      const response = await fetch("/api/parse", {
        method: "POST",
        body: form,
        cache: "no-store",
        headers: { "Idempotency-Key": crypto.randomUUID() },
        signal: controller.signal,
      });
      const body: unknown = await response.json().catch(() => null);
      if (!response.ok || !isParseResponse(body)) {
        throw new Error(readableError(body));
      }
      setState((current) => ({ ...current, result: body, status: "complete" }));
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setState((current) => ({
        ...current,
        error: error instanceof Error ? error.message : readableError(null),
        status: "error",
      }));
    } finally {
      abortRef.current = null;
    }
  }, [state.file, state.status]);

  return { clear, parse, selectFile, state };
}
