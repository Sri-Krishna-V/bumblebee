"use client";

import { useRef, useState } from "react";

import { useParse, type ParseState } from "@/hooks/use-parse";
import type { ParseResponse } from "@/lib/contracts";

type Tab = "source" | "markdown" | "chunks" | "receipt";

function formatBytes(bytes: number) {
  if (bytes < 1_000_000) return `${Math.ceil(bytes / 1_000)} KB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function uploadLabel(state: ParseState) {
  if (state.status === "parsing") return "Tracing each region…";
  if (state.status === "complete") return "Parsed into evidence";
  if (state.status === "error") return "The trace needs another pass";
  if (state.file) return "Ready when you are";
  return "Drop a PDF into the hive";
}

function UploadSurface({
  state,
  onClear,
  onParse,
  onSelect,
}: {
  state: ParseState;
  onClear: () => void;
  onParse: () => void;
  onSelect: (file: File) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const busy = state.status === "parsing";

  function accept(file: File | undefined) {
    if (file) onSelect(file);
  }

  return (
    <section className={`upload-surface ${dragging ? "is-dragging" : ""}`} aria-label="PDF upload">
      <div className="surface-grid" aria-hidden="true" />
      <div className="surface-topline">
        <span className="micro-label">01 / source material</span>
        <span className="retention-note">vanishes after response</span>
      </div>
      <div
        className="drop-target"
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          setDragging(false);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          accept(event.dataTransfer.files.item(0) ?? undefined);
        }}
      >
        <span className="document-glyph" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <p className="upload-title">{uploadLabel(state)}</p>
        {state.file ? (
          <div className="selected-file">
            <span>{state.file.name}</span>
            <small>{formatBytes(state.file.size)} · PDF</small>
          </div>
        ) : (
          <p className="upload-copy">A contract, report, archive, or technical PDF up to 50 MB.</p>
        )}
        <input
          accept="application/pdf,.pdf"
          className="visually-hidden"
          onChange={(event) => accept(event.target.files?.item(0) ?? undefined)}
          ref={input}
          type="file"
        />
        <div className="upload-actions">
          <button className="button button--paper" onClick={() => input.current?.click()} type="button">
            {state.file ? "Choose another" : "Choose PDF"}
          </button>
          {state.file && (
            <button className="button button--amber" disabled={busy} onClick={onParse} type="button">
              {busy ? "Reading…" : "Make evidence"}
            </button>
          )}
        </div>
        {state.file && (
          <button className="quiet-action" disabled={busy} onClick={onClear} type="button">
            Remove local file
          </button>
        )}
        {state.error && <p className="upload-error" role="alert">{state.error}</p>}
      </div>
      <div className="surface-footer">
        <span>PDF stays in transit only</span>
        <span>Metadata audit: 30 days</span>
      </div>
    </section>
  );
}

function ResultPanel({ result, previewUrl }: { result: ParseResponse; previewUrl: string | null }) {
  const [tab, setTab] = useState<Tab>("markdown");
  const [copied, setCopied] = useState(false);

  async function copyMarkdown() {
    await navigator.clipboard.writeText(result.markdown);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  function downloadResult() {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = `${result.filename.replace(/\.pdf$/i, "")}-bumblebee.json`;
    anchor.click();
    URL.revokeObjectURL(href);
  }

  const chunks = result.chunks ?? [];
  const tabs: Array<[Tab, string, number | null]> = [
    ["source", "Local source", null],
    ["markdown", "Markdown", null],
    ["chunks", "Chunks", chunks.length],
    ["receipt", "Receipt", null],
  ];

  return (
    <section className="result-panel" aria-live="polite">
      <div className="result-heading">
        <div>
          <span className="micro-label">02 / evidence produced</span>
          <h2>{result.filename}</h2>
        </div>
        <span className="verified-seal">trace complete</span>
      </div>

      <div className="metric-strip" aria-label="Parse metrics">
        <div><strong>{formatNumber(result.stats.pages)}</strong><span>pages</span></div>
        <div><strong>{formatNumber(result.stats.regions)}</strong><span>regions</span></div>
        <div><strong>{result.stats.seconds.toFixed(1)}s</strong><span>wall time</span></div>
        <div><strong>{formatNumber(result.stats.tokens.total_tokens)}</strong><span>tokens</span></div>
      </div>

      <div className="result-toolbar" role="tablist" aria-label="Evidence views">
        {tabs.map(([value, label, count]) => (
          <button
            aria-selected={tab === value}
            className={tab === value ? "result-tab is-active" : "result-tab"}
            key={value}
            onClick={() => setTab(value)}
            role="tab"
            type="button"
          >
            {label}{count !== null ? <b>{count}</b> : null}
          </button>
        ))}
        <span className="toolbar-spacer" />
        <button className="icon-action" onClick={copyMarkdown} type="button">
          {copied ? "Copied" : "Copy .md"}
        </button>
        <button className="icon-action" onClick={downloadResult} type="button">Export JSON</button>
      </div>

      <div className="evidence-view">
        {tab === "source" && (
          <div className="source-view">
            {previewUrl ? <iframe src={previewUrl} title="Local PDF preview" /> : <p>Local preview is no longer available.</p>}
            <p className="view-note">This browser preview is local to your device; it is not an upload archive.</p>
          </div>
        )}
        {tab === "markdown" && <pre className="markdown-view">{result.markdown}</pre>}
        {tab === "chunks" && (
          <div className="chunk-list">
            {chunks.length ? chunks.map((chunk) => (
              <article className="chunk" key={chunk.chunk_id}>
                <div><span>{chunk.chunk_id}</span><small>{chunk.section_path.join(" / ") || "Document"}</small></div>
                <p>{chunk.text}</p>
              </article>
            )) : <p className="empty-state">Chunk emission was disabled for this parse.</p>}
          </div>
        )}
        {tab === "receipt" && (
          <dl className="receipt">
            <div><dt>Request trace</dt><dd>{result.request.id}</dd></div>
            <div><dt>Document retention</dt><dd>None</dd></div>
            <div><dt>OCR output retention</dt><dd>None</dd></div>
            <div><dt>Metadata audit</dt><dd>{result.request.audit_metadata_retention_days} days</dd></div>
            <div><dt>Model work</dt><dd>{formatNumber(result.stats.ocr_regions)} OCR regions</dd></div>
          </dl>
        )}
      </div>
    </section>
  );
}

export function Workbench() {
  const { clear, parse, selectFile, state } = useParse();

  return (
    <section className="workbench" id="studio">
      <div className="workbench-intro">
        <div>
          <p className="eyebrow">The pilot studio</p>
          <h2>Extract first.<br /><em>Trust later.</em></h2>
        </div>
        <p>Every parse comes back with inspectable Markdown, retrieval-ready chunks, and a small receipt of what happened. The document itself is not kept.</p>
      </div>
      <div className="workbench-grid">
        <UploadSurface onClear={clear} onParse={parse} onSelect={selectFile} state={state} />
        <aside className="trust-rail" aria-label="Pilot guardrails">
          <span className="micro-label">pilot guardrails</span>
          <ol>
            <li><b>01</b><span>Tenant-scoped API access</span></li>
            <li><b>02</b><span>Page-metered usage caps</span></li>
            <li><b>03</b><span>Idempotent processing</span></li>
            <li><b>04</b><span>No document retention</span></li>
          </ol>
          <p>Built for a design partner, not an anonymous upload funnel.</p>
        </aside>
      </div>
      {state.status === "parsing" && (
        <div className="parsing-line" role="status"><span />The engine is tracing reading order, tables, and regions.</div>
      )}
      {state.result && <ResultPanel previewUrl={state.previewUrl} result={state.result} />}
    </section>
  );
}
