export type TokenUsage = {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
};

export type ParseStats = {
  pages: number;
  regions: number;
  ocr_regions: number;
  seconds: number;
  tokens: TokenUsage;
  timings: Record<string, number>;
};

export type EvidenceChunk = {
  chunk_id: string;
  text: string;
  section_path: string[];
  [key: string]: unknown;
};

export type ParseResponse = {
  filename: string;
  markdown: string;
  layout: unknown[];
  chunks?: EvidenceChunk[];
  stats: ParseStats;
  request: {
    id: string;
    document_retention: "none";
    audit_metadata_retention_days: number;
  };
};

export type ApiError = {
  detail?: string;
};

export function isParseResponse(value: unknown): value is ParseResponse {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return typeof record.markdown === "string" && typeof record.filename === "string" && !!record.stats;
}
