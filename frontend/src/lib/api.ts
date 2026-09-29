// Fetch helpers + TypeScript types for the doc-pilot backend API.
// Base URL is configurable via NEXT_PUBLIC_API_URL (see .env.example),
// defaulting to the local backend dev server.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export type DocumentStatus = "uploaded" | "processing" | "extracted" | "failed" | "refused";

export interface DocumentListItem {
  id: string;
  filename: string;
  mime_type: string;
  status: DocumentStatus;
  created_at: string;
}

export interface DocumentCreateResponse {
  id: string;
  filename: string;
  status: DocumentStatus;
  created_at: string;
}

// A leaf value as stored by the backend: { value: <scalar|null>, confidence }.
export interface FieldLeaf<T = unknown> {
  value: T;
  confidence: number;
}

export interface LineItem {
  description: FieldLeaf<string | null> | null;
  quantity: FieldLeaf<number | null> | null;
  unit_price: FieldLeaf<number | null> | null;
  total: FieldLeaf<number | null> | null;
}

// ExtractedFieldOut.value shape depends on field_name: line_items stores the
// array directly (its own confidence lives in the `confidence` column below),
// every other field stores the {value, confidence} leaf as-is.
export type ExtractedFieldValue = FieldLeaf | LineItem[] | null;

// Set once a human has resolved the field from the review queue.
export type ReviewAction = "approved" | "corrected";

export interface ExtractedField {
  id: string;
  field_name: string;
  value: ExtractedFieldValue;
  confidence: number;
  needs_review: boolean;
  reviewed_at: string | null;
  review_action: ReviewAction | null;
  // The human-supplied *semantic* value (scalar for scalar fields, the
  // plain-value line-item array for line_items) — only set when
  // review_action === "corrected". The original `value` is never
  // overwritten by a correction.
  corrected_value: unknown;
}

export interface Extraction {
  id: string;
  prompt_version: string;
  model: string;
  cost_usd: number;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  created_at: string;
  fields: ExtractedField[];
}

export interface DocumentDetail {
  id: string;
  filename: string;
  mime_type: string;
  status: DocumentStatus;
  created_at: string;
  extraction: Extraction | null;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") {
        // The common FastAPI HTTPException shape.
        detail = body.detail;
      } else if (body.detail !== undefined) {
        // Validation errors (422) send `detail` as an array of objects
        // ({loc, msg, type}); stringify rather than let it render as
        // "[object Object]".
        detail = JSON.stringify(body.detail);
      }
    } catch {
      // response body wasn't JSON; fall back to statusText
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export async function listDocuments(
  limit = 50,
  offset = 0
): Promise<DocumentListItem[]> {
  const res = await fetch(
    `${API_URL}/documents?limit=${limit}&offset=${offset}`,
    { cache: "no-store" }
  );
  return handle<DocumentListItem[]>(res);
}

export async function getDocument(id: string): Promise<DocumentDetail> {
  const res = await fetch(`${API_URL}/documents/${encodeURIComponent(id)}`, {
    cache: "no-store",
  });
  return handle<DocumentDetail>(res);
}

export function documentFileUrl(id: string): string {
  return `${API_URL}/documents/${encodeURIComponent(id)}/file`;
}

export interface ReviewQueueItem {
  field_id: string;
  field_name: string;
  value: ExtractedFieldValue;
  confidence: number;
  document_id: string;
  filename: string;
  extraction_id: string;
  model: string;
  extracted_at: string;
}

export async function getReviewQueue(
  limit = 50,
  offset = 0
): Promise<ReviewQueueItem[]> {
  const res = await fetch(
    `${API_URL}/review/queue?limit=${limit}&offset=${offset}`,
    { cache: "no-store" }
  );
  return handle<ReviewQueueItem[]>(res);
}

export async function getReviewQueueCount(): Promise<number> {
  const res = await fetch(`${API_URL}/review/queue/count`, {
    cache: "no-store",
  });
  const body = await handle<{ pending: number }>(res);
  return body.pending;
}

export async function resolveReviewField(
  fieldId: string,
  resolution:
    | { action: "approve" }
    | { action: "correct"; corrected_value: unknown }
): Promise<ExtractedField> {
  const res = await fetch(
    `${API_URL}/review/fields/${encodeURIComponent(fieldId)}/resolve`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(resolution),
    }
  );
  return handle<ExtractedField>(res);
}

export interface StatsReview {
  pending: number;
  approved: number;
  corrected: number;
}

export interface LastEval {
  model: string | null;
  prompt_version: string | null;
  dataset_version: string | null;
  started_at_utc: string;
  overall_accuracy: number | null;
  caught_by_review: number | null;
  mean_cost_per_doc: number;
  n_scored: number;
}

export interface Stats {
  documents_total: number;
  documents_by_status: Record<string, number>;
  documents_processed: number;
  extractions_total: number;
  total_cost_usd: number;
  mean_cost_per_doc: number | null;
  total_input_tokens: number;
  total_output_tokens: number;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  review: StatsReview;
  last_eval: LastEval | null;
}

export async function getStats(): Promise<Stats> {
  const res = await fetch(`${API_URL}/stats`, { cache: "no-store" });
  return handle<Stats>(res);
}

export async function uploadDocument(file: File): Promise<DocumentCreateResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_URL}/documents`, {
    method: "POST",
    body: formData,
  });
  return handle<DocumentCreateResponse>(res);
}

// ---- Retrieval: GET /search ------------------------------------------------

export type SearchMode = "hybrid" | "dense" | "lexical";

// One retrieved chunk and its citation: `text` is exactly the page's
// stored text at [char_start, char_end).
export interface SearchHit {
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  chunk_index: number;
  char_start: number;
  char_end: number;
  text: string;
  score: number;
  dense_rank: number | null;
  lexical_rank: number | null;
}

export interface SearchResponse {
  query: string;
  mode: SearchMode;
  embedding_model: string;
  results: SearchHit[];
}

export async function searchDocuments(
  q: string,
  mode: SearchMode = "hybrid",
  k = 8
): Promise<SearchResponse> {
  const params = new URLSearchParams({ q, mode, k: String(k) });
  const res = await fetch(`${API_URL}/search?${params}`, { cache: "no-store" });
  return handle<SearchResponse>(res);
}

// ---- Agent: POST /ask -------------------------------------------------------

export type AskStatus = "answered" | "refused" | "truncated" | "step_limit" | "budget_exceeded";

// `cited_text` is copied by the API from the tool result, never written by
// the model. Page passages carry a char span; extraction records name the
// fields cited.
export interface Citation {
  number: number;
  document_id: string;
  filename: string;
  page_number: number | null;
  kind: "chunk" | "page" | "record";
  cited_text: string;
  char_start: number | null;
  char_end: number | null;
  fields: string[];
}

export interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  is_error: boolean;
  result_summary: string;
}

export interface AskResponse {
  status: AskStatus;
  answer: string;
  citations: Citation[];
  tool_calls: ToolCall[];
  steps: number;
  model: string;
  prompt_version: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cost_usd: number;
  latency_ms: number;
  refusal_category: string | null;
  trace_id: string | null;
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return handle<AskResponse>(res);
}
