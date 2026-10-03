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
  // Extraction + transcription per processed document.
  mean_pipeline_cost_per_doc: number | null;
  spend: StatsSpend;
  budget: StatsBudget;
  ask: StatsAsk;
}

// The daily spend cap; daily_budget_usd is null when it's off.
export interface StatsBudget {
  daily_budget_usd: number | null;
  spent_today_usd: number;
  resets_in_seconds: number;
}

// Every model call the system has paid for, by stage.
export interface StatsSpend {
  extraction_usd: number;
  transcription_usd: number;
  agent_usd: number;
  judge_usd: number;
  total_usd: number;
}

export interface StatsAsk {
  runs: number;
  answered: number;
  mean_cost_usd: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  feedback_up: number;
  feedback_down: number;
  judge_sampled: number;
  judged: number;
  judge_grounded_rate: number | null;
  judge_answers_rate: number | null;
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

export type Feedback = "up" | "down";

// The background groundedness grader's verdict on a stored answer.
// grounded/answers_question are null when no verdict could be read.
export interface AskJudgment {
  grounded: boolean | null;
  answers_question: boolean | null;
  unsupported_claims: string[];
  explanation: string | null;
  model: string | null;
  prompt_version: string | null;
  cost_usd: number;
  error: string | null;
  judged_at: string;
}

// One /ask run (or chat turn), as answered and as stored.
export interface AskResponse {
  id: string;
  question: string;
  created_at: string;
  // Set for a turn of a per-document chat; null for an /ask question.
  document_id: string | null;
  conversation_id: string | null;
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
  feedback: Feedback | null;
  feedback_note: string | null;
  // A grading job was queued; `judgment` stays null until it finishes.
  judge_sampled: boolean;
  judgment: AskJudgment | null;
}

export interface AskRunSummary {
  id: string;
  question: string;
  status: AskStatus;
  created_at: string;
  document_id: string | null;
  cost_usd: number;
  latency_ms: number;
  feedback: Feedback | null;
  judge_sampled: boolean;
  judge_grounded: boolean | null;
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return handle<AskResponse>(res);
}

// POST /ask/stream: the run's progress as server-sent events, then the
// stored run (`answer`) or an `error`.
export type AskEvent =
  | { type: "model_call"; step: number; cost_usd: number; stop_reason: string | null }
  | { type: "tool_start"; name: string; input: Record<string, unknown> }
  | ({ type: "tool_call" } & ToolCall)
  | { type: "answer"; run: AskResponse }
  | { type: "error"; status: number; detail: string };

/** Split the complete SSE frames off the front of `buffer`; `rest` is
 * the incomplete tail to prepend to the next chunk. */
export function parseSseFrames(buffer: string): { events: AskEvent[]; rest: string } {
  const events: AskEvent[] = [];
  let rest = buffer;
  let boundary = rest.indexOf("\n\n");
  while (boundary !== -1) {
    const frame = rest.slice(0, boundary);
    rest = rest.slice(boundary + 2);
    const data = frame
      .split("\n")
      .filter((line) => line.startsWith("data: "))
      .map((line) => line.slice("data: ".length))
      .join("\n");
    if (data) events.push(JSON.parse(data) as AskEvent);
    boundary = rest.indexOf("\n\n");
  }
  return { events, rest };
}

/** Ask with live progress. Refusals that happen before the stream starts
 * (no API key, the daily budget, validation) throw ApiError like any other
 * call; failures after it starts arrive as an `error` event. A stream that
 * ends with neither (a dropped connection, a proxy timeout) throws
 * StreamInterruptedError -- the run itself keeps going on the server and
 * is stored, so it shows up under recent questions. */
export async function askQuestionStream(
  question: string,
  onEvent: (event: AskEvent) => void
): Promise<void> {
  await streamAgentRun(`${API_URL}/ask/stream`, { question }, onEvent);
}

async function streamAgentRun(
  url: string,
  body: Record<string, unknown>,
  onEvent: (event: AskEvent) => void
): Promise<void> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) await handle<never>(res);
  if (!res.body) throw new ApiError(res.status, "the response had no body to stream");
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let finished = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const parsed = parseSseFrames(buffer + value);
    buffer = parsed.rest;
    for (const event of parsed.events) {
      if (event.type === "answer" || event.type === "error") finished = true;
      onEvent(event);
    }
  }
  if (!finished) throw new StreamInterruptedError();
}

export class StreamInterruptedError extends Error {
  constructor() {
    super("The connection closed before the answer arrived.");
    this.name = "StreamInterruptedError";
  }
}

export async function listAskRuns(limit = 10): Promise<AskRunSummary[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  const res = await fetch(`${API_URL}/ask/runs?${params}`, { cache: "no-store" });
  return handle<AskRunSummary[]>(res);
}

export async function getAskRun(id: string): Promise<AskResponse> {
  const res = await fetch(`${API_URL}/ask/runs/${encodeURIComponent(id)}`, { cache: "no-store" });
  return handle<AskResponse>(res);
}

export async function sendAskFeedback(id: string, rating: Feedback): Promise<AskResponse> {
  const res = await fetch(`${API_URL}/ask/runs/${encodeURIComponent(id)}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rating }),
  });
  return handle<AskResponse>(res);
}

// ---- Per-document chat: POST /documents/{id}/chat ---------------------------

export interface ConversationSummary {
  conversation_id: string;
  first_question: string;
  turns: number;
  started_at: string;
  last_turn_at: string;
}

/** One chat turn, streamed: the same events as askQuestionStream. Pass
 * null to start a conversation; the `answer` event's run carries the
 * conversation_id to send with follow-ups. */
export async function chatStream(
  documentId: string,
  question: string,
  conversationId: string | null,
  onEvent: (event: AskEvent) => void
): Promise<void> {
  await streamAgentRun(
    `${API_URL}/documents/${encodeURIComponent(documentId)}/chat/stream`,
    { question, conversation_id: conversationId },
    onEvent
  );
}

export async function listConversations(
  documentId: string,
  limit = 20
): Promise<ConversationSummary[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  const res = await fetch(
    `${API_URL}/documents/${encodeURIComponent(documentId)}/chat/conversations?${params}`,
    { cache: "no-store" }
  );
  return handle<ConversationSummary[]>(res);
}

export async function getConversation(
  documentId: string,
  conversationId: string
): Promise<AskResponse[]> {
  const res = await fetch(
    `${API_URL}/documents/${encodeURIComponent(documentId)}/chat/conversations/${encodeURIComponent(conversationId)}`,
    { cache: "no-store" }
  );
  return handle<AskResponse[]>(res);
}

// ---- Monitoring dashboard: GET /monitoring/* --------------------------------

export type EvalSuite = "extraction" | "agent" | "retrieval";

// One committed eval run, normalized across suites. `accuracy` is the
// suite's headline metric, named by `accuracy_metric`; cost and latency
// are per document / question / query.
export interface EvalRun {
  suite: EvalSuite;
  started_at: string;
  series: string;
  detail: string;
  n: number;
  accuracy: number | null;
  accuracy_metric: string;
  mean_cost_usd: number | null;
  total_cost_usd: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  extra: Record<string, number | null>;
}

export type EvalHistory = Record<EvalSuite, EvalRun[]>;

export async function getEvalHistory(): Promise<EvalHistory> {
  const res = await fetch(`${API_URL}/monitoring/evals`, { cache: "no-store" });
  return handle<EvalHistory>(res);
}

// Live traffic for one UTC day.
export interface DailyPoint {
  date: string;
  spend: { documents_usd: number; answers_usd: number; grading_usd: number; total_usd: number };
  answers: {
    runs: number;
    answered: number;
    judged: number;
    grounded_rate: number | null;
    feedback_up: number;
    feedback_down: number;
    mean_cost_usd: number | null;
    latency_p50_ms: number | null;
    latency_p95_ms: number | null;
  };
  extractions: {
    documents: number;
    mean_cost_usd: number | null;
    latency_p50_ms: number | null;
    latency_p95_ms: number | null;
  };
}

export async function getDailyMetrics(days: number): Promise<DailyPoint[]> {
  const params = new URLSearchParams({ days: String(days) });
  const res = await fetch(`${API_URL}/monitoring/daily?${params}`, { cache: "no-store" });
  return handle<DailyPoint[]>(res);
}
