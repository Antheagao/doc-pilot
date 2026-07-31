// Fetch helpers + TypeScript types for the doc-pilot backend API.
// Base URL is configurable via NEXT_PUBLIC_API_URL (see .env.example),
// defaulting to the local backend dev server.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export type DocumentStatus = "uploaded" | "processing" | "extracted" | "failed";

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

export interface ExtractedField {
  field_name: string;
  value: ExtractedFieldValue;
  confidence: number;
  needs_review: boolean;
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

export async function uploadDocument(file: File): Promise<DocumentCreateResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_URL}/documents`, {
    method: "POST",
    body: formData,
  });
  return handle<DocumentCreateResponse>(res);
}
