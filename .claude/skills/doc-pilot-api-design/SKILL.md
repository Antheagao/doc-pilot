---
name: doc-pilot-api-design
description: doc-pilot's FastAPI routes, status codes, the {"detail"} error body, list limits, SSE streams, JSON conventions, and known gaps. Use when adding or changing an endpoint, status code, request or response model, or the matching types in frontend/src/lib/api.ts in doc-pilot.
---

# doc-pilot api-design overlay

Load the global `api-design` guide for the rules; this file holds what exists here.

## Routes (mounted in `backend/app/main.py`, no version prefix)

| Router | Routes |
| --- | --- |
| `documents.py` | `POST /documents` (multipart `file`), `GET /documents`, `GET /documents/{id}`, `GET /documents/{id}/file` |
| `chat.py` | `POST /documents/{id}/chat`, `POST /documents/{id}/chat/stream`, `GET /documents/{id}/chat/conversations`, `GET /documents/{id}/chat/conversations/{conversation_id}` |
| `review.py` | `GET /review/queue`, `GET /review/queue/count`, `POST /review/fields/{id}/resolve` |
| `search.py` | `GET /search` with `q`, `k`, and `mode` (`dense`, `lexical`, `hybrid`) |
| `ask.py` | `POST /ask`, `POST /ask/stream`, `GET /ask/runs`, `GET /ask/runs/{id}`, `POST /ask/runs/{id}/feedback` |
| `stats.py`, `monitoring.py` | `GET /stats`, `GET /monitoring/evals`, `GET /monitoring/daily` |

`GET /healthz` returns `{"status": "ok"}`; there is no auth, so no route returns `401` or `403`.

## Status codes today

- `201` for the upload; `400` for `action: "correct"` without `corrected_value`; `404` for missing rows; `409` for `Field already reviewed` and for chat on a document that is not `extracted`.
- `413` `File too large` (middleware and `_read_limited`); `415` for an unlisted type or mismatched magic bytes; `422` from FastAPI validation, with `detail` as a list.
- `429` from `budget.enforce_daily_budget` and `ratelimit._enforce`, both with `Retry-After`.
- `502` (`model request rejected`) and `503` (`model unavailable`, with `Retry-After` when known) from `ask.model_error`; `503` when `ANTHROPIC_API_KEY` is unset.
- Streams are `text/event-stream` frames `event: <type>` (`model_call`, `tool_start`, `tool_call`, `answer`, or `error` with `status` and `detail`); `parseSseFrames` in `frontend/src/lib/api.ts` reads them.

## Conventions in use

- Bodies are Pydantic models in `backend/app/schemas.py`; their TypeScript twins live in `frontend/src/lib/api.ts`.
- Closed sets are `Literal` in the schemas: search `mode`, `AskStatus`, review `action`, feedback `rating`.
- Timestamps are ISO 8601 strings from Pydantic `datetime`; costs are `cost_usd`, durations `latency_ms`.
- A field's `value` is a `{value, confidence}` leaf, except `line_items`, which stores the array; `corrected_value` is the plain value, and an explicit `null` is told apart from an omitted one through `model_fields_set`.
- Limits are `Query(ge=1, le=...)`: documents and review queue 50 (max 200), ask runs and conversations 20 (max 100), search `k` 5 (max 50), `/monitoring/daily` `days` 30 (max 366).

## Known gaps

- Errors are FastAPI's `{"detail": "<sentence>"}` with no stable `code`; `handle()` in `api.ts` raises `ApiError(status, message)`, so clients can branch only on status.
- Some `detail` strings embed upstream exception text (`model unavailable: {exc}`), and 404 wording varies in case (`Document not found`, `document not found`).
- Lists return bare arrays and page by `offset`.
- Money is floats; Batch A1 moves it to decimal strings and fixes the `Number()` parse in `src/app/review/page.tsx`.
- `status` is `str` in `DocumentListItem` and `DocumentDetail`, while `api.ts` declares the `DocumentStatus` union.
