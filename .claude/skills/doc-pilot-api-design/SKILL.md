---
name: doc-pilot-api-design
description: doc-pilot's FastAPI routes, status codes, the {"detail"} error body, list limits, SSE streams, JSON conventions, and known gaps. Use when adding or changing an endpoint, status code, request or response model, or the matching types in frontend/src/lib/api.ts in doc-pilot.
---

# doc-pilot api-design overlay

Load the global `api-design` guide for the rules; this file holds what exists here.

## Routes (mounted in `backend/app/main.py`, no version prefix)

| Router | Routes |
| --- | --- |
| `auth.py` (`/auth`) | `POST /auth/login` (200 `MeOut`, sets the cookie), `POST /auth/logout` (204), `GET /auth/me`, `POST /auth/organization` (switch; 404 for an organization the user is not in) |
| `api_keys.py` (`/api-keys`) | `GET /api-keys` (`{"api_keys": [...]}`), `POST /api-keys` (201 `{api_key, secret}`), `DELETE /api-keys/{id}` (204) |
| `documents.py` | `POST /documents` (multipart `file`), `GET /documents`, `GET /documents/{id}`, `GET /documents/{id}/file` |
| `chat.py` | `POST /documents/{id}/chat`, `POST /documents/{id}/chat/stream`, `GET /documents/{id}/chat/conversations`, `GET /documents/{id}/chat/conversations/{conversation_id}` |
| `review.py` | `GET /review/queue`, `GET /review/queue/count`, `POST /review/fields/{id}/resolve` |
| `search.py` | `GET /search` with `q`, `k`, and `mode` (`dense`, `lexical`, `hybrid`) |
| `ask.py` | `POST /ask`, `POST /ask/stream`, `GET /ask/runs`, `GET /ask/runs/{id}`, `POST /ask/runs/{id}/feedback` |
| `stats.py`, `monitoring.py` | `GET /stats`, `GET /monitoring/evals`, `GET /monitoring/daily` |

`GET /healthz` returns `{"status": "ok"}`.
Every other route needs a credential: a session cookie, or `Authorization: Bearer dpk_...` for an API key; only `/auth/login`, `/auth/logout`, and the docs (`/docs`, `/redoc`, `/openapi.json`) are open besides.
`tests/test_route_auth.py` fails any route that answers without one.

- `401` is `{"detail": "not authenticated"}` with `WWW-Authenticate: Bearer realm="doc-pilot"` for every way a credential can fail, so a response never says which; a failed sign-in is `401` `{"detail": "email or password is incorrect"}`.
- `403` is `origin not allowed` for a cookie-authenticated write (and login or logout) whose `Origin` is not in `CORS_ORIGINS`; `an API key cannot manage API keys` or `... sessions` for a key on a session-only route (`/auth/me`, `/auth/organization`, every `/api-keys` route); `only an owner can manage API keys` for a member on `POST` or `DELETE /api-keys`; and `/mcp` refusing an Origin header.
- `409` past 25 active keys; `422` for an API key name that is not 1 to 100 characters without control characters.
- `/auth/login` answers `429` before it hashes anything, and its `422` never echoes the input.

`/mcp` and `/mcp/` are plain Starlette `Route`s appended in `main.py` (an `McpHttpEndpoint`, not routers), so neither is in `/openapi.json`; a `Mount` would answer every request with a `307` that MCP clients do not follow on a POST.
They take `POST` only, speak JSON-RPC (MCP's Streamable HTTP) in stateless JSON mode, and answer `421` for a Host other than loopback, `403` for any Origin, `405` with `Allow: POST` for any other method, `401` with the `WWW-Authenticate` challenge for a missing, malformed, unknown, or revoked key, `429` from the `mcp` rate-limit bucket (per key), `413` for a body over 64 KiB, `400` for a non-JSON content type, and `503` outside the API's lifespan, checked in about that order.
Their own errors are the API's `{"detail": ...}` body, except the SDK's JSON-RPC errors for a malformed message.

## Status codes today

- `201` for the upload; `400` for `action: "correct"` without `corrected_value`; `404` for missing rows; `409` for `Field already reviewed` and for chat on a document that is not `extracted`.
- `413` `File too large` (middleware and `_read_limited`); `415` for an unlisted type or mismatched magic bytes; `422` from FastAPI validation, with `detail` as a list.
- `422` with `detail` as one sentence from `POST /review/fields/{id}/resolve` when a correction is not valid for the field (`MoneyError` text from `records.normalize_correction`: too many decimal places for the currency, not a plain number, a line item or tax line without exactly its keys, a tax rate outside 0 to 100 or with more than 6 decimal places, a `payment_card_last4` that is not exactly four digits, a `payment_method` that is not text or holds a card-number-like digit run, a bad currency code); the field stays pending.
- `429` from `budget.enforce_daily_budget` (the organization's cap, then the installation's, which prints no figure) and `ratelimit.enforce_rate_limit`, both with `Retry-After`; `/mcp` calls `enforce_rate_limit` itself and returns the same body.
- `502` (`model request rejected`) and `503` (`model unavailable`, with `Retry-After` when known) from `ask.model_error`; `503` when `ANTHROPIC_API_KEY` is unset.
- Streams are `text/event-stream` frames `event: <type>` (`model_call`, `tool_start`, `tool_call`, `answer`, or `error` with `status` and `detail`); `parseSseFrames` in `frontend/src/lib/api.ts` reads them.

## Conventions in use

- Bodies are Pydantic models in `backend/app/schemas.py`; their TypeScript twins live in `frontend/src/lib/api.ts`.
- Closed sets are `Literal` in the schemas: search `mode`, `AskStatus`, review `action`, feedback `rating`.
- Timestamps are ISO 8601 strings from Pydantic `datetime`; costs are `cost_usd`, durations `latency_ms`.
- A field's `value` is a `{value, confidence}` leaf (v1) or a `{value, confidence, evidence}` leaf (schema v2), except the array fields (`line_items`, and `tax_lines` in v2), which store the array with a leaf per cell; `corrected_value` is the plain value, and an explicit `null` is told apart from an omitted one through `model_fields_set`.
- Evidence rides inside the leaves, with no endpoint of its own: `evidence` is `{page, quote}` (an integer page from 1 to 10,000, a stripped quote of at most 200 characters) or `null`, and a model answer that does not fit is stored as `null` (`extraction._coerce_evidence`).
  The API passes evidence through untouched; only `leaf["value"]` is converted.
- Every number read from a document is a canonical decimal string (`DecimalString` in `api.ts`): the money fields of the extraction's schema (v1: subtotal, tax, total; v2 adds discount, shipping, tip, amount_paid, amount_due), each line item's quantity, unit_price, and total, and each tax line's rate, base, and amount, in `GET /documents/{id}`, the review queue, and the resolve response (`records.field_to_out` / `api_field_value`, which take the schema of the stored extraction through `records.schema_for_stored(prompt_version)` (an unregistered version reads as v1 with a warning, so one odd row never fails an endpoint) and change only `leaf["value"]`).
  Money is padded to the document currency's minor unit (`"46.20"`, JPY `"1500"`) with extra digits kept (`"46.255"`); quantity and tax rate have no trailing zeros (`"2.5"`, a rate is the printed percent, `"8.875"` for 8.875%); null stays null; a stored value that is not a number is passed through as-is.
  Machine metrics (`confidence`, `cost_usd`, tokens, latency) stay JSON numbers.
- A correction may send an amount as a string or a number; it is validated against the currency (`parse_amount_input`) and stored as the canonical string, so older numeric corrections and new string ones both serve as strings.
  `currency` corrections are three letters, upper-cased, not checked against the ISO table; `tax_lines` takes null or rows of exactly `label`, `rate`, `base`, `amount`; `payment_card_last4` takes null or exactly four ASCII digits and `payment_method` must be text without a card-number-like digit run (the card-number rule, see the security overlay); `vendor`, `document_date`, `due_date`, and the other text fields are stored as sent.
- `ReviewQueueItem.currency` is the document's currency as stated (a correction wins), trimmed and upper-cased, null when unstated; a non-ISO code passes through, and the queue's `value` is already padded to its digits.
- `ExtractedFieldOut.review_reason` and `ReviewQueueItem.review_reason` are `confidence`, `reconciliation:<check id>`, or `quote:<path>` (the number in the value's evidence quote is not the value; the check's `kind` is `quote`).
  `ExtractionOut.reconciliation` is the stored report validated by `ReconciliationOut` (null for an extraction nobody has reconciled), and a queue item adds the extraction's failing `checks` that involve its field and the `suggestion` when it targets that field; resolving a field recomputes the report in the same transaction.
- A `422` sentence names the currency (`JPY allows no decimal places`, or `no currency is stated, so amounts allow at most 2 decimal places`), echoes a rejected value cut to 80 characters, and line-item quantities are limited to 6 decimal places and a magnitude below 10^12 (negatives allowed), and a tax rate to 6 decimal places and 0 to 100.
- Every response but `/healthz` and the docs is `Cache-Control: private, no-store` (`SecurityHeadersMiddleware`); the sign-in, sign-out, `/auth/me`, and key-creation responses set `no-store` themselves.
- A new list endpoint is wrapped in an object (`/api-keys` returns `{"api_keys": [...]}`); the older lists are bare arrays.
- A response about the caller's own tenant data needs no organization in its path or body: the organization comes from the credential, and a foreign id is a `404`, never a `403`.
- Limits are `Query(ge=1, le=...)`: documents and review queue 50 (max 200), ask runs and conversations 20 (max 100), search `k` 5 (max 50), `/monitoring/daily` `days` 30 (max 366).

## Known gaps

- Errors are FastAPI's `{"detail": "<sentence>"}` with no stable `code`; `handle()` in `api.ts` raises `ApiError(status, message)`, so clients can branch only on status.
- Some `detail` strings embed upstream exception text (`model unavailable: {exc}`), and 404 wording varies in case (`Document not found`, `document not found`).
- Lists return bare arrays (except `/api-keys`) and page by `offset`.
- `cost_usd` is still a float; the search, ask, and chat citations carry document text, not parsed amounts, and `/stats` and `/monitoring` carry machine metrics only.
- `status` is `str` in `DocumentListItem` and `DocumentDetail`, while `api.ts` declares the `DocumentStatus` union.
