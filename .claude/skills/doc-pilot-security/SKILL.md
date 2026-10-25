---
name: doc-pilot-security
description: doc-pilot's security facts (no auth by design, upload and file-serving checks, CORS, spend caps, prompt-injection boundaries, secrets) and known gaps. Use when touching uploads, served files, user input, prompts or model output, CORS, env secrets, or dependencies in doc-pilot.
---

# doc-pilot security overlay

Load the global `security` guide for the rules; this file holds what exists here.
The auth, session, cookie, and CSRF sections of the global guide have nothing to apply to yet.

## Threat model

- Single-user and local-only: no authentication, and every compose port binds `127.0.0.1` (db 5434, api 8000, frontend 3000, Jaeger, demo api 8100 and frontend 3100).
- The GitHub repository `Antheagao/doc-pilot` is public: anything committed is published.
- The README "Security model" section is the narrative; keep it true when a guard changes.

## What exists today

- Uploads (`routers/documents.py`): the `ALLOWED_MIME_TYPES` allow-list, the `_content_matches_mime` magic-byte check (415 on mismatch), and a stored name of `<uuid><ext>` from `_safe_ext`, never the client filename.
- `MaxBodySizeMiddleware` (`main.py`) aborts bodies over `MAX_UPLOAD_SIZE` (20 MB) before multipart spooling; `_read_limited` is the backstop.
- `GET /documents/{id}/file` serves the stored type, and `SecurityHeadersMiddleware` adds `X-Content-Type-Options: nosniff` to every response.
- CORS: the `CORS_ORIGINS` allow-list with `allow_credentials` off on purpose (`main.py`).
- Spend: `DAILY_BUDGET_USD` (`budget.py`), per-client sliding windows (`ratelimit.py`), and per-question `AGENT_MAX_STEPS` and `AGENT_MAX_COST_USD`; uploads and `/ask` plus chat are the only billed entry points.
- PDFs over `pdf_max_pages` (20) are refused before any model call.
- Prompt injection: `extract_v1.md`, `extract_v2.md`, `transcribe_v1.md`, and `agent_v1.md` tell the model document text is data, never instructions; extraction runs under forced tool use with a fixed schema; `evals/docs/025-adversarial-office-order-injection.png` is scored in the extraction and agent evals.
- Card numbers, enforced in code and not left to the prompt (`app/pii.py`, `extraction.mask_card_data`): schema v2 stores `payment_method` (brand or method only) and `payment_card_last4` (exactly four digits), never a longer number.
  `mask_card_data` runs in `_extract_one_block` on the first answer and on any accepted repair answer, before `raw_response`, `_chunks`, eval artifacts, or logs are built from them, and again at the top of `build_extracted_fields`, so the demo seed and replays store the same masked data.
  A digit run is a stretch of digits that spaces, dashes, and dots may separate; it is a card or account number when it holds 13 digits in all or five in a row, so dates, amounts, and phone numbers stay readable.
  The payment leaves are rebuilt from allowed keys only (`value`, `confidence`, `evidence`): `payment_card_last4` is cut to its last four digits (null when it has fewer), `payment_method` keeps text with long runs masked to `****` plus the last four, a value of any other shape (a list, an object) is null, and extra keys are dropped.
  Every evidence quote of every field is swept with the Luhn-checked card rule (13 to 19 digits that pass the check), because the line that shows the amount paid often shows the card; values are never swept, so a sixteen-digit `invoice_number` stays intact.
  A reviewer's correction gets the same rule as a `422` (`records.normalize_correction`): `payment_method` must be text or null without a long digit run, and `payment_card_last4` exactly four digits.
  Card numbers in transcribed page text (`transcribe_v1`) are masked when the agent tools read it, not when it is stored: `_line_blocks` (`agent/tools.py`) runs `pii.mask_card_numbers` on each line that `search_documents` and `get_page` return, so a Luhn-valid 13 to 19 digit number becomes `****` plus its last four digits in what the model is sent, in `Source.lines`, and so in the `cited_text` of an `/ask` citation.
  The same function sweeps evidence quotes at write time, so its rule is shared: it finds a card inside a longer run (a card followed by an expiry, an auth code, or another card), never counts a date or a `NN.NN` amount as part of one, and leaves a number that fails the check alone (an order reference).
  Where stretches of a run overlap, one grouped like a card (4-4-4-4, 4-4-4-4-3, 4-6-5, 4-6-4, 6-13, or unbroken) wins, then the longer, then the leftmost.
  Known limits: a chance Luhn match that takes in digits beside a card can show the wrong last four digits, and a grouped reference number of 17 to 23 digits can be partly masked when a stretch of it passes the check (about 1 in 6 for 20-digit references in groups of four); a future gold quote holding one would be rejected by `evals/dataset.py` until it is masked.
  Character spans still index the stored line, so a masked line may be shorter than its span.
  The stored page text and the `/search` snippets still hold the full number, so masking it at write time stays open (Batch G4).
- Evidence is model output too: `_coerce_evidence` keeps `{page, quote}` only when the page is an integer from 1 to 10,000 and the quote is non-empty text of at most 200 characters, and stores `null` otherwise; the quote is displayed and compared (`app/quotes.py`) but never executed or trusted as the value.
- Model output is untrusted input: `_coerce_leaf` and `_schema_violations` (`extraction.py`) assume `tool_use.input` is adversarial, and the frontend renders values as React text with no `dangerouslySetInnerHTML`.
- Agent citations come from `search_result` blocks (`agent/tools.py`), so cited text is copied from stored pages (with card numbers masked, above), not written by the model.
- MCP (`app/mcp_server/`, README "MCP server"): the same read-only tools as `/ask` for any MCP client, with no model call and no spend.
  Document text reaches the client's model, so a result opens with a fixed untrusted-data notice and holds every document-derived string (page text, extracted fields, and the filename in a title) between `BEGIN DOCUMENT DATA <code>` and `END DOCUMENT DATA <code>` lines; the code is `secrets.token_hex(8)` per call and is replaced if it appears in the data.
  Arguments are checked against the tool's input schema in code (the SDK's low-level server does not) and strings over 500 characters are refused; a result is cut at `MCP_MAX_RESULT_CHARS` with `truncated` set; card numbers in page text are masked (above).
  `/mcp` is stateless and POST-only, serves only loopback Host values (`421` otherwise) and no Origin (`403`, the frontend included), checks both before the `mcp` rate-limit bucket, and refuses a body over 64 KiB; it has no authentication, so its threat model is `/search`'s.
  Deferred until auth exists: API keys or bearer tokens, a document scope and rate-limit bucket per key (`ToolContext.document_ids` is the seam), non-loopback hosts and TLS, browser origins, an audit log of tool calls, and MCP resources.
- Spans and Langfuse never carry prompts, documents, questions, or answers (`telemetry.py`, `langfuse_link.py`).
- Secrets: the Anthropic key lives in gitignored `backend/.env`, mirrored to the gitignored root `.env` for compose; `docker compose config` prints it, so never paste that output.

## Known gaps

- No CSP, `Referrer-Policy`, or frame protection on the API or the Next.js app; `nosniff` is the only security header.
- `ratelimit._client_key` uses `request.client.host`; behind a proxy it needs uvicorn `--proxy-headers` with `--forwarded-allow-ips`.
- Backend dependencies are unpinned ranges in `pyproject.toml` with no lockfile, and CI runs neither `pip-audit` nor `npm audit`.
- Uploads have no quota or cleanup, and nothing deletes a document's file, pages, and chunks (Batch G4).
- Some error `detail` strings echo upstream exception text (`model_error` in `routers/ask.py`).
- `/mcp` is as open as `/search` to anything on the machine that can reach loopback (another local user or process); the Host and Origin checks stop a browser tab and DNS rebinding, not a local program.
