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
- Prompt injection: `extract_v1.md`, `transcribe_v1.md`, and `agent_v1.md` tell the model document text is data, never instructions; extraction runs under forced tool use with a fixed schema; `evals/docs/025-adversarial-office-order-injection.png` is scored in the extraction and agent evals.
- Model output is untrusted input: `_coerce_leaf` and `_schema_violations` (`extraction.py`) assume `tool_use.input` is adversarial, and the frontend renders values as React text with no `dangerouslySetInnerHTML`.
- Agent citations come from `search_result` blocks (`agent/tools.py`), so cited text is copied from stored pages, not written by the model.
- Spans and Langfuse never carry prompts, documents, questions, or answers (`telemetry.py`, `langfuse_link.py`).
- Secrets: the Anthropic key lives in gitignored `backend/.env`, mirrored to the gitignored root `.env` for compose; `docker compose config` prints it, so never paste that output.

## Known gaps

- No CSP, `Referrer-Policy`, or frame protection on the API or the Next.js app; `nosniff` is the only security header.
- `ratelimit._client_key` uses `request.client.host`; behind a proxy it needs uvicorn `--proxy-headers` with `--forwarded-allow-ips`.
- Backend dependencies are unpinned ranges in `pyproject.toml` with no lockfile, and CI runs neither `pip-audit` nor `npm audit`.
- Uploads have no quota or cleanup, and nothing deletes a document's file, pages, and chunks (Batch G4).
- Some error `detail` strings echo upstream exception text (`model_error` in `routers/ask.py`).
