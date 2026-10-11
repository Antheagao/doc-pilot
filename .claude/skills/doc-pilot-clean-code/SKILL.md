---
name: doc-pilot-clean-code
description: doc-pilot overlay for clean-code - Python and TypeScript house naming, product words, versioned prompts, lint gates, and known gaps. Use when writing or refactoring backend Python or frontend TypeScript, naming things, or fixing a ruff or eslint finding in doc-pilot.
---

# doc-pilot clean-code overlay

Load the global `clean-code` guide for the rules; this file holds what exists here.
The Rust subsections of the global guide do not apply; the backend is Python 3.12+.

## House conventions

- Product words, spelled the same in SQL, Python, JSON, and TypeScript: `document`, `job` (`kind`, `state`), `extraction`, `extracted_field`, leaf (`{value, confidence}`), `needs_review`, `review_action` (`approved`, `corrected`), `page`, `chunk`, `ask_run`, `conversation`, `citation`.
- Units in names are already the norm: `cost_usd`, `latency_ms`, `retry_after_seconds`, `daily_budget_usd`, `ask_rate_limit_per_minute`, `chunk_max_chars`.
- Python: `X | None`, typed signatures, module-private helpers prefixed `_`, constants in `UPPER_CASE`, frozen `@dataclass` for small value objects (`ChatScope`).
- Shared rules have one home: `records.field_value` (a correction wins), `records.api_field_value` / `normalize_correction` (the decimal-string view and correction validation), `review._PENDING` (what "pending" means), `HANDLERS` (job dispatch), `TOP_LEVEL_FIELDS` (the field set).
- Pure, injectable helpers are the model to copy: `_backoff_delay`, `reciprocal_rank_fusion`, `merge_page_tool_inputs`, `_schema_violations`, `SlidingWindowLimiter(clock=...)`.
- Prompts are files, never inline strings: `backend/prompts/<name>_v<N>.md`, with the version taken from the file stem (`PROMPT_VERSION = PROMPT_PATH.stem`); new wording is a new file.
- Frontend API helpers are verb-first in `src/lib/api.ts` (`listDocuments`, `getDocument`, `uploadDocument`, `searchDocuments`, `askQuestion`, `resolveReviewField`); its `as` casts sit at the fetch and SSE parse boundaries.
- Each `eslint-disable-next-line` has its reason in the comment above or after `--` (`page.tsx`, `documents/[id]/page.tsx`, `review/page.tsx`).

## Lint gates here

- `ruff check .` and `ruff check ../scripts/demos` from `backend/` (alembic, tests, scripts, and the demo scripts included); `[tool.ruff.lint]` only ignores `B008`, so ruff's default rule set applies and nothing checks formatting or line length.
- No type checker runs on the backend (no mypy or pyright).
- Frontend: `npx eslint src` in CI (`eslint-config-next` core-web-vitals and typescript); `tsconfig.json` is `strict`.

## Known gaps

- Shared model-call helpers are private names in `extraction.py` (`_build_client`, `_classify_api_error`, `_ensure_model_priced`) imported by `transcription.py`, `agent/loop.py`, `routers/ask.py`, and `app/evals/`; every model failure is an `ExtractionError`, whatever the caller.
- `cost_usd` columns are `Numeric` but typed `Mapped[float]`; document amounts are `Decimal` in code and decimal strings in JSON (`app/money.py`).
- Status sets are bare strings in Python (`Document.status`, `Job.state`, `Job.kind` constants), while TypeScript has the `DocumentStatus` union.
- Some older comments use the long dash character; write `--` or `-`.
