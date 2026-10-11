---
name: doc-pilot-design-principles
description: doc-pilot overlay for design-principles - ports and seams in use, FastAPI dependency injection, transaction patterns, gaps not to copy, and planned modules. Use when adding a module, provider, job kind, or interface, or shaping a type, dependency, or transaction in doc-pilot.
---

# doc-pilot design-principles overlay

Load the global `design-principles` guide for the rules; this file holds what exists here.
The Rust and axum subsections of the global guide do not apply; the backend is FastAPI with async SQLAlchemy.

## Patterns already in the codebase

| Pattern | Where |
| --- | --- |
| Port with two adapters and a factory: `Embedder` (`Protocol`), `FastEmbedEmbedder`, `HashingEmbedder`, `build_embedder(settings)` | `app/retrieval/embeddings.py` |
| Kind-keyed dispatch `HANDLERS` plus `dispatch_job`; an unknown kind fails permanently | `app/worker.py` |
| Injected callable `ExtractFn`, with `build_mock_extract_fn` as the offline stand-in | `app/evals/runner.py` |
| FastAPI dependencies as seams: `get_session`, `get_session_factory`, `get_settings`, `get_embedder`, `get_anthropic_client` | `app/db.py`, `app/config.py`, `app/retrieval/embeddings.py`, `app/routers/ask.py` |
| One Anthropic client per API process, built by `_build_client` | `routers/ask.py`, `extraction.py` |
| Outbox: the extract job adds its `index` job in the same transaction | `extraction.process_document_job` |
| No transaction across a model call: copy ids, `rollback()`, call, re-fetch | all three job handlers |
| Model output parsed once at the edge: `_coerce_leaf`, `_schema_violations`, one repair reprompt | `app/extraction.py` |
| One rule, one home: `records.field_value`, `review._PENDING` | `app/records.py`, `routers/review.py` |
| Code computes the numbers the agent quotes: `query_extractions` returns counts and sums from code | `app/agent/tools.py` |
| Money is `Decimal`, parsed at every edge by `money.to_decimal` and printed to the ISO 4217 minor unit; never `float` arithmetic or `round()` | `app/money.py` |
| Results tied to `(model, prompt_version, dataset_version)` | `backend/prompts/`, `app/evals/` |

## Known gaps: do not copy these

- Routers query the ORM directly; there is no feature layer between `routers/` and `models.py`, and extraction persistence lives inside `process_document_job` (Batch 0 task T0.2 extracts `build_extracted_fields`).
- Shared model-call code is private to `extraction.py`, and its error types are named for extraction.
- Closed sets (`Document.status`, `Job.state`, `Job.kind`) are strings with module constants, not `Literal` or `Enum`.
- `complete_job` and `fail_job` write unconditionally instead of compare-and-set; that holds only while exactly one worker runs.
- Rate limits live in process memory, which assumes one API process.

## Planned modules (`FUTURE.md`)

- `app/reconcile.py` (pure: record in, `Check` list out, `RULES_VERSION`), built on `app/money.py`.
- `app/grounding.py` (evidence quote to page boxes) behind an `OcrProvider` interface selected by `OCR_PROVIDER` (PyMuPDF text layer, model boxes, optional Tesseract).
- `/reports/totals` aggregation that the agent's structured-data tool also calls (D5), so the two can never disagree.
- Decisions are recorded in README "Why these tech choices" and on the boards; there is no `docs/decisions/` folder.
