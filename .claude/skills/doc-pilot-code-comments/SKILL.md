---
name: doc-pilot-code-comments
description: doc-pilot overlay for code-comments - docstring house style, exemplars, Alembic migration headers, where reasons live, and known gaps. Use when adding or editing a docstring, comment, or migration header, or explaining a non-obvious decision in doc-pilot.
---

# doc-pilot code-comments overlay

Load the global `code-comments` guide for the rules; this file holds what exists here.
Python docstrings stand in for the guide's rustdoc rules.

## House style

- Module docstrings carry the reasoning that spans a module; read them before changing one: `worker.py` (claim and commit split, reclaim safety), `budget.py` (soft cap), `ratelimit.py` (per-process limits), `telemetry.py` (trace shape, no content), `retrieval/search.py` (why RRF).
- Docstrings are prose paragraphs with no Google or NumPy sections: what it does first, then why.
- Exemplars that guard a fix: `_amounts_match` (`app/evals/scoring.py`, the float boundary bug), `MaxBodySizeMiddleware` (`main.py`, why a handler check is too late), and the `async` note above `limit_ask` (`ratelimit.py`).
- Every `Settings` field in `config.py` has a comment naming the module that reads it and why the default was chosen.
- Prose dashes in comments are written `--`.
- Frontend: a file-header `//` comment in `src/lib/api.ts`, `/** */` on helpers whose contract is not obvious (`DocumentPreview`), and JSX comments for render quirks (`{/* Keyed by src ... */}`).

## Migration headers

- The Alembic docstring's first line is the header (`index the timestamps the daily budget filters on`); the why sits in comments beside the operation, as in `e3f19a7c2d40` (the literal `EMBEDDING_DIM`) and `a4d8e2f6b901` (the generated-column swap and its re-index note).
- Every revision keeps the generated `"""Upgrade schema."""` and `"""Downgrade schema."""` stubs.

## Where reasons live

- Task ids (`H3`, `H6`, `T0.2`, `A1`) come from `PRESENT.md` and `FUTURE.md` and leave those boards once done, so state the rule beside the id.
- History belongs in the commit message and `PREVIOUS.md` or `UPDATES.md`, not in code.
- Docstrings cite README sections by name (`README "Retrieval eval"` in `config.py` and `retrieval/search.py`); keep the heading and the citation in step.
- No `TODO`, `FIXME`, or `XXX` markers exist; queue work in `FUTURE.md`.

## Known gaps

- The long dash character appears in older comments (`main.py`, `worker.py`, `routers/documents.py`, `app/evals/report.py`, and most frontend files); replace it with `--` when you touch the line.
- `routers/review.py` cites README "Key decisions", a section that no longer exists ("Why these tech choices" is the nearest).
- Some docstrings narrate history (`extraction.py` "replaces the T5 placeholder", `_ensure_model_priced` "the previous behavior"); trim it when you edit them.
- `0eb534561f35` says only "initial schema", and `d2a7b5c9e013` gives no why beyond its title.
- The root `.env.example` comment says only the worker receives `ANTHROPIC_API_KEY`; the api does too.
