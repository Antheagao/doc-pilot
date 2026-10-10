# PRESENT — Current Batch

> Live task board for the current batch. The `/batch-loop` command reads this file, works the topmost unchecked task, checks it off, and loops. Completed batches move to [PREVIOUS.md](PREVIOUS.md); upcoming batches live in [FUTURE.md](FUTURE.md).

## (empty — PROJECT DONE 2026-08-03)

Batch 5 (ship signals: container images, full-stack compose, live smoke suite, README final pass — S1–S4) finished 2026-08-01 and moved to [PREVIOUS.md](PREVIOUS.md).

**Deploy is permanently cancelled** (Anthony no-hosting ruling 2026-08-03 — recorded in `projects upgrade v2.md`), which was the last blocked item: the queue goal (local demo GIF + verified eval table, no deploy) is met and the repo is marked ✅ DONE in the v2 queue. Stretch (pgvector RAG search, CSV export) is optional and no longer gated on anything.

**Carry-over notes (accepted, not blocking):**
- H3 known gap: well-formed `line_items` leaf with a non-list value → zero violations, silently stores `[]`.
- H6 retry amplification: retryable failure on page N of a chunked PDF re-bills pages 1..N-1 on the next attempt.
- No dependency pinning/lockfiles (S1 review): revisit before the hosted demo ships.
