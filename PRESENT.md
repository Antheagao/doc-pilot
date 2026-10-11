# PRESENT - Current Batch

> Live task board for the current batch. The `/batch-loop` command reads this file, works the topmost unchecked task, checks it off, and loops. Completed batches move to [PREVIOUS.md](PREVIOUS.md); upcoming batches live in [FUTURE.md](FUTURE.md).

## (empty - Batch 0 complete 2026-10-11; Batch A is next in FUTURE.md)

## Follow-ups from Batch 0 (not yet scheduled)

- Answers can contain Markdown (`**Date:**`) that the UI renders as raw asterisks: render a safe Markdown subset or tell the prompts to answer in plain text.
- The top `screenshots/demo.gif` predates the Search, Ask, and Dashboard navigation; re-recording it with the harness costs one live extraction (about $0.02).
- The Screenshots section still shows stills that the Features GIFs now cover (`search-hybrid.png`, `search-amount.png`, `review-correct.png`).
- The macOS session's four commits carry `Co-Authored-By` trailers, against the global no-attribution rule; that machine's claude-brain setup needs bootstrapping.

## Carry-over (accepted, not blocking)

- H3: a well-formed `line_items` leaf with a non-list value silently stores `[]` (resolved by Batch A2).
- H6 retry amplification: a retryable failure on page N of a chunked PDF re-bills pages 1..N-1 on the next attempt.
- No dependency pinning or lockfiles.
