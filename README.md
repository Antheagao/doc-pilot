# doc-pilot

AI document intelligence: upload messy real-world documents (receipts, invoices, IDs, forms) → a vision-language model extracts structured data → low-confidence fields route to a human review queue → clean data lands in Postgres with a full audit trail and per-document cost tracking.

## Planned architecture

- **Backend:** FastAPI (Python)
- **VLM:** Claude via the Anthropic API — image input with structured outputs (JSON schema), never regex-parsing free text
- **Queue:** Postgres `SKIP LOCKED` job queue
- **Frontend:** Next.js — upload, extraction results side-by-side with the document image, review/correct UI

## Core principles

- Evals from day one: labeled document set in `evals/`, field-level accuracy scored per model/prompt version
- Human-in-the-loop: low-confidence fields go to a review queue; corrections become new eval cases
- Cost & latency tracked per document
- Prompts and schemas versioned as code

*Status: pre-build. Scaffold coming next.*
