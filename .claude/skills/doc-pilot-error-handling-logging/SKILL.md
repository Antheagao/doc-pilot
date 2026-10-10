---
name: doc-pilot-error-handling-logging
description: doc-pilot's HTTPException detail errors, ExtractionError retry classes, job failure and backoff handling, stdlib logging, OpenTelemetry spans, and the rule about expired ORM objects in the worker. Use when adding an error path, retry, log line, or span, or debugging a stuck or failed job in doc-pilot.
---

# doc-pilot error-handling-logging overlay

Load the global `error-handling-logging` guide for the rules; this file holds what exists here.

## API errors

- Handlers raise `fastapi.HTTPException(status_code, detail=...)`, so bodies are `{"detail": "<sentence>"}`, or `{"detail": [...]}` for 422 validation; there is no backend `ApiError` type and no error `code`.
- `MaxBodySizeMiddleware` builds its 413 as a `JSONResponse` with the same `detail` shape.
- `model_error` (`routers/ask.py`) maps `NonRetryableExtractionError` to 502 and other `ExtractionError` to 503 with `Retry-After`; a streamed run sends an `error` event instead and logs surprises with `logger.exception("streamed agent run failed")`.
- Frontend: `handle()` in `src/lib/api.ts` raises `ApiError(status, message)`, and pages show it in `.error-banner`.

## Model and job failures

- `app/extraction.py` defines `ExtractionError` (retryable; optional `retry_after_seconds`, plus `input_tokens`, `output_tokens`, `cost_usd` when the call was billed), `NonRetryableExtractionError`, and `ModelRefusalError`.
- `_classify_api_error` sorts SDK errors: `NON_RETRYABLE_STATUS_CODES` (400, 401, 403, 404, 413, 422) fail permanently, 408, 429, 529, and 5xx retry with any `retry-after`; transcription, the agent, and the judge reuse it.
- Two retry layers: the SDK's own (`anthropic_max_retries`, set in `_build_client`) inside one call, then the queue's (`worker.fail_job`: `MAX_ATTEMPTS` 3, `_backoff_delay` equal jitter capped at `BACKOFF_MAX` 60 s, a `retry-after` floor capped at `RETRY_AFTER_MAX` 900 s).
- `fail_job` keeps the last 2,000 characters of the traceback in `jobs.last_error`; a permanent failure marks the document `failed` or `refused` (extract) or fills `ask_runs.judge_error` (judge).
- A persistence error after a billed call is re-raised as `ExtractionError` carrying its tokens and cost, so the spend stays visible.
- `run_worker` catches every per-iteration exception, logs it, and keeps polling; `reclaim_orphaned_jobs` resets `processing` rows at startup.

## Logging and tracing

- stdlib `logging` with `logger = logging.getLogger(__name__)` and `%`-style arguments (`logger.info("claimed %s job %s (attempt %d)", ...)`); only `worker.py` calls `logging.basicConfig`, and the API runs on uvicorn's defaults.
- There are no request ids; correlation is OpenTelemetry (`app/telemetry.py`): a `job <kind>` span with `docpilot.job.*` attributes and outcome, a `model_call_span` per model call, and `jobs.traceparent` plus `ask_runs.trace_id` to join them.
- Jaeger runs under the compose `tracing` profile; Langfuse score failures are logged and dropped, never raised.

## Expired ORM objects in the worker

- All three handlers roll back before their model call, which expires the session's `Job`; a read of `job.id` afterwards raises `MissingGreenlet`, which once escaped to `run_worker`, lost the real failure, and left the job in `processing`.
- So `complete_job` and `fail_job` take `job_id` and re-fetch, and `_run_handler` logs the carried `job_id`; after any rollback, never touch ORM attributes of an object loaded before it.
- `tests/test_worker.py` covers handlers that roll back and then succeed, fail, fail for good, or are cancelled.
- The worker touches a heartbeat file (`WORKER_HEARTBEAT_FILE`) every 10 s from a task beside the poll loop; compose's worker healthcheck fails when it is over 60 s old.
