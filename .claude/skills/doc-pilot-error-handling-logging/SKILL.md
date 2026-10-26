---
name: doc-pilot-error-handling-logging
description: doc-pilot's HTTPException detail errors, ExtractionError retry classes, job failure and backoff handling, stdlib logging, OpenTelemetry spans, and the rule about expired ORM objects in the worker. Use when adding an error path, retry, log line, or span, or debugging a stuck or failed job in doc-pilot.
---

# doc-pilot error-handling-logging overlay

Load the global `error-handling-logging` guide for the rules; this file holds what exists here.

## API errors

- Handlers raise `fastapi.HTTPException(status_code, detail=...)`, so bodies are `{"detail": "<sentence>"}`, or `{"detail": [...]}` for 422 validation; there is no backend `ApiError` type and no error `code`.
- `MaxBodySizeMiddleware` builds its 413 as a `JSONResponse` with the same `detail` shape.
- Auth failures are one body on purpose: `principal.unauthenticated()` is `401` `{"detail": "not authenticated"}` with `WWW-Authenticate: Bearer realm="doc-pilot"` for a missing, unknown, expired, or revoked credential, a disabled user, and a removed membership, so a response never says which; `/mcp` sends the same body.
  A failed sign-in is the one message `email or password is incorrect` (an unknown email, a disabled user, a wrong password, and an account with no membership alike); the last also logs a warning with the user id for the operator.
- The `RequestValidationError` handler in `main.py` is scoped to `/auth/login` and drops each error's `input` and `ctx`, because Pydantic puts the failing input in the error and that would echo a password; every other route keeps FastAPI's own 422.
- `TenantScopeError` (`app/tenancy.py`) is a programming error: a session touched tenant data with no organization bound, or a statement the filter cannot scope.
  It is raised before any SQL is sent, is not mapped to a status (the request is a 500), and must be fixed by binding the session, not caught.
  `TenantIsolationError` (`app/db.py`) is the startup refusal of `verify_tenant_isolation`: the process exits instead of serving with row-level security off, and its message names the role and what to set.
- `model_error` (`routers/ask.py`) maps `NonRetryableExtractionError` to 502 and other `ExtractionError` to 503 with `Retry-After`; a streamed run sends an `error` event instead and logs surprises with `logger.exception("streamed agent run failed")`.
- Frontend: `handle()` in `src/lib/api.ts` raises `ApiError(status, message)`, and pages show it in `.error-banner`.

## Model and job failures

- `app/extraction.py` defines `ExtractionError` (retryable; optional `retry_after_seconds`, plus `input_tokens`, `output_tokens`, `cost_usd` when the call was billed), `NonRetryableExtractionError`, and `ModelRefusalError`.
- `_classify_api_error` sorts SDK errors: `NON_RETRYABLE_STATUS_CODES` (400, 401, 403, 404, 413, 422) fail permanently, 408, 429, 529, and 5xx retry with any `retry-after`; transcription, the agent, and the judge reuse it.
- Two retry layers: the SDK's own (`anthropic_max_retries`, set in `_build_client`) inside one call, then the queue's (`worker.fail_job`: `MAX_ATTEMPTS` 3, `_backoff_delay` equal jitter capped at `BACKOFF_MAX` 60 s, a `retry-after` floor capped at `RETRY_AFTER_MAX` 900 s).
- `fail_job` keeps the last 2,000 characters of the traceback in `jobs.last_error`; a permanent failure marks the document `failed` or `refused` (extract) or fills `ask_runs.judge_error` (judge).
- A persistence error after a billed call is re-raised as `ExtractionError` carrying its tokens and cost, so the spend stays visible.
- `run_worker` catches every per-iteration exception, logs it, and keeps polling; `reclaim_orphaned_jobs` resets `processing` rows at startup, and with a lease (`WORKER_RECLAIM_AFTER_SECONDS` above 0) again every `RECLAIM_CHECK_SECONDS`.
- A job that has been claimed `MAX_ATTEMPTS` times without settling as done is failed at reclaim, not requeued: `last_error` carries `RECLAIM_EXHAUSTED_ERROR` and the worker logs a warning `failed job %s for good at reclaim: ...`.
- A stale settle is a warning, not an error: when `complete_job`, `fail_job`, or the shutdown `requeue_job` finds the job no longer this worker's claim (it was reclaimed, or already settled), `_drop_stale` rolls back the handler's writes and logs `dropped the <completion|failure|cancellation requeue> of job <id> (attempt <n>): it was reclaimed or already settled, so this worker's writes for it were rolled back`, and the outcome is `stale` on the `job <kind>` span (`docpilot.job.outcome`).
  Seeing it means a job outlasted `WORKER_RECLAIM_AFTER_SECONDS` (the model call may have been billed twice) or two workers raced; raise the lease above the slowest job.
  A Langfuse score is posted only after the job settles as `done`, so a stale run posts none.

## Logging and tracing

- stdlib `logging` with `logger = logging.getLogger(__name__)` and `%`-style arguments (`logger.info("claimed %s job %s (attempt %d)", ...)`); only `worker.py` calls `logging.basicConfig`, and the API runs on uvicorn's defaults.
- There are no request ids; correlation is OpenTelemetry (`app/telemetry.py`): a `job <kind>` span with `docpilot.job.*` attributes and outcome, a `model_call_span` per model call, and `jobs.traceparent` plus `ask_runs.trace_id` to join them.
- Jaeger runs under the compose `tracing` profile; Langfuse score failures are logged and dropped, never raised.
- No secret reaches a log, span, or error: nothing logs a password, session token, cookie, or API key; log lines carry ids (`login refused: user %s has no membership`, `could not record last_used_at for API key %s`); an API key's `AccessToken.token` is its public prefix; the accounts CLI turns a constraint refusal into one line naming the constraint (`refusal_message`), because the exception text holds the statement's parameters; and `ProvisioningError` never holds a password.
  `tests/test_api_keys.py` checks `caplog` and spans for a key's secret.
- `last_used_at` is an audit stamp: a failed write is logged with `exc_info` and the verified key is still accepted.
- The stdio MCP server prints nothing on stdout (it is the protocol channel): a missing or revoked key at startup is exit status 2 with the reason on stderr, and a key revoked later gives an `isError` result at the next call.

## Expired ORM objects in the worker

- All three handlers roll back before their model call, which expires the session's `Job`; a read of `job.id` afterwards raises `MissingGreenlet`, which once escaped to `run_worker`, lost the real failure, and left the job in `processing`.
- The session's organization survives the rollback: the `after_begin` listener re-applies it to the transaction the rollback begins, so a re-fetch with `session.get(Model, id)` sees the same tenant.
- So `complete_job` and `fail_job` take `job_id` and re-fetch, and `_run_handler` logs the carried `job_id`; after any rollback, never touch ORM attributes of an object loaded before it.
- `tests/test_worker.py` covers handlers that roll back and then succeed, fail, fail for good, or are cancelled.
- The worker touches a heartbeat file (`WORKER_HEARTBEAT_FILE`) every 10 s from a task beside the poll loop; compose's worker healthcheck fails when it is over 60 s old.
