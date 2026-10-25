"""Locust scenarios for the doc-pilot load test. Runs inside the loadgen
container (docker-compose.load.yml), started by scripts/load/load.py:

    locust -f /load/locustfile.py --headless --host http://api:8000 -u N -r N -t 60s

The harness owns the timeline and Locust is only the engine, so nothing here
relies on Locust's aggregated statistics (they are bucketed). Every request
writes one line, with its own UTC start time and duration, to the file named by
LOAD_SAMPLES_FILE; the harness computes exact percentiles from those lines.

One user class is made per tenant and kind of credential (`load-large:key`,
`load-large:cookie`, ...): a key user sends `Authorization: Bearer dpk_...`, a
cookie user signs in once with `/auth/login` (the real argon2 check, the real
session cookie) and sends the `Origin` that cookie-authenticated writes need.
Keys and passwords are read from the tenants file `load.py seed` wrote; they are
never logged and never part of a sample.

HttpUser, not FastHttpUser: FastHttpUser buffers a streamed response whole, so
the first event of /ask/stream would appear at the same time as its answer.
HttpUser's iter_content hands over each chunk as it arrives, and the SSE parser
(workload.py) stamps the moment each frame became whole.

Settings come from environment variables (set by load.py through `compose exec
-e`): LOAD_SAMPLES_FILE (required), LOAD_WORKLOAD (`read` or `ingest`),
LOAD_UPLOAD_QUOTA (ingest: stop after this many uploads), LOAD_USER_WEIGHTS and
LOAD_READ_MIX (JSON; see workload.py), LOAD_EXCLUDE_QUESTION_TYPES (comma-separated), LOAD_ORIGIN, LOAD_TENANTS_FILE,
LOAD_EVALS_DIR.
"""

import itertools
import json
import os
import random
import time
import uuid
from pathlib import Path

import gevent
import workload
from locust import HttpUser, constant
from locust.exception import StopUser

EVALS_DIR = Path(os.environ.get("LOAD_EVALS_DIR", "/evals"))
TENANTS_FILE = Path(os.environ.get("LOAD_TENANTS_FILE", "/load/out/tenants.json"))
SAMPLES_FILE = os.environ["LOAD_SAMPLES_FILE"]
WORKLOAD = os.environ.get("LOAD_WORKLOAD", "read")
UPLOAD_QUOTA = int(os.environ.get("LOAD_UPLOAD_QUOTA", "0"))
# An origin in the stack's CORS_ORIGINS (docker-compose.load.yml): a cookie write
# without one is a 403.
ORIGIN = os.environ.get("LOAD_ORIGIN", "http://localhost:3200")
# A request that has not answered by now is an error; the slowest legitimate
# one is a scripted answer at full replay latency, well inside it.
REQUEST_TIMEOUT_SECONDS = 120

TENANTS = json.loads(TENANTS_FILE.read_text(encoding="utf-8"))["tenants"]
USER_WEIGHTS = workload.parse_user_weights(os.environ.get("LOAD_USER_WEIGHTS"))
READ_MIX = workload.parse_read_mix(os.environ.get("LOAD_READ_MIX"))

CORPUS = workload.corpus_documents(EVALS_DIR)
QUERIES = workload.search_queries(EVALS_DIR)
QUESTIONS = workload.ask_questions(
    EVALS_DIR, workload.parse_exclude_types(os.environ.get("LOAD_EXCLUDE_QUESTION_TYPES"))
)
SAMPLES = workload.SampleWriter(SAMPLES_FILE)
# One count for the whole process, so uploads walk the corpus evenly and the
# quota is shared by every user.
UPLOADS = itertools.count()


class LoadUser(HttpUser):
    """A virtual user of one tenant. Closed loop: no think time, so the number
    of users is the concurrency."""

    abstract = True
    wait_time = constant(0)
    tenant = ""
    auth = ""

    def on_start(self) -> None:
        record = TENANTS[self.tenant]
        self.rng = random.Random(f"{self.tenant}:{self.auth}:{uuid.uuid4().hex}")
        if self.auth == "key":
            self.client.headers["Authorization"] = f"Bearer {record['api_key']}"
        else:
            self.client.headers["Origin"] = ORIGIN
            self._sign_in(record)

    def _record(
        self, name: str, start: float, started: float, status: int, error: str | None, **extra: object
    ) -> None:
        SAMPLES.write(
            workload.make_sample(
                name,
                tenant=self.tenant,
                auth=self.auth,
                start=start,
                ms=(time.perf_counter() - started) * 1000,
                status=status,
                error=error,
                **extra,
            )
        )

    def _sign_in(self, record: dict) -> None:
        start, started = time.time(), time.perf_counter()
        body = {"email": record["email"], "password": record["password"]}
        with self.client.post("/auth/login", json=body, name="login", catch_response=True) as response:
            error = _http_problem(response)
            if error:
                response.failure(error)
            else:
                response.success()
        self._record("login", start, started, response.status_code, error)
        if error:
            raise StopUser()

    def _call(self, name: str, method: str, path: str, check=None, **kwargs: object) -> dict | None:
        """One request, timed and recorded. `check` turns the decoded JSON body
        into an error code or None; the body is returned when the call
        succeeded."""
        start, started = time.time(), time.perf_counter()
        body = None
        with self.client.request(
            method, path, name=name, catch_response=True, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs
        ) as response:
            error = _http_problem(response)
            if error is None:
                try:
                    body = response.json()
                except ValueError:
                    error = "bad_json"
            if error is None and check is not None:
                error = check(body)
            if error:
                response.failure(error)
            else:
                response.success()
        extra = {}
        if error is None and name == "upload":
            extra["document_id"] = body.get("id")
        self._record(name, start, started, response.status_code, error, **extra)
        return None if error else body


def _http_problem(response) -> str | None:
    """An error code for a response that is not a success, else None."""
    if response.status_code == 0:
        return "connection"
    if response.status_code >= 400:
        return f"http_{response.status_code}"
    return None


# --- the tasks (the first argument is the user, as Locust passes it) --------


def search(user: LoadUser) -> None:
    query = user.rng.choice(QUERIES)
    params = {"q": query, "k": workload.SEARCH_K, "mode": workload.SEARCH_MODE}
    user._call(
        "search",
        "GET",
        "/search",
        check=lambda body: None if isinstance(body.get("results"), list) else "bad_search",
        params=params,
    )


def documents(user: LoadUser) -> None:
    user._call(
        "documents",
        "GET",
        "/documents",
        check=lambda body: None if isinstance(body, list) else "bad_documents",
    )


def ask(user: LoadUser) -> None:
    user._call(
        "ask",
        "POST",
        "/ask",
        check=workload.answer_problem,
        json={"question": user.rng.choice(QUESTIONS)},
    )


def ask_stream(user: LoadUser) -> None:
    """POST /ask/stream, reading the frames as they arrive: the time to the
    first event of any kind and to the `answer` event, from the request's
    start. The sample's `ms` is the whole stream."""
    start, started = time.time(), time.perf_counter()
    first_ms = answer_ms = None
    frames = 0
    error = None
    with user.client.post(
        "/ask/stream",
        json={"question": user.rng.choice(QUESTIONS)},
        name="ask_stream",
        stream=True,
        catch_response=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        error = _http_problem(response)
        if error is None:
            parser = workload.SseParser()
            for chunk in response.iter_content(chunk_size=None):
                arrived = (time.perf_counter() - started) * 1000
                for frame in parser.feed(chunk):
                    frames += 1
                    if first_ms is None:
                        first_ms = arrived
                    if frame.event == "answer" and answer_ms is None:
                        answer_ms = arrived
                    error = workload.stream_frame_problem(frame) or error
            if error is None and answer_ms is None:
                error = "no_answer_event"
        if error:
            response.failure(error)
        else:
            response.success()
    user._record(
        "ask_stream",
        start,
        started,
        response.status_code,
        error,
        first_event_ms=None if first_ms is None else round(first_ms, 3),
        answer_ms=None if answer_ms is None else round(answer_ms, 3),
        frames=frames,
    )


def upload(user: LoadUser) -> None:
    """Upload the next corpus document with a nonce of its own. The last
    upload of the quota ends the run (users mid-upload are allowed to finish by
    the harness's --stop-timeout)."""
    number = next(UPLOADS)
    if UPLOAD_QUOTA and number >= UPLOAD_QUOTA:
        raise StopUser()
    document = CORPUS[number % len(CORPUS)]
    data = workload.with_nonce(document.data, document.media_type, uuid.uuid4().hex)
    user._call(
        "upload",
        "POST",
        "/documents",
        files={"file": (f"load-{number:06d}-{document.name}", data, document.media_type)},
    )
    if UPLOAD_QUOTA and number == UPLOAD_QUOTA - 1:
        gevent.spawn(user.environment.runner.quit)


READ_TASKS = {"search": search, "ask": ask, "ask_stream": ask_stream, "documents": documents}
TASKS = {upload: 1} if WORKLOAD == "ingest" else {READ_TASKS[name]: weight for name, weight in READ_MIX.items()}


def _make_user_classes() -> dict[str, type[LoadUser]]:
    made = {}
    for name, weight in USER_WEIGHTS.items():
        tenant, auth = name.split(":")
        class_name = f"{tenant}_{auth}".replace("-", "_")
        made[class_name] = type(
            class_name, (LoadUser,), {"tenant": tenant, "auth": auth, "weight": weight, "tasks": TASKS}
        )
    return made


# Locust finds user classes among the module's names.
globals().update(_make_user_classes())
