"""Per-client rate limits (app/ratelimit.py): the sliding window itself,
and the billed routes that use it."""

import base64

import pytest
from httpx import AsyncClient

from app import ratelimit
from app.config import Settings, get_settings
from app.main import app
from app.ratelimit import WINDOW_SECONDS, SlidingWindowLimiter

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_window_slides() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(clock)
    for _ in range(3):  # hits at t=1000, 1010, 1020
        assert limiter.check("a", 3) is None
        clock.now += 10

    # t=1030: full; the oldest hit (t=1000) frees up at t=1060.
    assert limiter.check("a", 3) == pytest.approx(30)
    # Another client has its own window.
    assert limiter.check("b", 3) is None

    # A refused request isn't recorded: at t=1060 exactly one slot opens.
    clock.now = 1000 + WINDOW_SECONDS
    assert limiter.check("a", 3) is None
    assert limiter.check("a", 3) == pytest.approx(10)


def test_idle_clients_are_forgotten_past_the_tracking_cap(monkeypatch) -> None:
    monkeypatch.setattr(ratelimit, "MAX_TRACKED_CLIENTS", 3)
    clock = FakeClock()
    limiter = SlidingWindowLimiter(clock)
    for key in ("a", "b", "c"):
        limiter.check(key, 5)

    clock.now += WINDOW_SECONDS + 1
    limiter.check("d", 5)

    assert set(limiter._hits) == {"d"}


@pytest.fixture
def fresh_limiter(monkeypatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(ratelimit, "_limiter", SlidingWindowLimiter(clock))
    return clock


def _limits(tmp_path, *, ask: int = 0, upload: int = 0) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        upload_dir=str(tmp_path),
        daily_budget_usd=0,
        anthropic_api_key=None,  # /ask then answers 503 -- after the limiter
        ask_rate_limit_per_minute=ask,
        upload_rate_limit_per_minute=upload,
    )


async def test_ask_routes_share_one_per_client_bucket(
    client: AsyncClient, fresh_limiter, tmp_path
) -> None:
    _limits(tmp_path, ask=2)

    assert (await client.post("/ask", json={"question": "q"})).status_code == 503
    assert (await client.post("/ask/stream", json={"question": "q"})).status_code == 503
    limited = await client.post("/ask", json={"question": "q"})

    assert limited.status_code == 429
    assert "at most 2 per minute" in limited.json()["detail"]
    assert limited.headers["retry-after"] == "60"
    # Uploads are a different bucket, off by default.
    upload = await client.post("/documents", files={"file": ("r.png", TINY_PNG, "image/png")})
    assert upload.status_code == 201

    fresh_limiter.now += WINDOW_SECONDS
    assert (await client.post("/ask", json={"question": "q"})).status_code == 503


async def test_uploads_can_be_limited_too(client: AsyncClient, fresh_limiter, tmp_path) -> None:
    _limits(tmp_path, upload=1)
    files = {"file": ("r.png", TINY_PNG, "image/png")}

    assert (await client.post("/documents", files=files)).status_code == 201
    limited = await client.post("/documents", files=files)

    assert limited.status_code == 429 and "retry-after" in limited.headers
    # Refused before anything was written.
    assert len(list(tmp_path.iterdir())) == 1


async def test_a_zero_limit_is_off(client: AsyncClient, fresh_limiter, tmp_path) -> None:
    _limits(tmp_path, ask=0)

    statuses = {(await client.post("/ask", json={"question": "q"})).status_code for _ in range(15)}

    assert statuses == {503}


def test_limits_cannot_be_negative() -> None:
    with pytest.raises(ValueError):
        Settings(ask_rate_limit_per_minute=-1)


def test_limit_checks_run_on_the_event_loop() -> None:
    """A plain `def` dependency runs in FastAPI's threadpool, where two
    requests could interleave the limiter's check-then-record."""
    import inspect

    assert inspect.iscoroutinefunction(ratelimit.limit_ask)
    assert inspect.iscoroutinefunction(ratelimit.limit_upload)
