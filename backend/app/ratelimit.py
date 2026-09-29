"""Per-client rate limits on the routes that start billed model calls.

The daily budget (app/budget.py) bounds what a day of traffic can cost,
but one client could still spend all of it in a minute and lock everyone
else out until midnight. These limits bound how fast any one client can
spend: a sliding window of requests per minute, keyed by client address.
/ask (and /ask/stream) share one bucket; uploads have their own.

In-memory and per process, which fits this deployment (one API process;
see docker-compose.yml). It resets on restart, and N API processes would
mean N times the limit -- a shared store (Postgres, Redis) is the fix if
the API ever scales out. Behind a reverse proxy every request arrives
from the proxy's address, so run uvicorn with --proxy-headers and
--forwarded-allow-ips set to the proxy for request.client to be the real
client.
"""

import math
import time
from collections import deque
from collections.abc import Callable

from fastapi import Depends, HTTPException, Request

from app.config import Settings, get_settings

WINDOW_SECONDS = 60.0
# Past this many tracked clients, idle ones are dropped on the next check,
# so a flood of distinct addresses can't grow memory without bound.
MAX_TRACKED_CLIENTS = 10_000


class SlidingWindowLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str, limit: int) -> float | None:
        """Record a request from `key` and return None, or return the
        seconds until it may try again (recording nothing) if it has made
        `limit` requests in the last WINDOW_SECONDS."""
        now = self._clock()
        cutoff = now - WINDOW_SECONDS
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= limit:
            return hits[0] + WINDOW_SECONDS - now
        hits.append(now)
        if len(self._hits) > MAX_TRACKED_CLIENTS:
            self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > cutoff}
        return None


_limiter = SlidingWindowLimiter()


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _enforce(request: Request, bucket: str, limit: int) -> None:
    if limit <= 0:
        return
    retry_after = _limiter.check(f"{bucket}:{_client_key(request)}", limit)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"too many requests: at most {limit} per minute per client for this route",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )


# async on purpose, though nothing inside awaits: FastAPI runs plain `def`
# dependencies in its threadpool, where concurrent requests would race on
# the limiter's check-then-record (overshooting the limit, or popping an
# emptied deque). On the event loop each check runs to completion alone.
async def limit_ask(request: Request, settings: Settings = Depends(get_settings)) -> None:
    _enforce(request, "ask", settings.ask_rate_limit_per_minute)


async def limit_upload(request: Request, settings: Settings = Depends(get_settings)) -> None:
    _enforce(request, "upload", settings.upload_rate_limit_per_minute)
