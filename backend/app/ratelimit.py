"""Per-client rate limits on the routes that start billed model calls, and on
sign-in.

The daily budget (app/budget.py) bounds what a day of traffic can cost,
but one client could still spend all of it in a minute and lock everyone
else out until midnight. These limits bound how fast any one client can
spend: a sliding window of requests per minute. The client is the signed-in
principal (`Principal.rate_key`), not an address: behind Docker's bridge
network every request arrives from one address, which made an address limit
effectively global. /ask, /ask/stream and the per-document chat share one
bucket; uploads have their own. /mcp (app/mcp_server/http.py) calls no model but
has a bucket of its own, "mcp", to bound the database and embedder work a client
can ask for; it still counts by address until API keys name its callers.

Sign-in is the one route with no principal, so it is counted twice, both before
any password is hashed: per client address (`limit_login`, a dependency, so it
runs before the body is parsed), and per account (`limit_login_account`). One
client then cannot try a password against many accounts, nor many clients
against one account. The account bucket is keyed by a digest of the email as
the database folds it (accounts.folded_email), for every email whether or not an
account has it: every spelling the users table treats as one account shares a
bucket, and an unknown email is limited exactly like a known one, so a 429 never
says which exist.

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

from app.auth.principal import Principal, current_principal
from app.auth.tokens import digest
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


def enforce_rate_limit(
    request: Request, bucket: str, limit: int, *, key: str | None = None, per: str = "client"
) -> None:
    """Count one request against `bucket` and raise 429 past `limit` a minute.
    `key` names who is counted (default: the client address) and `per` how the
    refusal calls them."""
    if limit <= 0:
        return
    retry_after = _limiter.check(f"{bucket}:{key if key is not None else _client_key(request)}", limit)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"too many requests: at most {limit} per minute per {per} for this route",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )


# async on purpose, though nothing inside awaits: FastAPI runs plain `def`
# dependencies in its threadpool, where concurrent requests would race on
# the limiter's check-then-record (overshooting the limit, or popping an
# emptied deque). On the event loop each check runs to completion alone.
async def limit_ask(
    request: Request,
    principal: Principal = Depends(current_principal),
    settings: Settings = Depends(get_settings),
) -> None:
    enforce_rate_limit(request, "ask", settings.ask_rate_limit_per_minute, key=principal.rate_key)


async def limit_upload(
    request: Request,
    principal: Principal = Depends(current_principal),
    settings: Settings = Depends(get_settings),
) -> None:
    enforce_rate_limit(request, "upload", settings.upload_rate_limit_per_minute, key=principal.rate_key)


async def limit_login(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Sign-in attempts per client address; the per-account half is
    `limit_login_account`, which needs the request body."""
    enforce_rate_limit(request, "login", settings.login_rate_limit_per_minute)


def limit_login_account(request: Request, settings: Settings, *, folded_email: str) -> None:
    """Sign-in attempts per account, keyed by a digest of the folded email
    (never the email itself)."""
    enforce_rate_limit(
        request,
        "login-account",
        settings.login_account_rate_limit_per_minute,
        key=f"email:{digest(folded_email).hex()}",
        per="account",
    )
