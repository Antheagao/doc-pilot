"""Who is making a request: the browser's session cookie, checked on every route.

`current_principal` is the FastAPI dependency every router but /auth hangs off
(app/main.py), and `get_session` (app/db.py) binds its organization, so a route
that touches data cannot run unauthenticated or unscoped. It answers 401 with
one generic body for every way a credential can fail (missing, unknown,
expired, a disabled user, a removed membership), so a response never says which.

- An `Authorization: Bearer` header is a 401 for now: API keys arrive with the
  API key batch task, and ignoring the header would let a request that meant to
  authenticate as a key fall through to a cookie it did not choose.
- A session cookie is looked up by the SHA-256 digest of its token
  (app/auth/tokens.py) in one joined read. The lookup only reads: the expiry is
  absolute, there is no sliding renewal, and an expired row found on the way is
  deleted.
- Cookies ride along on requests a page did not mean to send, so a
  cookie-authenticated request that changes state (anything but GET, HEAD, and
  OPTIONS) must also carry an `Origin` listed in CORS_ORIGINS, or it gets 403.
  `SameSite=Lax` does not cover sibling origins: every other app on
  `localhost:<port>` is same-site with the API. Login and logout check the same
  rule (`require_allowed_origin`) though no principal exists there.

The identity tables have no row-level security (a login must read them before
any tenant is known), so the principal is resolved on a session of its own that
is closed before the route opens its tenant session: a streamed answer does not
hold a second connection for its whole run.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import Depends, HTTPException, Request
from sqlalchemy import delete, select

from app.auth.tokens import digest
from app.config import Settings, get_settings
from app.models import MEMBERSHIP_ROLE_MEMBER, AuthSession, Membership, User
from app.tenancy import SessionFactory

# Methods that do not change state, so a forged cross-site one does no harm.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


@dataclass(frozen=True)
class Principal:
    """The caller: the organization whose data it may see, and who is asking.
    `user_id` is None for a caller that is an organization rather than a person
    (the MCP server until API keys name one)."""

    organization_id: uuid.UUID
    user_id: uuid.UUID | None = None
    role: str = MEMBERSHIP_ROLE_MEMBER

    @property
    def rate_key(self) -> str:
        """What the per-minute limits (app/ratelimit.py) count this caller as:
        a signed-in person is one client wherever the request comes from, which
        a client address could not say behind Docker's bridge network."""
        if self.user_id is None:
            return f"org:{self.organization_id}"
        return f"user:{self.user_id}"


def unauthenticated() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="not authenticated",
        headers={"WWW-Authenticate": 'Bearer realm="doc-pilot"'},
    )


def get_identity_session_factory() -> SessionFactory:
    """The factory of sessions the identity tables are read and written on: not
    bound to an organization, because none is known yet. A dependency so tests
    can hand it their own transaction."""
    # Imported here because app.db imports this module for current_principal.
    from app.db import async_session_maker

    return async_session_maker


async def require_allowed_origin(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Refuse (403) a request whose Origin is not one of CORS_ORIGINS. A browser
    sends Origin on every cross-origin write; a request without one is not from
    the frontend."""
    if request.headers.get("origin") not in settings.cors_origins:
        raise HTTPException(status_code=403, detail="origin not allowed")


async def current_principal(
    request: Request,
    settings: Settings = Depends(get_settings),
    identity_sessions: SessionFactory = Depends(get_identity_session_factory),
) -> Principal:
    """The principal of the current request (a FastAPI dependency). Raises 401
    for any credential that does not resolve, and 403 for a cookie-authenticated
    write from an Origin that is not allowed."""
    if request.headers.get("authorization", "").partition(" ")[0].lower() == "bearer":
        raise unauthenticated()
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        raise unauthenticated()
    # Before the lookup: a forged write costs no database work.
    if request.method not in SAFE_METHODS:
        await require_allowed_origin(request, settings)
    async with identity_sessions() as db:
        # One read for the session and who it belongs to. Both joins hold for
        # any row: the foreign keys end a session with its membership or user.
        found = (
            await db.execute(
                select(
                    AuthSession.id,
                    AuthSession.user_id,
                    AuthSession.organization_id,
                    AuthSession.expires_at,
                    User.disabled_at,
                    Membership.role,
                )
                .join(User, User.id == AuthSession.user_id)
                .join(
                    Membership,
                    (Membership.user_id == AuthSession.user_id)
                    & (Membership.organization_id == AuthSession.organization_id),
                )
                .where(AuthSession.token_digest == digest(token))
            )
        ).first()
        if found is None:
            raise unauthenticated()
        if found.expires_at <= datetime.now(UTC):
            await db.execute(delete(AuthSession).where(AuthSession.id == found.id))
            await db.commit()
            raise unauthenticated()
    if found.disabled_at is not None:
        raise unauthenticated()
    return Principal(organization_id=found.organization_id, user_id=found.user_id, role=found.role)
