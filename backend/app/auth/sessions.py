"""Browser sessions as functions over an AsyncSession.

A session row holds the SHA-256 digest of the cookie's token, never the token
(app/auth/tokens.py). The expiry is absolute and checked in the query, so an
expired row is never returned even before it is swept. Like the job handlers,
nothing here commits: the caller owns the transaction.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.tokens import digest, new_session_token
from app.models import AuthSession


async def create_session(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    organization_id: uuid.UUID,
    ttl: timedelta,
    now: datetime | None = None,
) -> str:
    """Store a new session for a user's membership in an organization (a
    foreign key refuses a pair that is not a membership) and return its token,
    the only copy: it goes in the cookie."""
    token = new_session_token()
    session.add(
        AuthSession(
            token_digest=digest(token),
            user_id=user_id,
            organization_id=organization_id,
            expires_at=(now or datetime.now(UTC)) + ttl,
        )
    )
    await session.flush()
    return token


async def lookup_session(
    session: AsyncSession, token: str, now: datetime | None = None
) -> AuthSession | None:
    """The unexpired session `token` belongs to, or None."""
    stmt = select(AuthSession).where(
        AuthSession.token_digest == digest(token),
        AuthSession.expires_at > (now or datetime.now(UTC)),
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def delete_session(session: AsyncSession, token: str) -> bool:
    """End the session `token` belongs to (logout). False when there was none."""
    result = await session.execute(
        delete(AuthSession).where(AuthSession.token_digest == digest(token)).returning(AuthSession.id)
    )
    return result.first() is not None


async def delete_user_sessions(session: AsyncSession, user_id: uuid.UUID) -> int:
    """End every session of a user, in every organization. Returns how many."""
    result = await session.execute(
        delete(AuthSession).where(AuthSession.user_id == user_id).returning(AuthSession.id)
    )
    return len(result.all())


async def delete_expired_sessions(session: AsyncSession, now: datetime | None = None) -> int:
    """Sweep the sessions whose expiry has passed. Returns how many."""
    result = await session.execute(
        delete(AuthSession)
        .where(AuthSession.expires_at <= (now or datetime.now(UTC)))
        .returning(AuthSession.id)
    )
    return len(result.all())
