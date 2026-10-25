"""API keys as functions over an AsyncSession.

A key is shown once at creation; the row keeps its prefix (to find it and to
list it) and the SHA-256 digest of the whole token. A revoked key keeps its row
for audit and stops resolving. Like the job handlers, nothing here commits: the
caller owns the transaction. The one exception is `authenticate_api_key`, which
commits its throttled `last_used_at` write on the session it is given.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import set_committed_value

from app.auth.accounts import clean_name
from app.auth.tokens import digest, digests_match, mint_api_key, parse_api_key
from app.models import ApiKey

logger = logging.getLogger(__name__)


async def create_api_key(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    name: str,
    created_by_user_id: uuid.UUID | None = None,
) -> tuple[ApiKey, str]:
    """Store a new key for an organization and return its row with its token,
    the only copy of the secret. Raises AccountError for a name `clean_name`
    refuses."""
    name = clean_name(name, "an API key name")
    minted = mint_api_key()
    key = ApiKey(
        organization_id=organization_id,
        created_by_user_id=created_by_user_id,
        name=name,
        prefix=minted.prefix,
        token_digest=minted.token_digest,
    )
    session.add(key)
    await session.flush()
    return key, minted.token


async def find_active_api_key(session: AsyncSession, token: str) -> ApiKey | None:
    """The active key `token` is, or None. A token that is not exactly a key's
    shape is refused without a query; the digest is compared in constant time."""
    prefix = parse_api_key(token)
    if prefix is None:
        return None
    key = (
        await session.execute(
            select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked_at.is_(None))
        )
    ).scalar_one_or_none()
    if key is None or not digests_match(key.token_digest, digest(token)):
        return None
    return key


# At most this many active keys per organization (POST /api-keys answers 409
# past it); the CLI is the operator's and is not held to it.
MAX_ACTIVE_KEYS = 25
# `last_used_at` is written at most this often per key.
LAST_USED_INTERVAL = timedelta(minutes=1)


async def authenticate_api_key(
    session: AsyncSession, token: str, now: datetime | None = None
) -> ApiKey | None:
    """The active key `token` is, or None; every API request and MCP call with a
    bearer key comes through here. Writes `last_used_at` at most once per
    LAST_USED_INTERVAL: a key used again within it costs no write, and the
    update is conditional on the stored value, so concurrent requests write it
    once. The write is its own short transaction on an identity session (these
    tables have no row-level security and no tenant is bound), committed here so
    a read-only request never holds it open. It is an audit stamp, so a failure
    to write it is logged and the verified key is still returned."""
    key = await find_active_api_key(session, token)
    if key is None:
        return None
    now = now or datetime.now(UTC)
    if key.last_used_at is None or now - key.last_used_at >= LAST_USED_INTERVAL:
        # Detached first, so neither the commit nor a rollback expires what the
        # caller reads next.
        session.expunge(key)
        try:
            await session.execute(
                update(ApiKey)
                .where(
                    ApiKey.id == key.id,
                    or_(ApiKey.last_used_at.is_(None), ApiKey.last_used_at <= now - LAST_USED_INTERVAL),
                )
                .values(last_used_at=now)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
        except Exception:
            logger.warning("could not record last_used_at for API key %s", key.id, exc_info=True)
            await session.rollback()
        else:
            set_committed_value(key, "last_used_at", now)
    return key


async def list_api_keys(session: AsyncSession, organization_id: uuid.UUID) -> list[ApiKey]:
    """An organization's keys, revoked ones included, newest first."""
    rows = await session.execute(
        select(ApiKey)
        .where(ApiKey.organization_id == organization_id)
        .order_by(ApiKey.created_at.desc(), ApiKey.id)
    )
    return list(rows.scalars())


async def count_active_api_keys(session: AsyncSession, organization_id: uuid.UUID) -> int:
    return (
        await session.execute(
            select(func.count())
            .select_from(ApiKey)
            .where(ApiKey.organization_id == organization_id, ApiKey.revoked_at.is_(None))
        )
    ).scalar_one()


async def revoke_api_key(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    key_id: uuid.UUID,
    now: datetime | None = None,
) -> bool:
    """Revoke an organization's active key. False when the organization has no
    such active key (another organization's key is not found either)."""
    result = await session.execute(
        update(ApiKey)
        .where(
            ApiKey.id == key_id,
            ApiKey.organization_id == organization_id,
            ApiKey.revoked_at.is_(None),
        )
        .values(revoked_at=now or datetime.now(UTC))
        .returning(ApiKey.id)
    )
    return result.first() is not None
