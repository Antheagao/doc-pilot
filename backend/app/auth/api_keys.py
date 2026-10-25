"""API keys as functions over an AsyncSession.

A key is shown once at creation; the row keeps its prefix (to find it and to
list it) and the SHA-256 digest of the whole token. A revoked key keeps its row
for audit and stops resolving. Like the job handlers, nothing here commits: the
caller owns the transaction.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.accounts import clean_name
from app.auth.tokens import digest, digests_match, mint_api_key, parse_api_key
from app.models import ApiKey


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
