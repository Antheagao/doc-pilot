"""API keys: an organization's credentials for programmatic access (the MCP
server, scripts), managed by its owners in a browser session.

A key is shown once: the 201 of POST /api-keys is the only response that ever
holds its secret, and only its SHA-256 digest is stored (app/auth/tokens.py).
Issuing a key is an admin act, because a key outlives a session and bypasses the
cookie and Origin protections, so creating and revoking need the owner role and
a session: a key itself cannot manage keys (403). A member may list them.
Another organization's key is a 404, like any other missing row.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select

from app.auth import accounts, api_keys
from app.auth.principal import Principal, get_identity_session_factory, session_only
from app.models import MEMBERSHIP_ROLE_OWNER, Organization
from app.schemas import ApiKeyCreated, ApiKeyCreateRequest, ApiKeyList, ApiKeyOut
from app.tenancy import SessionFactory

router = APIRouter()

manage_api_keys = session_only("an API key cannot manage API keys")


def _owner(principal: Principal = Depends(manage_api_keys)) -> Principal:
    if principal.role != MEMBERSHIP_ROLE_OWNER:
        raise HTTPException(status_code=403, detail="only an owner can manage API keys")
    return principal


@router.get("", response_model=ApiKeyList)
async def list_keys(
    response: Response,
    principal: Principal = Depends(manage_api_keys),
    identity_sessions: SessionFactory = Depends(get_identity_session_factory),
) -> ApiKeyList:
    async with identity_sessions() as db:
        keys = await api_keys.list_api_keys(db, principal.organization_id)
    return ApiKeyList(api_keys=[ApiKeyOut.model_validate(key) for key in keys])


@router.post("", status_code=201, response_model=ApiKeyCreated)
async def create_key(
    body: ApiKeyCreateRequest,
    response: Response,
    principal: Principal = Depends(_owner),
    identity_sessions: SessionFactory = Depends(get_identity_session_factory),
) -> ApiKeyCreated:
    try:
        name = accounts.clean_name(body.name, "an API key name")
    except accounts.AccountError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    async with identity_sessions() as db:
        # Serializes this organization's creates, so two at once cannot both
        # pass the cap. FOR NO KEY UPDATE, not FOR UPDATE: every content insert
        # takes KEY SHARE on this row through its organization_id foreign key,
        # and FOR UPDATE would make uploads and worker writes wait on a key
        # creation (and the reverse).
        await db.execute(
            select(Organization.id)
            .where(Organization.id == principal.organization_id)
            .with_for_update(key_share=True)
        )
        if await api_keys.count_active_api_keys(db, principal.organization_id) >= api_keys.MAX_ACTIVE_KEYS:
            raise HTTPException(
                status_code=409,
                detail=f"an organization can have at most {api_keys.MAX_ACTIVE_KEYS} active API keys: revoke one first",
            )
        key, secret = await api_keys.create_api_key(
            db,
            organization_id=principal.organization_id,
            name=name,
            created_by_user_id=principal.user_id,
        )
        await db.refresh(key)  # created_at is the database's
        created = ApiKeyOut.model_validate(key)
        await db.commit()
    response.headers["Cache-Control"] = "no-store"
    return ApiKeyCreated(api_key=created, secret=secret)


@router.delete("/{key_id}", status_code=204)
async def revoke_key(
    key_id: uuid.UUID,
    principal: Principal = Depends(_owner),
    identity_sessions: SessionFactory = Depends(get_identity_session_factory),
) -> Response:
    async with identity_sessions() as db:
        revoked = await api_keys.revoke_api_key(
            db, organization_id=principal.organization_id, key_id=key_id
        )
        await db.commit()
    if not revoked:
        raise HTTPException(status_code=404, detail="API key not found")
    return Response(status_code=204)
