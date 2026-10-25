"""Sign-in for the browser: a session cookie over the accounts an operator made.

There is no signup route (scripts/accounts.py creates users). The cookie holds
a random token whose SHA-256 digest is the session row (app/auth/sessions.py),
so logout is a row delete and a database read never yields a usable cookie.

- POST /auth/login: the per-address limit, then the body's length checks, then
  the per-account limit, then the password check, and only then a session; the
  limits come before any hashing. One 401 message covers a wrong password, an
  unknown email, and a disabled user, and the last two still pay for a verify.
- POST /auth/logout: 204 whether or not there was a session.
- GET /auth/me and POST /auth/organization need the cookie like any route.

Login and logout have no principal to protect, but they change cookies, so they
need an allowed Origin like every cookie-authenticated write
(app/auth/principal.py). Nothing here logs a password, token, or cookie value.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import accounts, passwords, sessions
from app.auth.principal import (
    Principal,
    current_principal,
    get_identity_session_factory,
    require_allowed_origin,
    unauthenticated,
)
from app.config import Settings, get_settings
from app.models import Membership, Organization, User
from app.ratelimit import limit_login, limit_login_account
from app.schemas import LoginRequest, MeOut, OrganizationOut, SwitchOrganizationRequest
from app.tenancy import SessionFactory

logger = logging.getLogger(__name__)
router = APIRouter()

BAD_CREDENTIALS = "email or password is incorrect"


def _no_store(response: Response) -> None:
    """These responses are about one person and their credentials."""
    response.headers["Cache-Control"] = "no-store"


def _set_session_cookie(response: Response, settings: Settings, token: str, max_age: int) -> None:
    response.set_cookie(
        settings.session_cookie_name,
        token,
        max_age=max_age,
        path="/",
        secure=settings.session_cookie_secure,
        httponly=True,
        samesite="lax",
    )


async def _find_user(db: AsyncSession, email: str) -> User | None:
    try:
        return await accounts.get_user(db, email)
    except accounts.AccountError:
        return None


async def _memberships(db: AsyncSession, user_id: uuid.UUID) -> list[OrganizationOut]:
    """The user's organizations, the earliest membership first."""
    rows = await db.execute(
        select(Organization, Membership.role)
        .join(Membership, Membership.organization_id == Organization.id)
        .where(Membership.user_id == user_id)
        .order_by(Membership.created_at, Organization.id)
    )
    return [
        OrganizationOut(id=organization.id, name=organization.name, slug=organization.slug, role=role)
        for organization, role in rows.all()
    ]


def _me(user_id: uuid.UUID, email: str, organization_id: uuid.UUID, organizations: list[OrganizationOut]) -> MeOut:
    current = next(item for item in organizations if item.id == organization_id)
    return MeOut(id=user_id, email=email, organization=current, organizations=organizations)


@router.post(
    "/login",
    response_model=MeOut,
    dependencies=[Depends(require_allowed_origin), Depends(limit_login)],
)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
    identity_sessions: SessionFactory = Depends(get_identity_session_factory),
) -> MeOut:
    password = body.password.get_secret_value()
    async with identity_sessions() as db:
        user = await _find_user(db, body.email)
        folded = await accounts.folded_email(db, body.email)
    limit_login_account(request, settings, folded_email=folded)

    if user is None or user.disabled_at is not None:
        await passwords.verify_dummy_password(password)
        raise HTTPException(status_code=401, detail=BAD_CREDENTIALS)
    user_id, email, stored_hash = user.id, user.email, user.password_hash
    if not await passwords.verify_password(password, stored_hash):
        raise HTTPException(status_code=401, detail=BAD_CREDENTIALS)
    rehashed = await passwords.hash_password(password) if passwords.needs_rehash(stored_hash) else None

    ttl = timedelta(hours=settings.session_ttl_hours)
    async with identity_sessions() as db:
        if rehashed is not None:
            # Only if the password has not changed since it was read.
            await db.execute(
                update(User)
                .where(User.id == user_id, User.password_hash == stored_hash)
                .values(password_hash=rehashed)
            )
        await sessions.delete_expired_sessions(db)
        organizations = await _memberships(db, user_id)
        if not organizations:
            await db.commit()
            # The password was right but there is nowhere to sign in to: the
            # same answer as any other failure, and the reason for the operator.
            logger.warning("login refused: user %s has no membership", user_id)
            raise HTTPException(status_code=401, detail=BAD_CREDENTIALS)
        token = await sessions.create_session(
            db, user_id=user_id, organization_id=organizations[0].id, ttl=ttl
        )
        await db.commit()
    _set_session_cookie(response, settings, token, max_age=int(ttl.total_seconds()))
    _no_store(response)
    return _me(user_id, email, organizations[0].id, organizations)


@router.post("/logout", status_code=204, dependencies=[Depends(require_allowed_origin)])
async def logout(
    request: Request,
    settings: Settings = Depends(get_settings),
    identity_sessions: SessionFactory = Depends(get_identity_session_factory),
) -> Response:
    """End the session the cookie names and clear the cookie. Idempotent: no
    cookie, or one whose session is already gone, is the same 204."""
    token = request.cookies.get(settings.session_cookie_name)
    if token:
        async with identity_sessions() as db:
            await sessions.delete_session(db, token)
            await db.commit()
    response = Response(status_code=204)
    response.delete_cookie(
        settings.session_cookie_name,
        path="/",
        secure=settings.session_cookie_secure,
        httponly=True,
        samesite="lax",
    )
    _no_store(response)
    return response


@router.get("/me", response_model=MeOut)
async def me(
    response: Response,
    principal: Principal = Depends(current_principal),
    identity_sessions: SessionFactory = Depends(get_identity_session_factory),
) -> MeOut:
    async with identity_sessions() as db:
        email = (await db.execute(select(User.email).where(User.id == principal.user_id))).scalar_one()
        organizations = await _memberships(db, principal.user_id)
    _no_store(response)
    return _me(principal.user_id, email, principal.organization_id, organizations)


@router.post("/organization", response_model=MeOut)
async def switch_organization(
    body: SwitchOrganizationRequest,
    request: Request,
    response: Response,
    principal: Principal = Depends(current_principal),
    settings: Settings = Depends(get_settings),
    identity_sessions: SessionFactory = Depends(get_identity_session_factory),
) -> MeOut:
    """Work in another of the user's organizations. The session is replaced by
    a new token that keeps the old one's expiry, so a switch cannot extend a
    sign-in; the old cookie stops working at once."""
    old_token = request.cookies[settings.session_cookie_name]
    async with identity_sessions() as db:
        organizations = await _memberships(db, principal.user_id)
        if all(item.id != body.organization_id for item in organizations):
            raise HTTPException(status_code=404, detail="organization not found")
        old = await sessions.lookup_session(db, old_token)
        if old is None:
            raise unauthenticated()
        remaining = old.expires_at - datetime.now(UTC)
        # The delete is what makes the old token single-use: of two switches
        # racing on one token, the second finds no row and mints nothing.
        if not await sessions.delete_session(db, old_token):
            raise unauthenticated()
        token = await sessions.create_session(
            db, user_id=principal.user_id, organization_id=body.organization_id, ttl=remaining
        )
        email = (await db.execute(select(User.email).where(User.id == principal.user_id))).scalar_one()
        await db.commit()
    _set_session_cookie(response, settings, token, max_age=int(remaining.total_seconds()))
    _no_store(response)
    return _me(principal.user_id, email, body.organization_id, organizations)
