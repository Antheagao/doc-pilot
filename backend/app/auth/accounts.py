"""Organizations, users, and memberships as functions over an AsyncSession.

Accounts are made by an operator through scripts/accounts.py; there is no
signup route. Input is checked here against the same limits as the database
CHECKs (ck_organizations_*, ck_users_email), so a bad value is a clear error
rather than a constraint violation. Like the job handlers, nothing here
commits: the caller owns the transaction.

An email is put in Unicode normalization form NFKC, trimmed, and checked for its
shape (`canonical_email`) before it is stored or looked up, so the precomposed
and decomposed spellings of one address are one account. Case is left to the
database: the column is citext, and Python's own case folding disagrees with
it for a few characters, so nothing here folds case for storage.
"""

import re
import unicodedata
import uuid

from sqlalchemy import cast, func, select
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import sessions
from app.auth.passwords import hash_password
from app.models import MEMBERSHIP_ROLES, Membership, Organization, User

NAME_MAX_CHARS = 100
EMAIL_MAX_CHARS = 254
SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{1,62}")
# Exactly one "@" with something on each side, and no whitespace or control
# characters (ck_users_email_shape says the same in SQL).
_EMAIL_SHAPE = re.compile(r"[^@\s]+@[^@\s]+")


class AccountError(ValueError):
    """An account operation refused for a reason the operator can fix: bad
    input, a name already taken, or something that does not exist. A password
    of the wrong length raises the plain ValueError of `hash_password`."""


def canonical_email(email: str) -> str:
    """The stored form of `email`: NFKC-normalized and trimmed. Raises
    AccountError unless it is 1 to 254 characters with exactly one "@" and no
    whitespace or control characters (Unicode categories C and Z, a superset of
    what the database CHECK refuses)."""
    email = unicodedata.normalize("NFKC", email).strip()
    if not 1 <= len(email) <= EMAIL_MAX_CHARS:
        raise AccountError(f"an email is 1 to {EMAIL_MAX_CHARS} characters long")
    if _EMAIL_SHAPE.fullmatch(email) is None or any(
        unicodedata.category(char)[0] in "CZ" for char in email
    ):
        raise AccountError('an email has exactly one "@" and no spaces or control characters')
    return email


def clean_name(name: str, what: str) -> str:
    """`name` trimmed, for an organization or API key (`what` says which in the
    error). Raises AccountError unless it is 1 to NAME_MAX_CHARS characters with
    no control characters, which a NUL among them would otherwise turn into a
    database driver error."""
    name = name.strip()
    if not 1 <= len(name) <= NAME_MAX_CHARS or any(unicodedata.category(c)[0] == "C" for c in name):
        raise AccountError(f"{what} is 1 to {NAME_MAX_CHARS} characters long, with no control characters")
    return name


async def folded_email(session: AsyncSession, email: str) -> str:
    """`email` as the database compares it: normalized, then folded by the
    database's own lower() on a citext, which is how the users.email column
    decides two spellings are one account (Python's casefold() disagrees on a
    few characters, such as the dotted capital I and the capital sharp S). The
    sign-in limiter keys on it, so a spelling cannot dodge the account's bucket
    and no spelling tells an existing account from a missing one. Text that
    could not be stored as an email is folded in Python: no account has it, and
    a control character would be a driver error in the query."""
    try:
        email = canonical_email(email)
    except AccountError:
        return unicodedata.normalize("NFKC", email).strip().casefold()
    return (await session.execute(select(func.lower(cast(email, CITEXT))))).scalar_one()


async def get_organization(session: AsyncSession, slug: str) -> Organization:
    # A slug that could not exist is not queried: a NUL in it would be a driver
    # error rather than a miss.
    organization = None
    if SLUG_PATTERN.fullmatch(slug):
        organization = (
            await session.execute(select(Organization).where(Organization.slug == slug))
        ).scalar_one_or_none()
    if organization is None:
        raise AccountError(f"no organization with slug {slug!r}")
    return organization


async def get_user(session: AsyncSession, email: str) -> User:
    """The user with `email`, compared without regard to case (the column is
    citext) and in its canonical form. Raises AccountError for an email that
    could not be stored, and for one with no account."""
    email = canonical_email(email)
    user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None:
        raise AccountError(f"no user with email {email!r}")
    return user


async def create_organization(session: AsyncSession, *, name: str, slug: str) -> Organization:
    name = clean_name(name, "an organization name")
    if SLUG_PATTERN.fullmatch(slug) is None:
        raise AccountError(
            "a slug is 2 to 63 lowercase letters, digits, or hyphens, starting with a letter or digit"
        )
    taken = (
        await session.execute(select(Organization.id).where(Organization.slug == slug))
    ).first()
    if taken is not None:
        raise AccountError(f"the slug {slug!r} is already taken")
    organization = Organization(name=name, slug=slug)
    session.add(organization)
    await session.flush()
    return organization


async def create_user(session: AsyncSession, *, email: str, password: str) -> User:
    email = canonical_email(email)
    taken = (await session.execute(select(User.id).where(User.email == email))).first()
    if taken is not None:
        raise AccountError(f"the email {email!r} already has an account")
    user = User(email=email, password_hash=await hash_password(password))
    session.add(user)
    await session.flush()
    return user


async def add_membership(
    session: AsyncSession, *, organization_id: uuid.UUID, user_id: uuid.UUID, role: str
) -> Membership:
    if role not in MEMBERSHIP_ROLES:
        raise AccountError(f"a role is one of {', '.join(MEMBERSHIP_ROLES)}")
    existing = await session.get(Membership, (organization_id, user_id))
    if existing is not None:
        raise AccountError("the user is already a member of the organization")
    membership = Membership(organization_id=organization_id, user_id=user_id, role=role)
    session.add(membership)
    await session.flush()
    return membership


async def set_password(session: AsyncSession, user: User, password: str) -> int:
    """Replace a user's password and end all of their sessions, so a stolen
    cookie does not outlive the change. Returns how many sessions ended."""
    user.password_hash = await hash_password(password)
    ended = await sessions.delete_user_sessions(session, user.id)
    await session.flush()
    return ended
