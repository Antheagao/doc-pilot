"""Password hashing: argon2id, off the event loop, at pinned parameters.

The parameters are RFC 9106's second recommended option (argon2id, t=3, 64 MiB,
p=4) named explicitly, so a change to argon2-cffi's own defaults can never
silently change what is stored; tests/test_auth_core.py pins the numbers.
`needs_rehash` is how a stored hash made under older parameters is found at the
next sign-in.

A hash takes 64 MiB and tens of milliseconds, so it runs in the threadpool
(never on the event loop) under a semaphore of HASH_CONCURRENCY: without it a
burst of sign-ins would put the whole threadpool, 64 MiB each, in memory at
once. A sign-in for an unknown or disabled email still verifies against a dummy
hash, so the response time does not say which accounts exist.

A password is put in Unicode normalization form NFKC before it is hashed or
checked (NIST SP 800-63B), so the same password typed as precomposed or
decomposed characters (an "e" plus a combining accent, or "e-acute") is one
password, whatever keyboard or form produced it. Length limits count the
normalized characters. A string that is not valid text (a lone surrogate)
cannot be UTF-8 encoded: hashing refuses it with a ValueError, and checking
it simply does not match.
"""

import asyncio
import secrets
import unicodedata

from argon2 import PasswordHasher, profiles
from argon2.exceptions import InvalidHashError, VerificationError
from starlette.concurrency import run_in_threadpool

PASSWORD_MIN_CHARS = 12
PASSWORD_MAX_CHARS = 128

# Hashes in flight at once. Two keeps a second sign-in from waiting on the
# first while bounding the memory a burst can take to 128 MiB.
HASH_CONCURRENCY = 2

_hasher = PasswordHasher.from_parameters(profiles.RFC_9106_LOW_MEMORY)
_semaphore = asyncio.Semaphore(HASH_CONCURRENCY)

# Built on first use rather than at import, so importing this module (every
# CLI and test does) costs nothing; the lock makes "once" hold under concurrency.
_dummy_hash: str | None = None
_dummy_lock = asyncio.Lock()


def _canonical(password: str) -> str:
    return unicodedata.normalize("NFKC", password)


def _encode(password: str) -> bytes:
    """The UTF-8 bytes of the normalized password. Raises UnicodeEncodeError (a
    ValueError) for text with a lone surrogate."""
    return _canonical(password).encode("utf-8")


def check_password_length(password: str) -> None:
    """Raise ValueError unless `password` is PASSWORD_MIN_CHARS to
    PASSWORD_MAX_CHARS characters long once normalized. The request schema and
    the CLI call it before any hashing; `hash_password` calls it again as the
    backstop."""
    if not PASSWORD_MIN_CHARS <= len(_canonical(password)) <= PASSWORD_MAX_CHARS:
        raise ValueError(
            f"a password is {PASSWORD_MIN_CHARS} to {PASSWORD_MAX_CHARS} characters long"
        )


async def hash_password(password: str) -> str:
    """The encoded argon2id hash of the normalized `password`, in the
    threadpool. Raises ValueError for a password `check_password_length`
    refuses or that cannot be encoded as UTF-8."""
    check_password_length(password)
    try:
        encoded = _encode(password)
    except UnicodeEncodeError:
        raise ValueError("a password cannot contain characters that cannot be encoded as UTF-8") from None
    async with _semaphore:
        return await run_in_threadpool(_hasher.hash, encoded)


async def verify_password(password: str, password_hash: str) -> bool:
    """Whether `password` matches `password_hash`. A password over
    PASSWORD_MAX_CHARS once normalized is refused without hashing it, so a
    long one cannot burn CPU; text that cannot be encoded as UTF-8, and a stored
    hash that is not a valid argon2 hash, never match."""
    if len(_canonical(password)) > PASSWORD_MAX_CHARS:
        return False
    try:
        encoded = _encode(password)
    except UnicodeEncodeError:
        return False
    async with _semaphore:
        try:
            return await run_in_threadpool(_hasher.verify, password_hash, encoded)
        except (VerificationError, InvalidHashError):
            return False


async def verify_dummy_password(password: str) -> None:
    """Spend the time of a real verification and discard the result: the
    sign-in path for an email that is unknown or disabled."""
    global _dummy_hash
    if _dummy_hash is None:
        async with _dummy_lock:
            if _dummy_hash is None:
                _dummy_hash = await hash_password(secrets.token_urlsafe(24))
    await verify_password(password, _dummy_hash)


def needs_rehash(password_hash: str) -> bool:
    """Whether `password_hash` was made under parameters other than the
    current ones, so the caller should store a fresh hash after a successful
    verification. Raises argon2's InvalidHashError for text that is not a hash."""
    return _hasher.check_needs_rehash(password_hash)
