"""Session tokens and API keys: how they are made, shaped, and compared.

Both are random secrets with at least 256 bits, so they cannot be guessed and a
plain SHA-256 digest is enough to store (a slow hash would only add CPU to
every request). Only the digest is kept; the token itself is shown once.

An API key is `dpk_<prefix>_<secret>`: `prefix` is 12 lowercase base32
characters, stored in the clear to find the row and to show in a list, and
`secret` is `secrets.token_urlsafe(32)`. `parse_api_key` accepts exactly that
shape and nothing else, so a malformed bearer value is refused before any
database query.
"""

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass

SESSION_TOKEN_BYTES = 32
API_KEY_SECRET_BYTES = 32
API_KEY_SCHEME = "dpk"
API_KEY_PREFIX_CHARS = 12
_BASE32_ALPHABET = "abcdefghijklmnopqrstuvwxyz234567"

# token_urlsafe(32) is 43 characters of the URL-safe base64 alphabet. Spelled
# out as ASCII classes: \w and [0-9] would admit other scripts' digits. Matched
# with fullmatch, because `$` would let a trailing newline through.
_API_KEY_PATTERN = re.compile(
    rf"{API_KEY_SCHEME}_(?P<prefix>[a-z2-7]{{{API_KEY_PREFIX_CHARS}}})_[A-Za-z0-9_-]{{43}}"
)


@dataclass(frozen=True)
class MintedApiKey:
    """A new API key. `token` is the only copy of the secret: show it once,
    store `prefix` and `token_digest`."""

    token: str
    prefix: str
    token_digest: bytes


def new_session_token() -> str:
    """A browser session token: 256 random bits, URL-safe."""
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def mint_api_key() -> MintedApiKey:
    """A new random API key in the `dpk_<prefix>_<secret>` format."""
    prefix = "".join(secrets.choice(_BASE32_ALPHABET) for _ in range(API_KEY_PREFIX_CHARS))
    token = f"{API_KEY_SCHEME}_{prefix}_{secrets.token_urlsafe(API_KEY_SECRET_BYTES)}"
    return MintedApiKey(token=token, prefix=prefix, token_digest=digest(token))


def parse_api_key(value: str) -> str | None:
    """The prefix of `value` when it is exactly an API key's shape, else None.
    Never raises, whatever the input. Shape only: whether the key exists is a
    lookup (app/auth/api_keys.py)."""
    match = _API_KEY_PATTERN.fullmatch(value)
    return match.group("prefix") if match else None


def digest(token: str) -> bytes:
    """The SHA-256 digest of `token`, 32 bytes: what is stored in place of it.
    Total over any string: a lone surrogate (which cannot come from a real
    header, but a lookup may be handed any cookie text) is encoded with
    `surrogatepass`, which keeps distinct strings distinct and simply finds no
    row, where raising would turn a bad cookie into a server error."""
    return hashlib.sha256(token.encode("utf-8", errors="surrogatepass")).digest()


def digests_match(stored: bytes, presented: bytes) -> bool:
    """Whether two digests are equal, in constant time."""
    return hmac.compare_digest(stored, presented)
