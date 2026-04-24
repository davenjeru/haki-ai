"""
Clerk session JWT verification.

Clerk signs every session token with RS256 against a per-instance key set
published at ``{frontend_api}/.well-known/jwks.json``. We derive the
Frontend API host from the publishable key (it is literally base64-encoded
into ``pk_test_<...>$`` / ``pk_live_<...>$``), so no extra env var is
required.

The module is deliberately small and dependency-light:

  - ``PyJWT[crypto]`` handles RS256 verification.
  - JWKS is fetched via ``urllib`` and cached in-process keyed by ``kid``.
    A single key rotation invalidates the cache on the next mismatch,
    and the cache has a 1-hour soft TTL so key rotations eventually
    propagate even in long-lived Lambda containers.

Public entry point: :func:`verify_clerk_jwt`. Returns the Clerk user id
on success, ``None`` on any failure — callers treat any ``None`` as
anonymous.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import jwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, PyJWKClientError

logger = logging.getLogger(__name__)


_JWKS_CACHE_TTL_SECONDS = 60 * 60  # 1 hour


@dataclass(frozen=True)
class _ClerkKeySource:
    issuer: str
    jwks_url: str


def _pad_b64(s: str) -> str:
    """Restores missing base64 padding that Clerk strips from the pk suffix."""
    return s + "=" * (-len(s) % 4)


def derive_issuer(publishable_key: str) -> str:
    """
    Extracts the Clerk issuer URL from a publishable key.

    Publishable keys look like ``pk_test_<base64>`` or ``pk_live_<base64>``
    where the base64 body decodes to ``<frontend-api-host>$`` (trailing
    dollar is a separator Clerk uses to sentinel the end). We peel off the
    prefix, unpad, decode, strip the trailing ``$``, and prepend https.

    Raises:
        ValueError: if the key is empty or does not start with ``pk_``.
    """
    key = (publishable_key or "").strip()
    if not key:
        raise ValueError("Clerk publishable key is empty")
    if not (key.startswith("pk_test_") or key.startswith("pk_live_")):
        raise ValueError("Clerk publishable key must start with pk_test_ or pk_live_")
    _, _, body = key.partition("_")       # drop "pk"
    _, _, body = body.partition("_")      # drop env (test|live)
    try:
        decoded = base64.b64decode(_pad_b64(body)).decode("utf-8")
    except Exception as err:  # noqa: BLE001
        raise ValueError(f"Clerk publishable key base64 decode failed: {err}") from err
    host = decoded.rstrip("$").strip()
    if not host:
        raise ValueError("Clerk publishable key did not encode a host")
    return f"https://{host}"


def _key_source(publishable_key: str) -> _ClerkKeySource:
    issuer = derive_issuer(publishable_key)
    return _ClerkKeySource(issuer=issuer, jwks_url=f"{issuer}/.well-known/jwks.json")


# ── JWKS client cache ────────────────────────────────────────────────────────
# A single PyJWKClient handles JWKS fetches with its own signing-key cache
# keyed by ``kid``. We keep one client per publishable key and rebuild it
# every ``_JWKS_CACHE_TTL_SECONDS`` so rotations propagate even on warm
# Lambda containers.

_jwks_lock = threading.Lock()
_jwks_cache: dict[str, tuple[float, PyJWKClient]] = {}


def _get_jwks_client(publishable_key: str) -> tuple[PyJWKClient, str]:
    """Returns (client, issuer) for the given publishable key, cached."""
    source = _key_source(publishable_key)
    with _jwks_lock:
        entry = _jwks_cache.get(publishable_key)
        now = time.time()
        if entry and now - entry[0] < _JWKS_CACHE_TTL_SECONDS:
            return entry[1], source.issuer
        client = PyJWKClient(source.jwks_url, cache_keys=True, lifespan=_JWKS_CACHE_TTL_SECONDS)
        _jwks_cache[publishable_key] = (now, client)
        return client, source.issuer


def reset_cache() -> None:
    """Test hook — clears the module cache so each test starts cold."""
    with _jwks_lock:
        _jwks_cache.clear()


# ── Token extraction ─────────────────────────────────────────────────────────


def extract_bearer(event: dict) -> str | None:
    """
    Pulls the Bearer token out of an API Gateway v2 HTTP event.

    The v2 event lower-cases header names; v1 and hand-rolled local events
    keep the original case. We check both forms so the handler works under
    API Gateway, LocalStack, and :mod:`app.server_local` interchangeably.
    """
    headers = event.get("headers") or {}
    raw = headers.get("authorization") or headers.get("Authorization") or ""
    raw = raw.strip()
    if not raw:
        return None
    if raw.lower().startswith("bearer "):
        return raw[7:].strip() or None
    return None


# ── Verification ─────────────────────────────────────────────────────────────


def verify_clerk_jwt(token: str | None, publishable_key: str) -> str | None:
    """
    Verifies a Clerk session JWT and returns the Clerk user id.

    Returns ``None`` if the token is missing, malformed, expired, issued
    by a different Clerk instance, or fails any signature/claim check.
    Never raises — the handler treats anonymous and invalid-token turns
    identically.

    Args:
        token: the raw JWT (no "Bearer " prefix).
        publishable_key: the Clerk publishable key whose instance we trust.

    Returns:
        The ``sub`` claim (Clerk user id, e.g. ``user_2ab...``) on success,
        otherwise ``None``.
    """
    if not token or not publishable_key:
        return None
    try:
        client, issuer = _get_jwks_client(publishable_key)
    except ValueError as err:
        logger.warning("Clerk issuer derivation failed: %s", err)
        return None

    try:
        signing_key = client.get_signing_key_from_jwt(token).key
    except (PyJWKClientError, urllib.error.URLError) as err:
        logger.warning("Clerk JWKS lookup failed: %s", err)
        return None
    except Exception as err:  # noqa: BLE001
        logger.warning("Clerk JWKS lookup raised: %s", err)
        return None

    try:
        claims = jwt.decode(
            token,
            key=signing_key,
            algorithms=["RS256"],
            issuer=issuer,
            options={
                # Clerk session tokens don't carry an `aud` by default;
                # verification relies on issuer + signature + expiry.
                "verify_aud": False,
                "require": ["exp", "iat", "iss", "sub"],
            },
            leeway=30,  # accept modest clock skew between Lambda and Clerk
        )
    except InvalidTokenError as err:
        logger.info("Clerk JWT rejected: %s", err)
        return None
    except Exception as err:  # noqa: BLE001
        logger.warning("Clerk JWT decode raised: %s", err)
        return None

    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub.strip():
        return None
    return sub.strip()


# ── Small utility: peek at the unverified payload (used by tests) ─────────────


def _peek_unverified_claims(token: str) -> dict:
    """Returns the unverified JWT payload. Never use for auth decisions."""
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    try:
        padded = _pad_b64(parts[1])
        return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception:  # noqa: BLE001
        return {}
