"""
Server-verified signed session identity and security middleware.
Provides HMAC-SHA256 signing and verification for session identity tokens,
cookie/header extraction, and CSRF defense for cookie-authenticated mutations.
"""
import hmac
import hashlib
import logging
from typing import Optional
from fastapi import Request, Response, HTTPException, status

from app.config import settings

logger = logging.getLogger("app.security.session")

COOKIE_NAME = "app_session_token"
HEADER_NAME = "X-Session-Token"
MIN_KEY_LENGTH = 32
INSECURE_SECRET_PATTERNS = ("change-me", "secret", "default", "replace-with", "admin", "1234")


def get_signing_secret() -> bytes:
    """
    Returns the validated session signing secret from configuration.
    Fails safely if missing, too short, or matching known insecure patterns.
    """
    raw_key = (settings.session_secret_key or "").strip()
    if not raw_key or len(raw_key) < MIN_KEY_LENGTH:
        raise RuntimeError(
            "SESSION_SECRET_KEY is missing or too short (minimum 32 characters required). "
            "Configure a secure random hex key in .env. Generate with: "
            "python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    key_low = raw_key.lower()
    if any(pat in key_low for pat in INSECURE_SECRET_PATTERNS):
        raise RuntimeError(
            "SESSION_SECRET_KEY is set to an insecure placeholder. "
            "Configure a dedicated random key in .env using: "
            "python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return raw_key.encode("utf-8")


def sign_session_id(session_id: str) -> str:
    """Signs a session ID using HMAC-SHA256, returning '{session_id}.{hex_digest}'."""
    clean_id = (session_id or "").strip()
    if not clean_id:
        raise ValueError("Cannot sign an empty session_id.")
    secret = get_signing_secret()
    sig = hmac.new(secret, clean_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{clean_id}.{sig}"


def verify_session_token(token: Optional[str]) -> Optional[str]:
    """
    Validates a signed session token. Returns the session_id if valid, else None.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not token or not isinstance(token, str) or "." not in token:
        return None
    parts = token.strip().rsplit(".", 1)
    if len(parts) != 2:
        return None
    session_id, signature = parts[0], parts[1]
    if not session_id or not signature:
        return None
    try:
        secret = get_signing_secret()
        expected = hmac.new(secret, session_id.encode("utf-8"), hashlib.sha256).hexdigest()
        if hmac.compare_digest(signature, expected):
            return session_id
    except Exception as e:
        logger.warning(f"Session token verification error: {e}")
        return None
    return None


def extract_session_token(request: Request) -> Optional[str]:
    """Extracts signed session token from Header (X-Session-Token) or Cookie (app_session_token)."""
    # 1. Deliberately supported header has priority for programmatic/API clients
    hdr = request.headers.get(HEADER_NAME)
    if hdr and hdr.strip():
        return hdr.strip()
    # 2. Cookie for browser clients
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie and cookie.strip():
        return cookie.strip()
    return None


def get_verified_session(request: Request) -> Optional[str]:
    """
    Resolves the authenticated session ID strictly from verified token/cookie.
    Returns None if missing or invalid. Never falls back to request body/query params.
    """
    token = extract_session_token(request)
    if not token:
        return None
    return verify_session_token(token)


def require_verified_session(request: Request) -> str:
    """
    Resolves authenticated session ID or raises HTTP 401 Unauthorized.
    """
    session_id = get_verified_session(request)
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: a valid signed session token or cookie is required.",
        )
    return session_id


def set_session_cookie(response: Response, session_id: str, request: Optional[Request] = None) -> str:
    """
    Signs session_id and attaches both HttpOnly Lax cookie and X-Session-Token header to response.
    Sets secure=True dynamically when request is served over HTTPS or x-forwarded-proto is https.
    Returns the signed token string.
    """
    token = sign_session_id(session_id)
    is_secure = False
    if request:
        proto = request.headers.get("x-forwarded-proto", "").lower()
        is_secure = (request.url.scheme == "https") or (proto == "https")
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=is_secure,
        path="/",
    )
    response.headers[HEADER_NAME] = token
    return token


def verify_csrf_protection(request: Request) -> None:
    """
    CSRF defense for cookie-authenticated state-changing operations.
    If authenticated via cookie without explicit custom header, verifies Sec-Fetch-Site or Origin.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    # If authenticated via custom header, browser CORS preflight rules apply
    if request.headers.get(HEADER_NAME):
        return
    # If authenticated via cookie
    if COOKIE_NAME in request.cookies:
        sec_fetch_site = request.headers.get("sec-fetch-site")
        if sec_fetch_site and sec_fetch_site not in ("same-origin", "same-site", "none"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF validation failed: cross-origin state mutation is forbidden.",
            )
