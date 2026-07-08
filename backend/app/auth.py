"""Single shared-password auth: on successful login, issues a signed,
timestamped session token (no server-side session storage needed since the
app is stateless). The token is opaque to the client and expires after 7
days."""
import hmac
import os

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

_SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
_SESSION_PAYLOAD = "authenticated"

_REQUIRED_ENV_VARS = ("APP_PASSWORD", "SESSION_SECRET_KEY")
_missing = [name for name in _REQUIRED_ENV_VARS if name not in os.environ]
if _missing:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(_missing)}"
    )


def _serializer() -> URLSafeTimedSerializer:
    secret_key = os.environ["SESSION_SECRET_KEY"]
    return URLSafeTimedSerializer(secret_key)


def verify_password(password: str) -> bool:
    return hmac.compare_digest(password, os.environ["APP_PASSWORD"])


def create_session_token() -> str:
    return _serializer().dumps(_SESSION_PAYLOAD)


def verify_session_token(token: str) -> bool:
    try:
        payload = _serializer().loads(token, max_age=_SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return False
    return payload == _SESSION_PAYLOAD
