import os

os.environ["APP_PASSWORD"] = "test-password-123"
os.environ["SESSION_SECRET_KEY"] = "test-secret-key"

from app.auth import create_session_token, verify_password, verify_session_token


def test_verify_password_accepts_correct_password():
    assert verify_password("test-password-123") is True


def test_verify_password_rejects_wrong_password():
    assert verify_password("wrong") is False


def test_session_token_round_trips():
    token = create_session_token()
    assert verify_session_token(token) is True


def test_session_token_rejects_tampered_value():
    token = create_session_token()
    tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
    assert verify_session_token(tampered) is False


def test_session_token_rejects_garbage():
    assert verify_session_token("not-a-real-token") is False
