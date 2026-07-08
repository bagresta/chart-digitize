import os

os.environ["APP_PASSWORD"] = "test-password-123"
os.environ["SESSION_SECRET_KEY"] = "test-secret-key"

import cv2
from fastapi.testclient import TestClient

from app.main import app
from tests.fixtures import make_line_chart

client = TestClient(app)


def _login_client():
    response = client.post("/api/login", json={"password": "test-password-123"})
    assert response.status_code == 200
    assert "session" in response.cookies
    return response.cookies


def test_login_with_correct_password_sets_session_cookie():
    _login_client()


def test_login_with_wrong_password_returns_401():
    response = client.post("/api/login", json={"password": "wrong"})
    assert response.status_code == 401


def test_upload_without_session_returns_401():
    image, _ = make_line_chart()
    success, buf = cv2.imencode(".png", image)
    response = client.post("/api/upload", files={"file": ("chart.png", buf.tobytes(), "image/png")})
    assert response.status_code == 401


def test_upload_with_session_returns_extracted_series():
    cookies = _login_client()
    image, _ = make_line_chart()
    success, buf = cv2.imencode(".png", image)

    response = client.post(
        "/api/upload",
        files={"file": ("chart.png", buf.tobytes(), "image/png")},
        cookies=dict(cookies),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chart_type"] == "line"
    assert len(body["series"]) == 1
    assert len(body["series"][0]["points"]) > 5
    assert "overlay_image_base64" in body


def test_export_csv_requires_session():
    response = client.post("/api/export/csv", json={"series": []})
    assert response.status_code == 401


def test_export_csv_returns_file():
    cookies = _login_client()
    response = client.post(
        "/api/export/csv",
        json={"series": [{"name": "Series A", "points": [[1.0, 2.0]]}]},
        cookies=dict(cookies),
    )
    assert response.status_code == 200
    assert "series,x,y" in response.text
