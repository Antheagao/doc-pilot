from fastapi.testclient import TestClient

from app.main import app


def test_healthz():
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_the_configured_frontend_origin_only():
    client = TestClient(app)
    preflight = {"Access-Control-Request-Method": "POST"}

    allowed = client.options("/ask", headers={"Origin": "http://localhost:3000", **preflight})
    other = client.options("/ask", headers={"Origin": "https://evil.example", **preflight})

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in other.headers


def test_cors_origins_come_from_the_environment(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("CORS_ORIGINS", '["https://doc-pilot-web-1.us-central1.run.app"]')

    assert Settings().cors_origins == ["https://doc-pilot-web-1.us-central1.run.app"]
