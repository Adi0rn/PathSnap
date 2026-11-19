"""Integration tests for FastAPI health endpoint."""

from fastapi.testclient import TestClient
from src.api.main import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["implementation"] == "complete"
    assert data["model_loaded"] is True
