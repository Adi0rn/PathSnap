"""Integration tests for trips listing endpoint."""

from fastapi.testclient import TestClient
from src.api.main import app


def test_list_trips():
    client = TestClient(app)
    response = client.get("/trips")
    assert response.status_code == 200
    data = response.json()
    assert data["partition"] == "student_test"
    assert data["count"] == 10
    assert "00000000" in data["trips"]
    assert "00000007" in data["trips"]
