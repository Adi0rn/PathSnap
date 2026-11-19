"""Integration tests for trip prediction endpoint."""

from fastapi.testclient import TestClient
from src.api.main import app


def test_predict_real_trip():
    client = TestClient(app)
    # Fast test on trip 00000052 (which has only 150 points)
    response = client.post("/predict/00000052")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["trip_id"] == "00000052"
    assert data["n_observations"] > 0
    assert len(data["full_route_arcs"]) > 0
    assert "metrics" in data
    assert 0.0 <= data["metrics"]["observation_accuracy"] <= 1.0
    assert 0.0 <= data["metrics"]["route_continuity"] <= 1.0


def test_predict_nonexistent_trip():
    client = TestClient(app)
    response = client.post("/predict/99999999")
    assert response.status_code == 404
