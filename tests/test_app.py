"""API tests against the real model. Skipped automatically if TensorFlow isn't installed
(CI runs these inside the container job instead)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as farmiq  # noqa: E402

client = farmiq.app.test_client()


def test_health():
    assert client.get("/healthz").get_json()["grid_cells"] > 6000


def test_predict_dublin():
    body = client.post("/predict", json={"latitude": 53.35, "longitude": -6.26}).get_json()
    assert set(body["prediction"]) == {"Water Availability", "Soil Condition", "Crop Health",
                                       "Frost Risk", "Yield Quality"}
    assert (body["location"]["grid_lat"], body["location"]["grid_lon"]) == (53.5, -6.25)


def test_different_places_use_different_cells():
    a = client.post("/predict", json={"latitude": 40.4, "longitude": -3.7}).get_json()["location"]
    b = client.post("/predict", json={"latitude": 59.3, "longitude": 18.1}).get_json()["location"]
    assert (a["grid_lat"], a["grid_lon"]) != (b["grid_lat"], b["grid_lon"])


def test_outside_coverage():
    assert client.post("/predict", json={"latitude": -33.9, "longitude": 151.2}).status_code == 422


@pytest.mark.parametrize("payload", [{}, {"latitude": "x", "longitude": 1}, {"latitude": 95, "longitude": 0}])
def test_bad_input(payload):
    assert client.post("/predict", json=payload).status_code == 400


def test_cors_enabled():
    r = client.post("/predict", json={"latitude": 53.35, "longitude": -6.26}, headers={"Origin": "https://example.com"})
    assert r.headers.get("Access-Control-Allow-Origin") in ("*", "https://example.com")
