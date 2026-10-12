"""
FarmIQ — Flask API serving crop-condition predictions from NASA MERRA-2 data.

GET  /           Map UI (Leaflet)
POST /predict    {"latitude": 53.35, "longitude": -6.26} -> five farm indicators
GET  /healthz    Liveness/readiness probe
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from farmiq_core import FEATURES, MAX_DISTANCE_DEG, interpret, nearest_cell

BASE_DIR = Path(__file__).resolve().parent
GRID_PATH = BASE_DIR / "data" / "grid_latest.csv"

app = Flask(__name__)
CORS(app)  # allow the map UI to be hosted separately from the API

# ── Load model, scaler and data once at startup ──
model = tf.keras.models.load_model(BASE_DIR / "cnn_final.h5", compile=False)
scaler = joblib.load(BASE_DIR / "scaler.pkl")

if not GRID_PATH.exists():
    from prepare_data import main as prepare
    prepare(BASE_DIR / "combined_europe_data_Test1.csv.zip", GRID_PATH)
grid = pd.read_csv(GRID_PATH)


def predict_conditions(lat: float, lon: float) -> dict:
    cell, distance = nearest_cell(grid, lat, lon)

    # Same preprocessing as training: StandardScaler, then shape (batch, steps=1, features=10).
    x = pd.DataFrame([cell[FEATURES].astype(float).to_numpy()], columns=FEATURES)
    x_scaled = scaler.transform(x).reshape(1, 1, len(FEATURES))
    pred = model.predict(x_scaled, verbose=0)[0]

    return {
        "prediction": interpret(pred),
        "location": {
            "grid_lat": float(cell["lat"]),
            "grid_lon": float(cell["lon"]),
            "distance_deg": round(distance, 3),
            "period": f"{cell['period_start']} to {cell['period_end']} (UTC, 24 h mean)",
        },
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        lat = float(data["latitude"])
        lon = float(data["longitude"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Send JSON with numeric 'latitude' and 'longitude'."}), 400
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({"error": "Latitude/longitude out of range."}), 400

    result = predict_conditions(lat, lon)
    if result["location"]["distance_deg"] > MAX_DISTANCE_DEG:
        return jsonify({"error": "Location is outside the dataset's coverage (Europe, 36-71°N, 10°W-60°E).",
                        "location": result["location"]}), 422
    return jsonify(result)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "grid_cells": int(len(grid))})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
