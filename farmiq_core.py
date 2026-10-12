"""
farmiq_core.py — data preparation, location lookup and prediction interpretation.

Kept free of TensorFlow and Flask so it can be unit-tested quickly.

Data: NASA MERRA-2 land-surface variables over Europe on the native
0.5° (lat) x 0.625° (lon) grid, hourly, 18 Aug - 1 Sep 2024
(combined_europe_data_Test1.csv.zip, 2.3M rows, 6,480 grid cells).
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Model input features, in the order the scaler and CNN were trained on.
FEATURES = [
    "EVLAND",       # evaporation from land (kg m-2 s-1)
    "PRECTOTLAND",  # total precipitation over land (kg m-2 s-1)
    "GWETPROF",     # soil wetness, full profile (0-1)
    "GWETROOT",     # soil wetness, root zone (0-1)
    "GWETTOP",      # soil wetness, surface layer (0-1)
    "LAI",          # leaf area index (m2 m-2)
    "TSOIL1",       # soil temperature, layer 1 (K)
    "TSURF",        # surface skin temperature (K)
    "lat",
    "lon",
]

# Grid spacing of MERRA-2; a click further than ~one cell from any data point
# is outside the dataset's coverage.
MAX_DISTANCE_DEG = 1.0


def build_grid(zip_path: str | Path, hours: int = 24, chunksize: int = 500_000) -> pd.DataFrame:
    """
    Reduce the hourly dataset to one row per grid cell: the mean of each
    variable over the most recent ``hours`` timestamps. Reads the zip in
    chunks so memory stays low.
    """
    with zipfile.ZipFile(zip_path) as zf:
        # Ignore macOS resource-fork entries such as __MACOSX/._file.csv
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        with zf.open(csv_name) as f:
            times = set()
            for chunk in pd.read_csv(f, usecols=["time"], chunksize=chunksize):
                times.update(chunk["time"].unique())
        keep = set(sorted(times)[-hours:])

        parts = []
        with zf.open(csv_name) as f:
            for chunk in pd.read_csv(f, usecols=["time"] + FEATURES, chunksize=chunksize):
                parts.append(chunk[chunk["time"].isin(keep)])

    recent = pd.concat(parts, ignore_index=True)
    grid = recent.groupby(["lat", "lon"], as_index=False)[FEATURES[:-2]].mean()
    grid["period_start"] = min(keep)
    grid["period_end"] = max(keep)
    return grid


def nearest_cell(grid: pd.DataFrame, lat: float, lon: float) -> tuple[pd.Series, float]:
    """
    Closest grid cell to (lat, lon). Longitude differences are scaled by
    cos(latitude) so distances are roughly isotropic on the map.
    Returns (row, distance in degrees of latitude).
    """
    dlat = grid["lat"].to_numpy() - lat
    dlon = (grid["lon"].to_numpy() - lon) * np.cos(np.radians(lat))
    dist = np.sqrt(dlat ** 2 + dlon ** 2)
    i = int(np.argmin(dist))
    return grid.iloc[i], float(dist[i])


def interpret(pred) -> dict:
    """Turn the model's five continuous outputs into the labels the UI shows."""
    water, soil, crop, frost, yield_q = (float(x) for x in pred)
    return {
        # Water Availability Index
        "Water Availability": "Good" if water > 2.16 else "Average" if water > 0.90 else "Poor",
        # Soil Moisture Index (0-1)
        "Soil Condition": "Optimal" if soil > 0.5 else "Dry",
        # Crop Health Index was trained on labels -1 / 0 / 1; the regression output
        # is continuous, so threshold at the midpoints instead of testing equality.
        "Crop Health": "Healthy" if crop >= 0.5 else "At Risk" if crop <= -0.5 else "Average",
        # Frost Risk Indicator was trained on 0 / 1 labels.
        "Frost Risk": "High" if frost >= 0.5 else "Low",
        # Yield Quality Index (0-1), clamped for display.
        "Yield Quality": f"{min(max(yield_q, 0.0), 1.0) * 100:.2f}%",
    }
