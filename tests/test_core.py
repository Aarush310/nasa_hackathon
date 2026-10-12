"""Unit tests for data prep, location lookup and output interpretation (no TensorFlow needed)."""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farmiq_core import FEATURES, build_grid, interpret, nearest_cell  # noqa: E402


@pytest.fixture
def grid():
    return pd.DataFrame({"lat": [36.0, 53.5, 53.5, 60.0], "lon": [-6.25, -6.25, 10.0, 20.0]})


class TestNearestCell:
    def test_finds_true_nearest(self, grid):
        cell, dist = nearest_cell(grid, 53.35, -6.26)  # Dublin
        assert (cell["lat"], cell["lon"]) == (53.5, -6.25)
        assert dist < 0.2

    def test_regression_for_star_two_bug(self, grid):
        # The old code used (d * 2) instead of (d ** 2), which made sqrt() return NaN for
        # points south/west of the click and picked the wrong cell. Points on every side must work.
        cell, _ = nearest_cell(grid, 37.0, -6.0)
        assert (cell["lat"], cell["lon"]) == (36.0, -6.25)
        cell, _ = nearest_cell(grid, 59.0, 19.0)
        assert (cell["lat"], cell["lon"]) == (60.0, 20.0)

    def test_longitude_scaled_by_latitude(self):
        # At 60°N one degree of longitude is half a degree of latitude.
        g = pd.DataFrame({"lat": [60.0, 61.2], "lon": [12.0, 10.0]})
        cell, _ = nearest_cell(g, 60.0, 10.0)
        assert cell["lon"] == 12.0  # 2° lon east ≈ 1° away, closer than 1.2° north


class TestInterpret:
    def test_labels(self):
        out = interpret([2.5, 0.7, 0.98, 0.002, 0.83])
        assert out == {"Water Availability": "Good", "Soil Condition": "Optimal",
                       "Crop Health": "Healthy", "Frost Risk": "Low", "Yield Quality": "83.00%"}

    @pytest.mark.parametrize("crop,label", [(1.004, "Healthy"), (0.01, "Average"), (-1.03, "At Risk"),
                                            (0.5, "Healthy"), (-0.5, "At Risk")])
    def test_crop_health_thresholds_not_equality(self, crop, label):
        # The model is a regressor — outputs like 0.996 must still read as "Healthy".
        assert interpret([1, 1, crop, 0, 0.5])["Crop Health"] == label

    def test_frost_and_water_bands(self):
        assert interpret([0.5, 0.2, 0, 0.9, 0.5])["Frost Risk"] == "High"
        assert interpret([0.5, 0.2, 0, 0.9, 0.5])["Water Availability"] == "Poor"
        assert interpret([1.5, 0.2, 0, 0.9, 0.5])["Water Availability"] == "Average"

    def test_yield_is_clamped(self):
        assert interpret([1, 1, 0, 0, 1.26])["Yield Quality"] == "100.00%"
        assert interpret([1, 1, 0, 0, -0.1])["Yield Quality"] == "0.00%"


def test_build_grid_means_last_24_hours(tmp_path):
    rows = []
    for hour in range(30):
        for lat, lon in [(50.0, 0.0), (50.5, 0.625)]:
            row = {f: 1.0 for f in FEATURES}
            row.update(time=f"2024-09-01 {hour:02d}:30:00" if hour < 24 else f"2024-09-02 {hour - 24:02d}:30:00",
                       lat=lat, lon=lon, TSURF=float(hour))
            rows.append(row)
    csv = pd.DataFrame(rows)[["time"] + FEATURES].to_csv(index=False)
    path = tmp_path / "data.csv.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data.csv", csv)
        zf.writestr("__MACOSX/._data.csv", "junk")  # macOS artefact like the real file has

    g = build_grid(path, hours=24, chunksize=7)
    assert len(g) == 2
    assert g["TSURF"].iloc[0] == pytest.approx(sum(range(6, 30)) / 24)  # hours 6..29
    assert g["period_end"].iloc[0] == "2024-09-02 05:30:00"
