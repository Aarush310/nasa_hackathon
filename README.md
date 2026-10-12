# 🌾 FarmIQ — Data-Driven Farming from NASA Satellite Data

[![CI](https://github.com/Aarush310/nasa_hackathon/actions/workflows/ci.yml/badge.svg)](https://github.com/Aarush310/nasa_hackathon/actions/workflows/ci.yml)

**NASA Space Apps Challenge 2024** — built in 48 hours by team **Stellar Crew**: Mamtha M Vathar, Ramya T Gowda, Emmanuella Nyamekye, Akash K R and Aarush Prasad.

Farmers in the EU deal with unpredictable soil moisture, frost and water availability. FarmIQ turns NASA **MERRA-2** Earth-observation data into five simple indicators for any point on a map of Europe: **water availability, soil condition, crop health, frost risk and yield quality**.

Click anywhere on the map → the backend finds the nearest MERRA-2 grid cell → a neural network predicts the five indicators → the sidebar shows them as labels and bars.

---

## How it works

```
NASA MERRA-2 (.nc4 → CSV)                      Leaflet map (click / geolocation)
2.3M hourly rows, 6,480 grid cells                         │  POST /predict {lat, lon}
        │                                                  ▼
prepare_data.py ── 24 h mean per grid cell ──▶ nearest grid cell (cos-lat scaled distance)
                                                           │
                                         StandardScaler → CNN (TensorFlow/Keras)
                                                           │  5 continuous indices
                                                   thresholds → labels → JSON
```

### Data
MERRA-2 land-surface variables over Europe (36–71°N, 10°W–60°E) on the native 0.5° × 0.625° grid, hourly from 18 Aug to 1 Sep 2024 (`combined_europe_data_Test1.csv.zip`):

| Variable | Meaning |
|---|---|
| `EVLAND` | Evaporation from land |
| `PRECTOTLAND` | Total precipitation over land |
| `GWETTOP` / `GWETROOT` / `GWETPROF` | Soil wetness: surface, root zone, full profile |
| `LAI` | Leaf area index |
| `TSOIL1` / `TSURF` | Soil temperature (layer 1) and surface temperature, in Kelvin |

The app serves predictions from the **mean of the most recent 24 hours** for each grid cell (`data/grid_latest.csv`, pre-computed by `prepare_data.py`).

### Model
`cnn_final.h5` (Keras), trained with MSE loss and Adam (lr 1e-3) on the 8 variables plus lat/lon, standardized with `scaler.pkl`:

```
Input (1 × 10) → Conv1D(64, kernel 1, ReLU) → MaxPool1D(1) → Flatten → Dense(64, ReLU) → Dense(5, linear)
```

Outputs are five continuous indices, mapped to labels for display:

| Output | Label rule |
|---|---|
| Water Availability Index | > 2.16 Good · > 0.90 Average · else Poor |
| Soil Moisture Index | > 0.5 Optimal · else Dry |
| Crop Health Index (trained on −1/0/1) | ≥ 0.5 Healthy · ≤ −0.5 At Risk · else Average |
| Frost Risk Indicator (trained on 0/1) | ≥ 0.5 High · else Low |
| Yield Quality Index | shown as a 0–100 % bar |

> Design note: with a sequence length of 1 and a kernel size of 1, the Conv1D layer acts as a per-feature dense layer, so the network behaves like a small multilayer perceptron.

---

## Run locally

```bash
git clone https://github.com/Aarush310/nasa_hackathon.git
cd nasa_hackathon
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py                    # http://localhost:8080
```

`data/grid_latest.csv` is included; to rebuild it from the raw zip run `python prepare_data.py`.

## Run with Docker

```bash
docker build -t farmiq .
docker run -p 8080:8080 farmiq   # http://localhost:8080
```

The multi-stage build pre-computes the grid table in a throwaway stage, so the 97 MB dataset zip isn't shipped in the final image. The container runs under **gunicorn** as a non-root user (OpenShift random-UID compatible), honours `$PORT` (Cloud Run), and has a `/healthz` health check.

## API

```bash
curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' \
     -d '{"latitude": 53.35, "longitude": -6.26}'
```

```json
{
  "prediction": {"Water Availability": "Good", "Soil Condition": "Optimal", "Crop Health": "Average",
                 "Frost Risk": "Low", "Yield Quality": "100.00%"},
  "location": {"grid_lat": 53.5, "grid_lon": -6.25, "distance_deg": 0.15,
               "period": "2024-09-01 00:30:00 to 2024-09-01 23:30:00 (UTC, 24 h mean)"}
}
```

Invalid input returns `400`; points outside the dataset's coverage return `422`.

## Tests & CI

```bash
pip install -r requirements-dev.txt && pytest -v
```

Unit tests cover data preparation, nearest-cell lookup (including a regression test for an earlier distance bug) and output interpretation; API tests run against the real model when TensorFlow is installed. GitHub Actions runs the unit tests, builds the image, starts it as a random non-root UID and checks real predictions for Dublin and Madrid.

## Project structure

```
nasa_hackathon/
├── app.py                 # Flask API + UI routes
├── farmiq_core.py         # data prep, nearest-cell lookup, output labels (no TF)
├── prepare_data.py        # hourly zip → per-cell 24 h mean table
├── cnn_final.h5           # trained Keras model
├── scaler.pkl             # StandardScaler fitted on training data
├── data/grid_latest.csv   # pre-computed grid table (~1 MB)
├── combined_europe_data_Test1.csv.zip   # raw MERRA-2 extract
├── templates/             # Leaflet map UI
├── tests/
├── Dockerfile             # multi-stage, gunicorn, non-root
├── app.yaml               # Google App Engine config
└── stellar_crew.pptx      # hackathon presentation
```

## Changelog (post-hackathon fixes)
- Nearest-point lookup used `(d * 2)` instead of `(d ** 2)`, so it ignored grid cells south/west of the click and often returned the wrong cell. It now uses a proper cos-latitude-scaled distance.
- The Flask app was created twice, which silently removed CORS; fixed.
- Crop health and frost risk compared float outputs with `== 1`, so they almost never showed "Healthy"/"High"; now thresholded at the label midpoints.
- Serves from a pre-computed 24 h mean per grid cell instead of loading 2.3M rows into memory at startup.
- gunicorn instead of Flask's dev server, non-root multi-stage image, input validation, tests and CI.
