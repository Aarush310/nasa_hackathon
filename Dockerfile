# FarmIQ container image (multi-stage)
#
# Stage 1 turns the 97 MB hourly MERRA-2 zip into a ~1 MB per-grid-cell table,
# so the large zip never ships in the runtime image.
FROM python:3.12-slim AS data
WORKDIR /build
RUN pip install --no-cache-dir "pandas>=2.0" "numpy>=1.26"
COPY farmiq_core.py prepare_data.py combined_europe_data_Test1.csv.zip ./
RUN python prepare_data.py

# Stage 2: the API server.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py farmiq_core.py prepare_data.py cnn_final.h5 scaler.pkl ./
COPY templates/ templates/
COPY static/ static/
COPY --from=data /build/data/grid_latest.csv data/grid_latest.csv

# Non-root; group 0 so it also runs under OpenShift's random-UID policy.
RUN useradd --uid 1001 --gid 0 --no-create-home appuser && \
    chgrp -R 0 /app && chmod -R g=u /app
USER 1001

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8080\")}/healthz')"

# gunicorn instead of Flask's dev server. One worker keeps a single copy of the
# model in memory; threads handle concurrent requests. $PORT is honoured for Cloud Run.
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 app:app
