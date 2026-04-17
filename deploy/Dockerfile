# ============================================================
# Taxi Fare Prediction API — Multi-stage Docker build
# ============================================================
# Build:  docker build -t taxi-fare-api deploy/
# Run:    docker run -p 8000:8000 taxi-fare-api
# ============================================================

# ---------- stage 1: build deps ----------
FROM python:3.9-slim AS builder

WORKDIR /build

COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt \
    -r requirements-serve.txt

# ---------- stage 2: runtime ----------
FROM python:3.9-slim AS runtime

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/            src/
COPY configs/        configs/
COPY models/         models/
COPY deploy/server/  deploy/server/
COPY deploy/monitoring/ deploy/monitoring/
COPY deploy/__init__.py deploy/__init__.py

# Create logs directory
RUN mkdir -p logs && chown -R appuser:appuser /app

USER appuser

# Env defaults
ENV APP_ENV=prod \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    MODEL_NAME=catboost \
    MODEL_VERSION=v1 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "deploy.server.rest_server:app", "--host", "0.0.0.0", "--port", "8000"]
