ARG PYTHON_IMAGE=python:3.11-slim
ARG API_BASE_IMAGE=api-deps-local

# Local dependency stage. Can be published and reused as external base image.
FROM ${PYTHON_IMAGE} AS api-deps-local

WORKDIR /tmp

COPY requirements-api.txt /tmp/requirements-api.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r /tmp/requirements-api.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip /tmp/*

# Final runtime image.
# By default it uses the local dependency stage.
# In CD we can pass API_BASE_IMAGE=<registry>/<repo>-api-base:<tag>
FROM ${API_BASE_IMAGE} AS runtime

WORKDIR /app

COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/

RUN mkdir -p /app/models /app/logs /app/feature_store

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
