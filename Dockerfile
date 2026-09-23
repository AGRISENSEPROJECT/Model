FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.prod.txt .
RUN pip install --upgrade pip && pip install -r requirements.prod.txt

COPY wsgi.py docker-entrypoint.sh ./
COPY app ./app
COPY artifacts ./artifacts
COPY data ./data
COPY scripts ./scripts
COPY templates ./templates
COPY static ./static
COPY public ./public
COPY .env.example ./

RUN chmod +x docker-entrypoint.sh \
    && mkdir -p uploads artifacts \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5000/api/health >/dev/null || exit 1

# Single worker: TensorFlow + Keras models are memory-heavy on a 4GB VPS.
ENTRYPOINT ["./docker-entrypoint.sh"]
