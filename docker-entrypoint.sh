#!/bin/sh
set -eu

# GitHub rejects this ~114 MB file; train it on first boot if missing.
if [ ! -f artifacts/yield_predictor.pkl ]; then
  echo "yield_predictor.pkl missing — training environmental models (1–3 min)"
  python scripts/train_environmental_models.py
fi

exec gunicorn \
  --bind 0.0.0.0:5000 \
  --workers 1 \
  --threads 4 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  wsgi:app
