# Agrisense Model

Flask API + ML backend for **AGRISENSE** — soil texture CNN, crop/yield ranking, fertilizer & irrigation advice, ESP32 sensor ingest, and a continuous retraining pipeline.

**Repo:** https://github.com/AGRISENSEPROJECT/Model  
**Related:** ESP32 firmware lives in the separate `AGRISENSE/` Arduino project (not this repo).

---

## What this service does

| Capability | How |
|------------|-----|
| Soil texture from photo | MobileNetV2 CNN → sandy / loamy / clayey / alluvial |
| Crop ranking | Environmental yield models + precision 4-domain engine |
| Fertilizer / irrigation / nutrients | Rule + model hybrid in `app/services/` |
| ESP32 live readings | `POST /api/sensor/reading` (Modbus probe → server AI) |
| Retrain pipeline | Captures prediction images → human labels → scheduled retrain |

---

## Requirements

- **Python 3.11–3.13** (TensorFlow may not install on 3.14+ yet)
- **~2 GB disk** after clone (includes training images + models)
- Optional: **OpenWeatherMap API key** for live weather when lat/lon is sent
- Optional: **GPU** speeds up soil CNN training only; inference runs on CPU

---

## Clone & first-time setup

```bash
git clone https://github.com/AGRISENSEPROJECT/Model.git
cd Model

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# Edit .env — at minimum set SENSOR_API_KEY (must match ESP32 secrets.h)
```

### Required after clone — regenerate one large model

GitHub blocks files **>100 MB**. This file is **not** in git:

| File | Size | Action |
|------|------|--------|
| `artifacts/yield_predictor.pkl` | ~114 MB | **Run once after clone** (see below) |

Everything else needed to run the API is already in the repo.

```bash
source .venv/bin/activate
python scripts/train_environmental_models.py
# Creates: artifacts/yield_predictor.pkl, soil_quality_predictor.pkl, environmental_model_meta.json
# Takes ~1–3 min on CPU
```

### Run the server

```bash
source .venv/bin/activate
python wsgi.py
# Production: gunicorn wsgi:app --bind 0.0.0.0:5000
```

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:5000 | Web UI |
| http://127.0.0.1:5000/playground | Precision feature playground |
| http://127.0.0.1:5000/apidocs | Swagger API docs |
| http://127.0.0.1:5000/api/health | Health check |

### Quick smoke test

```bash
curl http://127.0.0.1:5000/api/health

pytest tests/ -q
```

---

## Environment variables (`.env`)

Copy from `.env.example`:

```bash
OPENWEATHERMAP_API_KEY=your_key_here    # optional — auto weather from coordinates
SENSOR_API_KEY=agrisense-local        # required for ESP32 uploads; match firmware secrets.h
```

Optional overrides:

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGRISENSE_ARTIFACTS` | `./artifacts` | Model files directory |
| `AGRISENSE_DATA` | `./data` | Datasets |
| `AGRISENSE_UPLOADS` | `./uploads` | Temporary uploaded images |

---

## API endpoints (prefix `/api`)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service health |
| GET | `/feature-schema` | Full precision input catalog |
| GET | `/sample-request` | Example JSON for `/predict` |
| POST | `/predict` | Main analysis (image + soil + weather) |
| POST | `/comprehensive-analyze` | Raw analysis envelope |
| POST | `/sensor/reading` | **ESP32** live sensor JSON (header `X-API-Key`) |
| GET | `/retrain/status` | Retrain inbox counts |
| POST | `/retrain/label` | Human label for captured image |
| POST | `/retrain/reject` | Reject bad capture |

### ESP32 → server (colleague reference)

Firmware POSTs to: `http://<server-ip>:5000/api/sensor/reading`  
Header: `X-API-Key: <SENSOR_API_KEY>`  
Stored under: `data/device_readings/`

Default server IP in firmware `config.h`: `192.168.0.115` — change to your machine's LAN IP.

---

## Project layout

```
Model/
├── app/
│   ├── api/routes.py       # HTTP routes
│   ├── config.py           # Paths, crop list, class indices
│   ├── features/           # 4-domain precision vector (soil/weather/history/economic)
│   ├── knowledge/          # Crop profiles, fertilizer, irrigation rules
│   └── services/           # ML inference, weather, sensor ingest, retrain store
├── artifacts/              # Trained models (see table below)
├── data/
│   ├── train/              # Soil CNN training images (~1k+)
│   ├── validation/         # Soil CNN validation set (292 images)
│   ├── environmental/      # crop_yield_dataset.csv (36,520 rows)
│   ├── retrain/            # Production capture inbox + manifest
│   ├── samples/            # Curated reference photos
│   └── device_readings/    # ESP32 JSON logs
├── docs/                   # Detailed guides (read these!)
├── scripts/                # Train / evaluate / retrain jobs
├── static/ + templates/    # Web UI
├── tests/                  # pytest suite
├── wsgi.py                 # App entrypoint
├── requirements.txt
├── Procfile                # Heroku/Railway: gunicorn wsgi:app
└── .env.example
```

---

## Artifacts in git (what you get on clone)

| File | In git? | Role |
|------|---------|------|
| `soil_texture_mobilenetv2.keras` | ✅ | Soil texture CNN (~23 MB) |
| `soil_class_indices.json` | ✅ | Label index map (alluvial=0 … sandy=3) |
| `crop_predictor.pkl` | ✅ | Legacy RF crop model (~80 MB) |
| `scaler.pkl`, `label_encoder.pkl` | ✅ | RF preprocessing |
| `precision_crop_yield.pkl` | ✅ | 4-domain precision ranker |
| `precision_feature_meta.json` | ✅ | Precision model metadata |
| `soil_quality_predictor.pkl` | ✅ | Environmental quality score |
| `environmental_model_meta.json` | ✅ | Env model metadata |
| `yield_predictor.pkl` | ❌ | **Regenerate** — `train_environmental_models.py` |
| `soil_validation_report.json` | ✅ | CNN eval metrics (~88.7% accuracy) |

---

## Training & evaluation scripts

Run from repo root with venv active:

```bash
# Environmental yield + quality models (REQUIRED after clone)
python scripts/train_environmental_models.py

# Precision 4-domain crop ranker
python scripts/train_precision_crop_model.py

# Legacy crop Random Forest
python scripts/train_crop_rf.py

# Soil CNN — retrain with class weights (needs TensorFlow)
python scripts/train_soil_texture.py --epochs 20

# Evaluate soil CNN on data/validation/
python scripts/evaluate_soil_model.py

# Scheduled retrain gate (cron-friendly)
python scripts/scheduled_retrain.py --check-only
python scripts/scheduled_retrain.py --force
```

---

## How `/predict` works (live path)

1. Optional soil **image** → MobileNetV2 → texture class  
2. Soil sensors (N, P, K, pH, moisture, EC…) + **weather** (manual or OpenWeatherMap from coordinates)  
3. **Precision engine** (`precision_crop_yield.pkl`) ranks 10 crops by predicted yield  
4. Fertilizer, irrigation, nutrient analysis from knowledge modules  
5. If image provided → archived to `data/retrain/inbox/` for future retraining  

Soil class indices **must** match alphabetical Keras order — see `artifacts/soil_class_indices.json`.

---

## Documentation (read next)

| Doc | Content |
|-----|---------|
| [docs/ROADMAP.md](docs/ROADMAP.md) | Accuracy gaps & priorities |
| [docs/ACCURACY_AND_ROADMAP.md](docs/ACCURACY_AND_ROADMAP.md) | Known model limitations |
| [docs/PRECISION_FEATURES.md](docs/PRECISION_FEATURES.md) | 4-domain feature engine |
| [docs/BACKEND_RETRAINING.md](docs/BACKEND_RETRAINING.md) | Retrain pipeline for DevOps |

---

## Current priorities for development

1. Run `evaluate_soil_model.py` after any CNN retrain; keep `soil_validation_report.json` updated  
2. Collect more **sandy** soil images (class is under-represented)  
3. Wire ESP32 sensor path end-to-end (`/api/sensor/reading` + firmware)  
4. Improve sandy-class accuracy and validate on real farm photos  
5. See [docs/ROADMAP.md](docs/ROADMAP.md) for full backlog  

---

## Deploy notes

- **Procfile:** `web: gunicorn wsgi:app`
- Mount `data/retrain/` on persistent storage in production (do not wipe on redeploy)
- Set `SENSOR_API_KEY` and `OPENWEATHERMAP_API_KEY` in host environment
- After deploy, run `train_environmental_models.py` on the server **or** copy `yield_predictor.pkl` from a trusted build machine

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: yield_predictor.pkl` | Run `python scripts/train_environmental_models.py` |
| TensorFlow install fails | Use Python 3.11–3.13 |
| ESP32 gets 401 | Match `SENSOR_API_KEY` in `.env` and firmware `secrets.h` |
| All crops N/A / low confidence | Send real NPK + pH + moisture; check `production_mode` in payload |
| Soil CNN wrong class names | Verify `soil_class_indices.json` matches training order |

---

## Contact / handoff

- **Company:** Velora Tech Labs LTD (AGRISENSE product)
- **Lead:** Saly Nelson IRASUBIZA — irasubizasalynelson@gmail.com
- **Stack:** Flask 3, scikit-learn, TensorFlow 2.x, MobileNetV2

When in doubt: start server → open `/apidocs` → run `pytest` → read `docs/ROADMAP.md`.
