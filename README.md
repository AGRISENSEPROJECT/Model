# Agrisense Model

Production-oriented soil texture classification + ML crop recommendation API.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Retrain crop RF (already done in artifacts/)
python scripts/train_crop_rf.py

# Evaluate soil CNN (requires TensorFlow + validation images)
python scripts/evaluate_soil_model.py

# Optional: retrain soil CNN with class weights
python scripts/train_soil_texture.py --epochs 20

# Run API + UI
python wsgi.py
# or: gunicorn wsgi:app
```

Open http://127.0.0.1:5000 — Swagger at `/apidocs`.

## Layout

```
app/                 Flask package (API, services, knowledge)
artifacts/           .keras + .pkl + soil_class_indices.json
data/train|validation  Soil CNN datasets
data/samples/        Curated / provisional soil photos by class
scripts/             Train + evaluate + classify helpers
static/              CSS / JS
templates/           Web UI
tests/               Unit tests (no TF required for most)
wsgi.py              Gunicorn entrypoint
```

## Accuracy fixes shipped

1. **Keras label map** — inference uses alphabetical indices (`alluvial=0 … sandy=3`) via `artifacts/soil_class_indices.json`.
2. **Class weights** — `scripts/train_soil_texture.py` applies balanced class weights for sandy under-sampling.
3. **RF live path** — `/predict` uses `crop_predictor.pkl` with LabelEncoder + StandardScaler features (not one-hot, not rule score).

## Environmental Factors dataset

`data/environmental/crop_yield_dataset.csv` (36,520 rows) drives production crop ranking:

- Features: Soil_Type, Crop_Type, Soil_pH, Temperature, Humidity, Wind_Speed, N, P, K
- Targets: Crop_Yield, Soil_Quality
- Train: `python scripts/train_environmental_models.py`
- Artifacts: `artifacts/yield_predictor.pkl`, `soil_quality_predictor.pkl`
- Holdout (current train): yield **R² ≈ 0.98**, MAE ≈ 2.3 (vs ~22 baseline)

`/predict` ranks all 10 crops by **predicted yield** under current conditions (falls back to the older synthetic RF if env models are missing).


## What's still missing / next improvements

See the "Gaps" section in the chat response, or `docs/ROADMAP.md`.
