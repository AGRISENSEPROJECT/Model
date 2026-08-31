# Roadmap — accuracy & product gaps

## Must-do for production soil accuracy
- [ ] Run `python scripts/evaluate_soil_model.py` and archive `artifacts/soil_validation_report.json`
- [ ] Retrain soil CNN with class weights: `python scripts/train_soil_texture.py`
- [ ] Collect more **sandy** images (target ≥250–300 train) to reduce imbalance
- [ ] Human-verify `data/samples/*` labels; remove plant-only photos from soil training

## Environmental Factors (done / next)
- [x] Ingest `crop_yield_dataset.csv` (36.5k rows)
- [x] Train yield + soil-quality RF models
- [x] Wire live crop ranking by predicted yield (10 crops)
- [ ] Add rainfall feature if a richer weather feed is available (dataset has humidity/wind, not rainfall)
- [ ] Investigate Soil_Quality near-perfect fit (may be a derived formula — still useful as a score)
- [ ] Domain-calibrate yield units for local farms (dataset scale is synthetic-like)

## Model / data
- [ ] Train a plant-disease CNN (PlantVillage or local labeled leaves)
- [ ] Wire real weather API (Open-Meteo / OpenWeather)
- [ ] Optional Sentinel-2 NDVI via coordinates
- [ ] Use or remove `data/table/bangladesh_divisions_dataset.csv`

## Engineering
- [ ] Auth + rate limits on `/predict`
- [ ] Upload retention / virus scanning
- [ ] Pin exact dependency versions from a locked install
- [ ] CI: pytest + soil eval smoke on GPU/CPU runner
- [ ] Calibration of yield predictions against local harvest records
