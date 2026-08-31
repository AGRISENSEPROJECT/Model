# Precision 4-domain feature engine

## Architecture
Inputs are grouped before scoring:

1. **soil** — pH, NPK, micros, EC, texture, VWC, soil temp, SOM, CEC, slope
2. **weather** — precip, Tmax/Tmin, GDD, RH, solar, ET0, wind, forecast horizon
3. **history** — 3-season rotation, fallow, pest risk, prior amendments, mono-crop streak
4. **economic** — maturity, market price, seed access, labor demand/capacity, market distance

Assembled into a fixed numeric + categorical vector → `artifacts/precision_crop_yield.pkl`.

## Status legend (`feature_provenance` in responses)
- **live** — provided in the request
- **derived** — computed (GDD, ET0, N-fixation credit, mono-crop streak, …)
- **default** — scientific midpoint until sensors/APIs exist
- **pending_api** — reserved for weather/market integrations

## Train
```bash
source .venv/bin/activate
python scripts/train_precision_crop_model.py
```

## Test
- Playground: http://127.0.0.1:5000/playground
- Schema: http://127.0.0.1:5000/feature-schema
- Sample: http://127.0.0.1:5000/sample-request
