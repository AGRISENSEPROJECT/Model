# Accuracy snapshot & improvement plan

## Current metrics (your last successful runs)

| System | Metric | Value | Production use? |
|--------|--------|------:|-----------------|
| Soil CNN (validation) | Accuracy | **88.7%** | Yes — primary texture model |
| Soil CNN | Sandy recall | **0.42** | Weak — priority fix |
| Soil CNN | Alluvial / Clayey / Loamy F1 | ~0.88–0.92 | Good |
| Environmental yield RF | R² / MAE | **0.976** / 2.32 | Yes |
| Precision 4-domain RF | R² / MAE | **0.967** / 2.46 | Yes — live crop ranking |
| Legacy 10-class crop RF | Accuracy | **45%** | No — fallback only |
| Soil quality RF | R² = 1.0 | Suspicious | Derived label in CSV |

Label-order bug check: wrong map would give ~1.7% accuracy; correct map gives **88.7%** — fixed.

## How accuracy keeps rising
1. Keep using the app → photos auto-save to `data/retrain/inbox/`.
2. Human-label sandy/hard cases via `/retrain/label`.
3. Every ~90 days (or ≥50 new gold labels) run `scripts/scheduled_retrain.py`.
4. Collect more **sandy** images (biggest gap).
5. Wire real weather/market APIs into the 4-domain vector (replace defaults).
6. Never replace the base dataset with tiny Kaggle-only sets; only augment.

## Challenges now
- Sandy under-represented → biased toward alluvial/clayey.
- Extra Kaggle sets (Black/Red/Laterite) don’t map cleanly to 4 texture classes.
- Many 4-domain features still **default/pending_api** (not live sensors).
- Pseudo-labels can reinforce model mistakes if used without human review.
- CPU-only TF training is slow; no GPU on current Kali host.
- Soil_Quality R²=1.0 means that target isn’t a real independent label.
- Need durable disk + labeling UI + deploy gate (backend/DevOps work).

See also: `docs/BACKEND_RETRAINING.md`
