# Backend handoff — continuous soil-model improvement (scheduled retrain)

## Goal
Every farm soil photo used in crop prediction is **kept on the server** and later used to retrain the soil CNN (default every **90 days**, or earlier if enough human labels arrive).

This is **not** magic self-learning like ChatGPT chat memory. It is a controlled MLOps loop:

1. Capture production images  
2. (Optional) human verify labels  
3. Merge into training set  
4. Retrain + evaluate  
5. Deploy new `.keras` artifact  

## Folder layout (already created by the Model service)
```
data/retrain/
  inbox/                 # EVERY prediction image + {sample_id}.json sidecar
  pseudo_labeled/        # auto copy if confidence ≥ 0.85 (weak labels)
    alluvial|clayey|loamy|sandy/
  human_labeled/         # gold labels after agronomist/API correction
    alluvial|clayey|loamy|sandy/
  rejected/
  manifest.jsonl         # append-only audit log
  retrain_state.json     # last retrain timestamp + metrics
```

## What the Model API already does
| Endpoint | Purpose |
|----------|---------|
| `POST /predict` (with image) | Runs CNN; archives image → `inbox/`; returns `retrain_capture.sample_id` |
| `GET /retrain/status` | Counts inbox / human / pseudo labels |
| `POST /retrain/label` | `{sample_id, human_label, notes?}` → gold `human_labeled/` |
| `POST /retrain/reject` | Mark bad photos (blurry / not soil) |

## What YOU (backend / DevOps) must implement

### 1. Persistent storage
- Mount `data/retrain/` on durable disk (S3/EBS/NFS). **Do not wipe** on redeploy.
- Keep `uploads/` as short-lived; retrain inbox is the long-term store.
- Budget: ~0.5–5 MB/image × expected monthly predictions.

### 2. Human labeling UI (mobile/web admin)
- List unverified `inbox` samples (read status API or DB sync from `manifest.jsonl`).
- Show image + model prediction + confidence.
- Agronomist picks true class → call `POST /retrain/label`.
- Reject garbage → `POST /retrain/reject`.
- **Do not auto-train only on model predictions** without review (causes confirmation bias). Pseudo labels are optional and weaker.

### 3. Cron / scheduler (example: daily check, retrain at most every 90 days)
```cron
# Every day at 02:15 — gate decides whether retrain is due
15 2 * * * cd /path/to/Model && .venv/bin/python scripts/scheduled_retrain.py >> /var/log/agrisense-retrain.log 2>&1
```
Manual force (after a labeling campaign):
```bash
.venv/bin/python scripts/scheduled_retrain.py --force
.venv/bin/python scripts/scheduled_retrain.py --check-only
```

Env knobs:
- `AGRISENSE_RETRAIN_INTERVAL_SECONDS` (default `7776000` = 90 days)
- `AGRISENSE_RETRAIN_MIN_NEW_LABELED` (default `50`)
- `AGRISENSE_PSEUDO_MIN_CONFIDENCE` (default `0.85`)

### 4. Deploy after retrain
- Job writes new `artifacts/soil_texture_mobilenetv2.keras` + `soil_class_indices.json` + `soil_validation_report.json`.
- Restart/reload API workers so they load the new weights (or hot-reload if you add it).
- Alert if val accuracy **drops** vs previous report — do not auto-promote a worse model (add approval gate in CI/CD).

### 5. Privacy / consent
- Soil photos may include GPS/metadata — strip EXIF if required by policy.
- Document retention (e.g. 24 months) and farmer consent in ToS.

### 6. Optional DB mirror
If the main backend is Nest/Django/etc., mirror `sample_id`, paths, predicted/human labels in Postgres for the labeling UI; keep filesystem as source of training truth.

## Success criteria
- After 3 months: ≥50 human-verified images (ideally balanced, **especially sandy**).
- Scheduled job runs, promotes labels, retrains, `evaluate_soil_model.py` accuracy ≥ previous.
- API `/retrain/status` shows growing `human_labeled_total`.
