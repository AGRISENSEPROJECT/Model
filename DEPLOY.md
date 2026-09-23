# Model deploy (VPS)

Push to `main` (or **Actions → Deploy Model to VPS → Run workflow**) syncs this repo to the server and rebuilds the Flask/TensorFlow `model` container.

This deploy is for the current package layout (`wsgi:app`, `app/`, `artifacts/`). Do not use the old root `app.py` image.

Target paths on the VPS:

- Code: `/opt/agrisense/Model/`
- Compose: `/opt/agrisense/deploy/` (service name: `model`)
- Model is **internal only** — Nest calls `http://model:5000` on the Docker network

Rebuilds can take a while (TensorFlow + pip). First boot also trains `artifacts/yield_predictor.pkl` if it is missing (GitHub blocks files over 100 MB). The workflow waits for `GET /api/health` before finishing.

## One-time GitHub secrets

Same values as the web / backend repos:

| Secret | Value |
|---|---|
| `VPS_HOST` | VPS public IP |
| `VPS_PORT` | SSH port (example: `222`) |
| `VPS_USER` | Deploy user (example: `root`) |
| `VPS_SSH_KEY` | Deploy private key (full PEM / OpenSSH private key) |

On the VPS, copy `.env.example` to `/opt/agrisense/Model/.env` and set at least:

- `SENSOR_API_KEY` — must match the ESP32 firmware
- `OPENWEATHERMAP_API_KEY` — optional; Open-Meteo is the fallback
- `SECRET_KEY` — change from the development default

## After secrets are set

```bash
git push origin main
```

Or run the workflow manually from the Actions tab.

## Verify

```bash
ssh -p 222 USER@VPS_IP 'docker exec agrisense-model curl -fsS http://127.0.0.1:5000/api/health'
curl -fsS http://VPS_IP/api/health
```

Swagger (if the reverse proxy exposes the model): `http://VPS_IP/apidocs/`
