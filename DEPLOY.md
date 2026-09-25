# Model deploy (VPS)

A push to `main` generates environmental artifacts for CI, runs pytest,
syncs this repo to `/opt/agrisense/Model/`, rebuilds only the Compose
`model` service, and checks model health, API-to-model
reachability, and the public API over HTTPS. The workflow also supports manual
runs. The model stays internal to Docker and is not publicly exposed.

The workflow runs on pushes to `main` and can also be started manually.

## One-time GitHub Actions setup

Set these secrets in the **Model** repository:

| Secret | Value |
| --- | --- |
| `VPS_HOST` | `92.4.152.193` |
| `VPS_PORT` | `22` (optional; the workflow defaults to 22) |
| `VPS_USER` | `ubuntu` |
| `VPS_SSH_KEY` | Private key authorized for `ubuntu`; preferably a dedicated deploy key |
| `VPS_KNOWN_HOSTS` | Verified SSH host-key line for `92.4.152.193` |

Obtain the server's public host key through the existing trusted
`ssh agrisense-oracle` connection, verify its fingerprint, and store a
`92.4.152.193 ssh-ed25519 <public-key>` line as `VPS_KNOWN_HOSTS`. Do not
generate this value with an unauthenticated `ssh-keyscan` in the workflow.
The ED25519 host-key fingerprint observed through the existing SSH alias is
`SHA256:Hmha/BVjMlbtOFQyjWnIe5lbiRoM4ZlWIbcd4AGwPGE`.
Never commit the private key or production configuration.

## Artifact ownership

GitHub is the source for committed crop, precision, and soil CNN artifacts.
Normal pushes refuse to replace a differing live artifact. For an intentional
update, run the workflow manually with `approve_artifact_update=true`.

`yield_predictor.pkl`, `soil_quality_predictor.pkl`, and
`environmental_model_meta.json` are derived from
`data/environmental/crop_yield_dataset.csv`. CI regenerates them for tests,
and Git ignores them. The deployment excludes them from artifact sync. The VPS keeps its own
copies in the persistent artifact mount. The runtime uses
`scikit-learn==1.9.0`, matching the committed crop and precision pickles.
When changing this version or the environmental training code, back up the
current image and artifacts, rebuild, and regenerate these three derived files
with the new image before restarting the model.

## Repository and VPS setup

- Keep generated environmental artifacts out of Git when changing model code
  or training data.
- The VPS now bind-mounts `/app/artifacts`, `/app/data/retrain`,
  `/app/data/field_visits`, and `/app/data/device_readings` from
  `/opt/agrisense/Model/`. The live files were preserved before the first
  model-only restart. The backup is at
  `/opt/agrisense/deploy/backups/model-persistence-20260925T005432Z/`.
- The workflow checks those mounts before syncing. It protects runtime data,
  uploads, and all three generated artifacts. Other tracked artifacts are synced
  separately with shared write permissions.
- Keep the working Compose file at
  `/opt/agrisense/deploy/docker-compose.yml`; this workflow does not sync it.

The Compose healthcheck uses `/api/health`. Serving readiness is independent
of training-image availability: `production_ready: true` means the artifacts
and validation thresholds pass, while `training.ready` remains false when
training images are omitted from the inference image. The workflow checks
serving readiness, API-to-model reachability, and public API health. It does not
perform an authenticated prediction or automatically roll back on failure.

The model-runtime backup is at
`/opt/agrisense/deploy/backups/model-runtime-upgrade-20260925T014448Z/`,
with rollback image `agrisense-model:rollback-20260925T014448Z`.
