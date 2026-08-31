from __future__ import annotations

from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

from app.api import api_bp, register_web_routes
from app.config import Config, ROOT_DIR
from app.utils.uploads import ensure_upload_dir


def _load_dotenv() -> None:
    """Minimal .env loader (no extra dependency)."""
    import os

    path = ROOT_DIR / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def create_app(config_object: type[Config] = Config) -> Flask:
    _load_dotenv()
    app = Flask(
        __name__,
        static_folder=str(ROOT_DIR / "static"),
        static_url_path="/static",
        template_folder=str(ROOT_DIR / "templates"),
    )
    app.config.from_object(config_object)
    CORS(app)
    Swagger(app)
    ensure_upload_dir()

    app.register_blueprint(api_bp, url_prefix="/api")
    register_web_routes(app)
    return app


# Gunicorn / flask CLI entry: `gunicorn wsgi:app`
app = create_app()
