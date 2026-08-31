from __future__ import annotations

import uuid
from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from app.config import ALLOWED_IMAGE_EXTENSIONS, UPLOAD_DIR


def ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def is_allowed_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def save_upload(file: FileStorage) -> Path:
    ensure_upload_dir()
    if not file or not file.filename:
        raise ValueError("No image file provided")
    if not is_allowed_image(file.filename):
        raise ValueError("Unsupported image type")
    safe = secure_filename(file.filename)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe}"
    file.save(path)
    return path
