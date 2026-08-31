"""Verify soil class index map is alphabetical Keras order."""
import json
from pathlib import Path

from app.config import DEFAULT_SOIL_CLASS_INDICES, SOIL_CLASS_INDICES_PATH
from app.services.soil_texture import index_to_label_map, load_class_indices


def test_default_alphabetical_order():
    assert DEFAULT_SOIL_CLASS_INDICES == {
        "alluvial": 0,
        "clayey": 1,
        "loamy": 2,
        "sandy": 3,
    }


def test_artifact_class_indices_file():
    assert SOIL_CLASS_INDICES_PATH.exists()
    data = json.loads(Path(SOIL_CLASS_INDICES_PATH).read_text(encoding="utf-8"))
    assert data["alluvial"] == 0
    assert data["sandy"] == 3


def test_index_to_label_inverse():
    mapping = index_to_label_map()
    assert mapping[0] == "alluvial"
    assert mapping[3] == "sandy"
    assert load_class_indices()["loamy"] == 2
