"""Unit tests that do not require TensorFlow."""
from app.utils.crops import display_crop_name, normalize_crop_id


def test_normalize_crop_aliases():
    assert normalize_crop_id("Irish Potatoes") == "irish_potatoes"
    assert normalize_crop_id("tomatoes") == "tomatoes"
    assert normalize_crop_id("rice") == "rice"
    assert normalize_crop_id("potato") == "irish_potatoes"
    assert normalize_crop_id("maize") == "corn"
    assert normalize_crop_id("beans") == "beans"


def test_display_names():
    assert display_crop_name("irish_potatoes") == "Irish Potatoes"
