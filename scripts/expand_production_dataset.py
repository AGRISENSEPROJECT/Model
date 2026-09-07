"""Compile extra training data and report class balance for production."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps
import random

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
VAL = ROOT / "data" / "validation"
SEED = ROOT / "data" / "images" / "soil_seeds"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASSES = ("alluvial", "clayey", "loamy", "sandy")
IMG_SIZE = (224, 224)


def _count(split: Path) -> dict[str, int]:
    out = {}
    for klass in CLASSES:
        folder = split / klass
        out[klass] = len([p for p in folder.glob("*") if p.suffix.lower() in EXTS]) if folder.exists() else 0
    return out


def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in EXTS]


def _augment(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    out = img.convert("RGB").resize(IMG_SIZE)
    if rng.random() < 0.5:
        out = ImageOps.mirror(out)
    if rng.random() < 0.3:
        out = ImageOps.flip(out)
    out = out.rotate(rng.uniform(-28, 28), resample=Image.BILINEAR, fillcolor=out.getpixel((4, 4)))
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.72, 1.28))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.8, 1.3))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.85, 1.2))
    left = rng.randint(0, 18)
    top = rng.randint(0, 18)
    out = out.crop((left, top, 224 - (18 - left), 224 - (18 - top))).resize(IMG_SIZE)
    return out


def expand_sandy(train_target: int = 220, val_target: int = 36) -> dict:
    """Oversample REAL sandy photos; keep original files untouched."""
    train_dir = TRAIN / "sandy"
    val_dir = VAL / "sandy"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    originals = [p for p in _list_images(train_dir) if "aug_" not in p.name and "gen_" not in p.name]
    if not originals:
        originals = _list_images(train_dir)
    if not originals:
        raise FileNotFoundError("No sandy training images to expand")

    extra_seeds = _list_images(SEED / "sandy")
    sources = originals + extra_seeds
    val_sources = extra_seeds or originals

    before = _count(TRAIN)
    need = max(0, train_target - before["sandy"])
    written = 0
    idx = 0
    while written < need:
        src = sources[idx % len(sources)]
        dest = train_dir / f"aug_sandy_{idx:04d}.jpg"
        if not dest.exists():
            _augment(Image.open(src), seed=7000 + idx).save(dest, quality=90)
            written += 1
        idx += 1
        if idx > need + 400:
            break

    val_before = _count(VAL)["sandy"]
    val_need = max(0, val_target - val_before)
    val_written = 0
    for i in range(val_need):
        src = val_sources[i % len(val_sources)]
        dest = val_dir / f"aug_sandy_val_{i:03d}.jpg"
        if dest.exists():
            continue
        _augment(Image.open(src), seed=91000 + i).save(dest, quality=90)
        val_written += 1

    report = {
        "train_before": before,
        "train_after": _count(TRAIN),
        "val_after": _count(VAL),
        "train_sandy_added": written,
        "val_sandy_added": val_written,
        "note": "Sandy oversampled from existing real photos + seed close-ups. Retrain CNN with class weights.",
    }
    out = ROOT / "artifacts" / "dataset_balance.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# NISR Seasonal Agricultural Survey 2023 (published national totals).
NISR_2023 = [
    {
        "crop": "maize",
        "canonical_id": "corn",
        "production_tonnes": 508000,
        "season_a_area_ha": 226982,
        "season_b_area_ha": 93927,
        "yield_t_ha_season_a": None,
        "source": "NISR SAS 2023 (allAfrica / Milling MEA summaries)",
        "notes": "Annual maize >508,000 t; up from ~458,500 t in 2022",
    },
    {
        "crop": "beans",
        "canonical_id": None,
        "production_tonnes": 441000,
        "season_a_area_ha": 312279,
        "season_b_area_ha": 309489,
        "yield_t_ha_season_a": 0.632,
        "source": "NISR SAS 2023",
        "notes": "Season C beans 3,476 ha; yield 632/789/1003 kg/ha A/B/C",
    },
    {
        "crop": "irish_potato",
        "canonical_id": "irish_potatoes",
        "production_tonnes": 865000,
        "season_a_area_ha": 55613,
        "season_b_area_ha": 48210,
        "yield_t_ha_season_a": 8.2,
        "source": "NISR SAS 2023 / FreshPlaza",
        "notes": "Yield 8.2 / 6.7 / 7.9 t/ha seasons A/B/C",
    },
    {
        "crop": "cassava",
        "canonical_id": None,
        "production_tonnes": 1340000,
        "season_a_area_ha": 239221,
        "season_b_area_ha": 159089,
        "yield_t_ha_season_a": 13.5,
        "source": "NISR SAS 2023",
        "notes": "Season B yield 14.8 t/ha; harvested area ~45k / 50k ha A/B",
    },
    {
        "crop": "banana",
        "canonical_id": None,
        "production_tonnes": 1142552,
        "season_a_area_ha": None,
        "season_b_area_ha": 258564,
        "yield_t_ha_season_a": None,
        "source": "NISR SAS 2024B",
        "notes": "Season B production; banana is perennial",
    },
]


def write_rwanda_official_table() -> Path:
    dest = ROOT / "data" / "rwanda" / "nisr_sas_major_crops.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "crop",
        "canonical_id",
        "production_tonnes",
        "season_a_area_ha",
        "season_b_area_ha",
        "yield_t_ha_season_a",
        "source",
        "notes",
    ]
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in NISR_2023:
            writer.writerow({k: row.get(k, "") for k in fields})
    return dest


def main():
    table = write_rwanda_official_table()
    report = expand_sandy()
    report["rwanda_official_table"] = str(table)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
