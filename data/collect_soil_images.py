"""Download and synthesize soil-texture photos for the MobileNet classifier.

Creates:
  data/train/{sandy,loamy,clayey,alluvial}
  data/validation/{sandy,loamy,clayey,alluvial}
"""

from __future__ import annotations

import io
import os
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(ROOT, 'train')
VAL_DIR = os.path.join(ROOT, 'validation')
SEED_DIR = os.path.join(ROOT, 'images', 'soil_seeds')

CLASSES = ['sandy', 'loamy', 'clayey', 'alluvial']
IMG_SIZE = (224, 224)
TRAIN_PER_CLASS = 60
VAL_PER_CLASS = 15

# Wikimedia Commons (CC) close-up soil / ground photos used as seeds.
SEED_URLS = {
    'sandy': [
        'https://upload.wikimedia.org/wikipedia/commons/5/5a/Sand_from_Gobi_Desert.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/a/a4/Sand_texture.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/3/3b/Sand_closeup.jpg',
    ],
    'loamy': [
        'https://upload.wikimedia.org/wikipedia/commons/4/4c/Soil.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/1/1c/SoilTexture.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/6/6d/Dirt.JPG',
    ],
    'clayey': [
        'https://upload.wikimedia.org/wikipedia/commons/6/61/Red_clay_soil.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/8/80/Clay_soil.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/d/d4/Laterite_soil.jpg',
    ],
    'alluvial': [
        'https://upload.wikimedia.org/wikipedia/commons/2/2e/Alluvial_soil.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/9/9a/Silt_soil.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/7/7e/Wet_soil.jpg',
    ],
}

PALETTES = {
    'sandy': [(194, 166, 122), (210, 180, 130), (176, 148, 104), (222, 198, 154)],
    'loamy': [(92, 64, 40), (120, 82, 50), (74, 52, 32), (138, 96, 58)],
    'clayey': [(148, 72, 42), (168, 82, 48), (128, 58, 36), (186, 96, 58)],
    'alluvial': [(72, 68, 52), (88, 84, 64), (60, 70, 48), (96, 92, 70)],
}


def _ensure_dirs():
    for split in (TRAIN_DIR, VAL_DIR, SEED_DIR):
        for klass in CLASSES:
            os.makedirs(os.path.join(split, klass) if split != SEED_DIR else os.path.join(SEED_DIR, klass), exist_ok=True)


def _download(url, dest):
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'AgrisenseSoilDataset/1.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        Image.open(io.BytesIO(data)).verify()
        with open(dest, 'wb') as handle:
            handle.write(data)
        return True
    except Exception as exc:
        print(f'  skip {url}: {exc}')
        return False


def _make_procedural(klass, index, size=IMG_SIZE):
    rng = np.random.default_rng(index + (sum(ord(c) for c in klass) % 10_000))
    h, w = size[1], size[0]
    base = rng.choice(PALETTES[klass])
    img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        noise = rng.normal(0, 18, (h, w))
        # Coarser grain for sand, finer for clay.
        scale = {'sandy': 7, 'loamy': 4, 'clayey': 2, 'alluvial': 5}[klass]
        yy, xx = np.mgrid[0:h, 0:w]
        wave = 8 * np.sin(xx / (12 + scale)) + 6 * np.cos(yy / (10 + scale))
        img[:, :, c] = np.clip(base[c] + noise + wave, 0, 255)
    if klass == 'sandy':
        speck = rng.integers(0, 255, (h, w, 3))
        mask = rng.random((h, w, 1)) > 0.92
        img = np.where(mask, speck, img)
    if klass == 'alluvial':
        img = img * 0.9
        img[:, :, 1] += 6
    pil = Image.fromarray(img.astype(np.uint8), 'RGB').filter(ImageFilter.GaussianBlur(radius=0.6 if klass != 'sandy' else 0.2))
    return pil


def _augment(img, seed):
    rng = random.Random(seed)
    out = img.convert('RGB').resize(IMG_SIZE)
    if rng.random() < 0.5:
        out = ImageOps.mirror(out)
    if rng.random() < 0.5:
        out = ImageOps.flip(out)
    out = out.rotate(rng.uniform(-25, 25), resample=Image.BILINEAR, fillcolor=tuple(out.getpixel((2, 2))))
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.75, 1.25))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.8, 1.25))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.85, 1.15))
    left = rng.randint(0, 16)
    top = rng.randint(0, 16)
    out = out.crop((left, top, 224 - (16 - left), 224 - (16 - top))).resize(IMG_SIZE)
    return out


def collect_seeds():
    seeds = {klass: [] for klass in CLASSES}
    for klass, urls in SEED_URLS.items():
        folder = os.path.join(SEED_DIR, klass)
        os.makedirs(folder, exist_ok=True)
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                try:
                    seeds[klass].append(Image.open(os.path.join(folder, fname)).convert('RGB'))
                except Exception:
                    pass
        for i, url in enumerate(urls):
            dest = os.path.join(folder, f'seed_{i}.jpg')
            if not os.path.exists(dest) and _download(url, dest):
                try:
                    seeds[klass].append(Image.open(dest).convert('RGB'))
                except Exception:
                    pass
        for extra in range(4):
            seeds[klass].append(_make_procedural(klass, extra))
        print(f'{klass}: {len(seeds[klass])} seed images')
    return seeds


def write_split(seeds):
    for klass in CLASSES:
        bank = seeds[klass]
        for i in range(TRAIN_PER_CLASS):
            src = bank[i % len(bank)]
            _augment(src, seed=1000 + i).save(os.path.join(TRAIN_DIR, klass, f'{klass}_train_{i:03d}.jpg'), quality=90)
        for i in range(VAL_PER_CLASS):
            src = bank[i % len(bank)]
            _augment(src, seed=9000 + i).save(os.path.join(VAL_DIR, klass, f'{klass}_val_{i:03d}.jpg'), quality=90)
        print(f'wrote {TRAIN_PER_CLASS} train + {VAL_PER_CLASS} val for {klass}')
        for j, src in enumerate(bank[:8]):
            src.convert('RGB').resize(IMG_SIZE).save(
                os.path.join(TRAIN_DIR, klass, f'{klass}_seed_{j:02d}.jpg'), quality=92
            )


def main():
    _ensure_dirs()
    seeds = collect_seeds()
    write_split(seeds)
    total = 0
    for split, folder in (('train', TRAIN_DIR), ('val', VAL_DIR)):
        for klass in CLASSES:
            n = len([f for f in os.listdir(os.path.join(folder, klass)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total += n
            print(f'{split}/{klass}: {n}')
    print(f'Total soil images: {total}')


if __name__ == '__main__':
    main()
