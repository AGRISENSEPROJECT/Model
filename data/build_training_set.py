"""Build an agronomically consistent Rwanda crop-suitability training table.

Climate: RAB/OFRA AEZ notes and national means (~20 C, ~1000 mm).
Crops: MINAGRI CIP / RwaSIS priority list plus tomato.
Seasons: A (Sep-Dec), B (Mar-May), C (Jun-Aug marshland).
"""

from __future__ import annotations

import csv
import os
from datetime import date, timedelta

import numpy as np

from data.crop_data import CROP_METADATA, CROP_SUITABILITY, canonical_crops, normalize_soil
from data.province_data import (
    DISTRICT_OFFSETS,
    LAND_USE_TYPES,
    PROVINCE_PROFILES,
    SEASON_RAIN_SHARE,
    SEASON_TEMP_OFFSET,
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'table', 'rwanda_crop_training.csv')
LEGACY_PATH = os.path.join(os.path.dirname(__file__), 'table', 'rwanda_provinces_dataset.csv')

SEASONS = ['A', 'B', 'C']
RNG = np.random.default_rng(42)


def _clip(value, low, high):
    return float(np.clip(value, low, high))


def _district_profile(district):
    offset = DISTRICT_OFFSETS[district]
    profile = dict(PROVINCE_PROFILES[offset['province']])
    profile['annual_rainfall_mm'] = profile['annual_rainfall_mm'] + offset['rain']
    low, high = profile['rainfall_range']
    profile['rainfall_range'] = (low + offset['rain'], high + offset['rain'])
    profile['mean_temp_c'] = profile['mean_temp_c'] + offset['temp']
    profile['province'] = offset['province']
    profile['district'] = district
    return profile


def _season_temperature(profile, season):
    base = profile['mean_temp_c'] + SEASON_TEMP_OFFSET[season]
    noise = float(RNG.normal(0, 1.1))
    low = profile['winter_min_c'] - 1
    high = profile['summer_max_c'] + 1
    return round(_clip(base + noise, low, high), 1)


def _annual_rainfall(profile, season):
    low, high = profile['rainfall_range']
    annual = float(RNG.uniform(low, high))
    if season == 'C':
        annual *= float(RNG.uniform(0.9, 1.0))
    elif season == 'B':
        annual *= float(RNG.uniform(1.0, 1.06))
    return round(annual, 0)


def _growing_season_rain(annual, season):
    share = SEASON_RAIN_SHARE[season]
    return round(annual * share * float(RNG.uniform(0.85, 1.15)), 0)


def _suitability_score(crop, soil, temp, humidity, rainfall, ph):
    reqs = CROP_SUITABILITY[crop]
    score = 0.0
    score += 30 if soil in reqs['soil_texture'] else 8
    tmin, tmax = reqs['temperature']['min'], reqs['temperature']['max']
    if tmin <= temp <= tmax:
        score += 20
    else:
        dist = min(abs(temp - tmin), abs(temp - tmax))
        score += max(0, 20 - dist * 2.5)
    hmin, hmax = reqs['humidity']['min'], reqs['humidity']['max']
    if hmin <= humidity <= hmax:
        score += 15
    else:
        dist = min(abs(humidity - hmin), abs(humidity - hmax))
        score += max(0, 15 - dist * 0.5)
    rmin, rmax = reqs['growing_season_rainfall']['min'], reqs['growing_season_rainfall']['max']
    if rmin <= rainfall <= rmax:
        score += 20
    else:
        dist = min(abs(rainfall - rmin), abs(rainfall - rmax)) / 80
        score += max(0, 20 - dist)
    pmin, pmax = reqs['ph']['min'], reqs['ph']['max']
    if pmin <= ph <= pmax:
        score += 15
    else:
        dist = min(abs(ph - pmin), abs(ph - pmax))
        score += max(0, 15 - dist * 8)
    return int(np.clip(round(score), 0, 100))


def _remark(score):
    if score >= 80:
        return 'High potential'
    if score >= 60:
        return 'Moderate potential'
    if score >= 40:
        return 'Requires attention'
    return 'Low potential'


def generate_rows(n_per_district=110):
    rows = []
    start = date(2023, 9, 1)
    for district in DISTRICT_OFFSETS:
        profile = _district_profile(district)
        for i in range(n_per_district):
            season = SEASONS[i % len(SEASONS)]
            crops = profile['crops_by_season'][season]
            crop = str(RNG.choice(crops))
            soil_raw = str(RNG.choice(profile['soils']))
            soil = normalize_soil(soil_raw)
            temp = _season_temperature(profile, season)
            humidity = _clip(profile['humidity'][season] + float(RNG.normal(0, 4)), 40, 95)
            rainfall = _annual_rainfall(profile, season)
            grow_rain = _growing_season_rain(rainfall, season)
            ph = round(float(RNG.uniform(*profile['ph_range'])), 2)
            fertility = int(RNG.integers(profile['fertility'][0], profile['fertility'][1] + 1))
            alt_low, alt_high = profile['altitude_m']
            altitude = int(RNG.integers(alt_low, alt_high + 1))
            land_use = str(RNG.choice(LAND_USE_TYPES, p=[0.5, 0.18, 0.1, 0.1, 0.07, 0.05]))
            score = _suitability_score(crop, soil, temp, humidity, grow_rain, ph)
            obs = start + timedelta(days=int(RNG.integers(0, 700)))
            meta = CROP_METADATA.get(crop, {})
            rows.append({
                'Country': 'Rwanda',
                'Province': profile['province'],
                'District': district,
                'Soil_Type': soil_raw.capitalize(),
                'Soil_Texture_Canonical': soil,
                'Fertility_Index': fertility,
                'Land_Use_Type': land_use,
                'Altitude_m': altitude,
                'Annual_Rainfall_mm': rainfall,
                'Growing_Season_Rainfall_mm': grow_rain,
                'Temperature_C': temp,
                'Humidity_pct': round(humidity, 1),
                'Soil_pH': ph,
                'Crop_Suitability': crop,
                'Season': season,
                'Suitability_Score': score,
                'Satellite_Observation_Date': obs.isoformat(),
                'Remarks': _remark(score),
                'Agroecology_Notes': profile['remarks'],
                'Data_Sources': '; '.join(meta.get('sources', ['RAB', 'OFRA'])),
            })
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = generate_rows()
    write_csv(OUTPUT_PATH, rows)
    crops = sorted({row['Crop_Suitability'] for row in rows})
    print(f'Wrote {len(rows)} rows to {OUTPUT_PATH}')
    print(f'Districts: {len(DISTRICT_OFFSETS)} | Crops: {", ".join(crops)}')
    print(f'Canonical crop envelopes: {len(canonical_crops())}')


if __name__ == '__main__':
    main()
