"""
Multi-factor crop ranker.

Combines, in one score:
  1. Soil texture, pH, NPK vs crop demand
  2. Weather (temperature, rainfall/moisture, humidity)
  3. Rwanda season A/B/C
  4. Crop rotation (previous crop / family)
  5. ML predicted yield (canonical crops) or NISR typical yield
  6. Farmgate price -> expected income RWF/ha

ML yield models stay on the original 10 canonical crops.
Rwanda staples (beans, cassava, banana, ...) are scored from the
crop-factor dataset so they can still win on income and season fit.
"""
from __future__ import annotations

from typing import Any

from app.knowledge.catalog import load_catalog
from app.knowledge.rotation import rotation_fit
from app.knowledge.seasons import (
    SEASON_LABELS,
    current_season,
    infer_province,
    next_seasons,
)
from app.utils.crops import normalize_crop_id

WEIGHTS_INCOME = {
    "agronomic": 0.28,
    "yield": 0.18,
    "income": 0.32,
    "season": 0.12,
    "rotation": 0.10,
}
WEIGHTS_YIELD = {
    "agronomic": 0.36,
    "yield": 0.28,
    "income": 0.16,
    "season": 0.12,
    "rotation": 0.08,
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _range_score(value: float | None, lo: float, hi: float) -> float:
    if value is None:
        return 0.72
    if lo <= value <= hi:
        return 1.0
    span = max(hi - lo, 1.0)
    if value < lo:
        return _clamp(1.0 - (lo - value) / span)
    return _clamp(1.0 - (value - hi) / span)


def _npk_score(measured: float | None, optimum: float) -> float:
    if measured is None or optimum <= 0:
        return 0.72
    ratio = float(measured) / optimum
    if 0.7 <= ratio <= 1.4:
        return 1.0
    if ratio < 0.7:
        return _clamp(ratio / 0.7)
    return _clamp(1.4 / ratio)


def agronomic_fit(record: dict[str, Any], conditions: dict[str, Any]) -> tuple[float, dict[str, float]]:
    texture = (conditions.get("soil_texture") or "").lower() or None
    soil_score = 1.0 if texture and texture in record["soils"] else (0.72 if not texture else 0.38)
    parts = {
        "soil_texture": soil_score,
        "temperature": _range_score(conditions.get("temperature_c"), record["temp_min"], record["temp_max"]),
        "rainfall": _range_score(conditions.get("precip_accum_mm"), record["rain_min"], record["rain_max"]),
        "humidity": _range_score(conditions.get("relative_humidity"), record["humidity_min"], record["humidity_max"]),
        "ph": _range_score(conditions.get("soil_ph"), record["ph_min"], record["ph_max"]),
        "nitrogen": _npk_score(conditions.get("nitrogen"), record["n_opt"]),
        "phosphorus": _npk_score(conditions.get("phosphorus"), record["p_opt"]),
        "potassium": _npk_score(conditions.get("potassium"), record["k_opt"]),
    }
    if conditions.get("province") and record["provinces"]:
        parts["province"] = 1.0 if conditions["province"] in record["provinces"] else 0.55
    weights = {
        "soil_texture": 1.4,
        "temperature": 1.2,
        "rainfall": 1.1,
        "humidity": 0.7,
        "ph": 0.8,
        "nitrogen": 0.9,
        "phosphorus": 0.8,
        "potassium": 0.8,
        "province": 0.6,
    }
    num = sum(parts[k] * weights.get(k, 1.0) for k in parts)
    den = sum(weights.get(k, 1.0) for k in parts)
    fit = _clamp((num / den) * float(record["rwanda_priority"]))
    return round(fit, 4), {k: round(v, 3) for k, v in parts.items()}


def _season_fit(record: dict[str, Any], season: str) -> tuple[float, str]:
    if record.get("perennial"):
        return 1.0, "Perennial — grows across seasons"
    if season in record["seasons"]:
        return 1.0, f"In season {season}"
    return 0.28, f"Off-season for {season} (better in {', '.join(record['seasons']) or 'other seasons'})"


def _ml_yield_map(ml_rows: list[dict[str, Any]] | None) -> dict[str, float]:
    out = {}
    for row in ml_rows or []:
        cid = row.get("crop_id")
        if cid:
            out[str(cid)] = float(row.get("predicted_yield") or 0)
    return out


def _expected_yield_t_ha(record: dict[str, Any], agro: float, ml_y: float | None, ml_max: float) -> float:
    typical = float(record["typical_yield_t_ha"])
    relative = 1.0
    if ml_y is not None and ml_max > 0:
        relative = _clamp(0.45 + 0.7 * (ml_y / ml_max), 0.4, 1.25)
    return round(max(0.05, typical * agro * relative), 3)


def rank_crops(
    *,
    conditions: dict[str, Any],
    ml_rows: list[dict[str, Any]] | None = None,
    history: dict[str, Any] | None = None,
    economic: dict[str, Any] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    catalog = load_catalog()
    history = dict(history or {})
    economic = dict(economic or {})
    season = str(history.get("season") or conditions.get("season") or current_season()).upper()
    if season not in ("A", "B", "C"):
        season = current_season()
    previous = normalize_crop_id(history.get("previous_crop") or history.get("last_crop"))
    province = history.get("province") or conditions.get("province")
    maximize_income = economic.get("maximize_income", True)
    if isinstance(maximize_income, str):
        maximize_income = maximize_income.strip().lower() not in {"0", "false", "no"}
    weights = WEIGHTS_INCOME if maximize_income else WEIGHTS_YIELD
    price_overrides = economic.get("market_prices") or economic.get("prices") or {}

    ml_map = _ml_yield_map(ml_rows)
    ml_max = max(ml_map.values()) if ml_map else 0.0

    scored: list[dict[str, Any]] = []
    for crop_id, record in catalog.items():
        agro, agro_parts = agronomic_fit(record, conditions)
        season_score, season_reason = _season_fit(record, season)
        rot_score, rot_reason = rotation_fit(previous, crop_id, catalog)
        ml_y = ml_map.get(crop_id)
        yield_t = _expected_yield_t_ha(record, agro, ml_y, ml_max)
        price = float(price_overrides.get(crop_id) or price_overrides.get(record["display_name"]) or record["farmgate_rwf_kg"])
        income = round(yield_t * 1000.0 * price, 0)
        scored.append(
            {
                "crop_id": crop_id,
                "crop": record["display_name"],
                "family": record["family"],
                "predicted_yield": round(ml_y, 2) if ml_y is not None else yield_t,
                "expected_yield_t_ha": yield_t,
                "farmgate_price_rwf_kg": price,
                "expected_income_rwf_ha": income,
                "agronomic_fit": round(agro * 100, 1),
                "season_fit": round(season_score * 100, 1),
                "rotation_fit": round(min(rot_score, 1.15) * 100 / 1.15, 1),
                "ml_backed": bool(record["ml_backed"]),
                "source": "multi_factor_ranker",
                "method": "soil_weather_season_rotation_income",
                "factor_breakdown": {
                    "agronomic": agro_parts,
                    "season": season_reason,
                    "rotation": rot_reason,
                    "yield_basis": "ml+nisr" if ml_y is not None else "nisr_typical",
                    "price_source": "request" if crop_id in price_overrides else record["price_source"],
                },
                "_agro": agro,
                "_season": season_score,
                "_rotation": _clamp(rot_score / 1.15),
                "_yield_t": yield_t,
                "_income": income,
            }
        )

    max_yield = max((r["_yield_t"] for r in scored), default=1.0) or 1.0
    max_income = max((r["_income"] for r in scored), default=1.0) or 1.0
    for row in scored:
        y_n = row["_yield_t"] / max_yield
        i_n = row["_income"] / max_income
        composite = (
            weights["agronomic"] * row["_agro"]
            + weights["yield"] * y_n
            + weights["income"] * i_n
            + weights["season"] * row["_season"]
            + weights["rotation"] * row["_rotation"]
        )
        row["suitability_score"] = round(composite * 100.0, 2)
        row["probability"] = round(composite, 4)
        row["explanation"] = (
            f"{row['crop']}: agronomy {row['agronomic_fit']:.0f}/100, "
            f"season {season} ({row['factor_breakdown']['season']}), "
            f"~{row['expected_yield_t_ha']} t/ha, "
            f"{int(row['expected_income_rwf_ha']):,} RWF/ha at {int(row['farmgate_price_rwf_kg'])} RWF/kg. "
            f"{row['factor_breakdown']['rotation']}."
        )
        for key in ("_agro", "_season", "_rotation", "_yield_t", "_income"):
            del row[key]

    scored.sort(key=lambda x: x["suitability_score"], reverse=True)
    ranked = scored[:top_k]
    best = ranked[0] if ranked else None
    income_best = max(scored, key=lambda x: x["expected_income_rwf_ha"]) if scored else None

    plan = []
    follow_from = best["crop_id"] if best else None
    for nxt_season in next_seasons(season, 3):
        pick = None
        pick_reason = ""
        for cand in scored:
            rec = catalog[cand["crop_id"]]
            if rec.get("perennial") and cand["crop_id"] == follow_from:
                continue
            s_fit, _ = _season_fit(rec, nxt_season)
            r_fit, r_reason = rotation_fit(follow_from, cand["crop_id"], catalog)
            if s_fit < 0.9:
                continue
            if r_fit < 0.7:
                continue
            pick = cand
            pick_reason = r_reason
            break
        if pick:
            plan.append(
                {
                    "season": nxt_season,
                    "season_label": SEASON_LABELS[nxt_season],
                    "crop_id": pick["crop_id"],
                    "crop": pick["crop"],
                    "reason": pick_reason,
                }
            )
            follow_from = pick["crop_id"]

    return {
        "crop_recommendations": ranked,
        "season": season,
        "season_label": SEASON_LABELS.get(season, season),
        "province": province,
        "previous_crop": previous,
        "maximize_income": bool(maximize_income),
        "weights": weights,
        "income_maximizing_crop": (
            {
                "crop_id": income_best["crop_id"],
                "crop": income_best["crop"],
                "expected_income_rwf_ha": income_best["expected_income_rwf_ha"],
                "farmgate_price_rwf_kg": income_best["farmgate_price_rwf_kg"],
            }
            if income_best
            else None
        ),
        "crop_rotation_plan": plan,
        "factors_used": [
            "soil_texture",
            "temperature",
            "rainfall",
            "humidity",
            "soil_ph",
            "npk",
            "season",
            "crop_rotation",
            "typical_or_ml_yield",
            "farmgate_price",
            "expected_income",
            "province",
        ],
        "source": "multi_factor_ranker",
    }


def conditions_from_blocks(
    *,
    soil: dict[str, Any] | None = None,
    weather: dict[str, Any] | None = None,
    merged: dict[str, Any] | None = None,
    history: dict[str, Any] | None = None,
    coordinates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    soil = soil or {}
    weather = weather or {}
    merged = merged or {}
    history = history or {}
    coords = coordinates or {}
    lat = coords.get("lat")
    lon = coords.get("lon")
    return {
        "soil_texture": soil.get("soil_texture") or merged.get("soil_texture"),
        "soil_ph": _to_float(soil.get("soil_ph") or merged.get("soil_ph")),
        "nitrogen": _to_float(soil.get("nitrogen") or merged.get("nitrogen")),
        "phosphorus": _to_float(soil.get("phosphorus") or merged.get("phosphorus")),
        "potassium": _to_float(soil.get("potassium") or merged.get("potassium")),
        "temperature_c": _to_float(weather.get("temperature_c") or merged.get("temperature_c")),
        "precip_accum_mm": _to_float(weather.get("precip_accum_mm") or merged.get("precip_accum_mm")),
        "relative_humidity": _to_float(weather.get("relative_humidity") or merged.get("relative_humidity")),
        "province": history.get("province") or infer_province(_to_float(lat), _to_float(lon)),
        "season": history.get("season"),
    }


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
