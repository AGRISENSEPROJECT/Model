"""
Generate realistic, large-scale synthetic datasets for Rwanda.
This expands the seed files into full datasets suitable for ML training and testing.
"""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "rwanda"

DISTRICTS = {
    "Northern": ["Musanze", "Burera", "Gicumbi", "Rulindo", "Gakenke"],
    "Western": ["Nyabihu", "Rubavu", "Rutsiro", "Ngororero", "Karongi", "Nyamasheke", "Rusizi"],
    "Southern": ["Huye", "Nyanza", "Gisagara", "Nyaruguru", "Nyamagabe", "Ruhango", "Muhanga", "Kamonyi"],
    "Eastern": ["Nyagatare", "Gatsibo", "Kayonza", "Rwamagana", "Ngoma", "Kirehe", "Bugesera"],
    "Kigali": ["Gasabo", "Kicukiro", "Nyarugenge"]
}

BASE_PRICES = {
    "corn": 350, "beans": 1123, "irish_potatoes": 512, "cassava": 180, 
    "banana": 250, "rice": 954, "tomatoes": 450, "wheat": 600, 
    "soybean": 700, "sorghum": 400, "sweet_potato": 220, "coffee": 2800, 
    "tea": 400, "groundnut": 900, "pea": 800, "barley": 500, 
    "sunflower": 650, "cotton": 700, "sugarcane": 80
}

def load_crop_factors():
    factors = {}
    with open(DATA_DIR / "crop_recommendation_factors.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            factors[row["crop_id"]] = row
    return factors

def generate_market_prices():
    out_path = DATA_DIR / "district_market_prices.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["district", "crop_id", "price_rwf_kg", "date_updated"])
        
        date_str = "2026-09-15"
        for prov, dists in DISTRICTS.items():
            for dist in dists:
                for crop, base_price in BASE_PRICES.items():
                    # Add regional variation (-15% to +15%)
                    noise = random.uniform(-0.15, 0.15)
                    
                    # Specific regional discounts (e.g., potatoes cheaper in North)
                    if crop == "irish_potatoes" and prov == "Northern":
                        noise -= 0.10
                    if crop == "banana" and prov == "Eastern":
                        noise -= 0.10
                    if crop == "rice" and prov == "Southern":
                        noise -= 0.10
                        
                    final_price = int(base_price * (1 + noise))
                    # Round to nearest 5
                    final_price = 5 * round(final_price / 5)
                    writer.writerow([dist, crop, final_price, date_str])
    print(f"Generated {out_path.name} with {len(DISTRICTS.keys()) * 6 * len(BASE_PRICES)} rows")

def generate_nisr_data():
    out_path = DATA_DIR / "nisr_sas_major_crops.csv"
    # Expanded list of national production based on FAO/NISR estimates
    data = [
        ["maize", "corn", 508000, 226982, 93927, 2.2, "NISR SAS 2023", "Annual maize >508,000 t"],
        ["beans", "", 441000, 312279, 309489, 0.632, "NISR SAS 2023", "Season C beans 3,476 ha"],
        ["irish_potato", "irish_potatoes", 865000, 55613, 48210, 8.2, "NISR SAS 2023", "Yield 8.2 / 6.7 / 7.9 t/ha seasons A/B/C"],
        ["cassava", "", 1340000, 239221, 159089, 13.5, "NISR SAS 2023", "Season B yield 14.8 t/ha"],
        ["banana", "", 1142552, "", 258564, 8.5, "NISR SAS 2024B", "Season B production; banana is perennial"],
        ["rice", "rice", 141932, 15000, 18000, 3.5, "FAOSTAT 2024", "Marshland production"],
        ["sweet_potato", "sweet_potato", 850000, 120000, 90000, 6.0, "FAOSTAT 2023", "Major food security crop"],
        ["sorghum", "sorghum", 140000, 80000, 40000, 1.2, "NISR SAS 2023", "Drought tolerant"],
        ["soybean", "soybean", 25000, 15000, 12000, 1.0, "NISR SAS 2023", "Growing legume sector"],
        ["wheat", "wheat", 12000, 5000, 3000, 1.5, "FAOSTAT 2023", "Highland crop"],
        ["tomatoes", "tomatoes", 110000, 8000, 9000, 12.0, "FAOSTAT 2023", "High value horticultural"],
        ["coffee", "coffee", 22000, "", 35000, 0.8, "NAEB 2023", "Export cash crop"],
        ["tea", "tea", 35000, "", 28000, 1.5, "NAEB 2023", "Export cash crop"],
        ["groundnut", "groundnut", 15000, 10000, 8000, 0.8, "FAOSTAT 2023", "Eastern province staple"],
        ["pea", "pea", 18000, 12000, 8000, 1.1, "FAOSTAT 2023", "Highland legume"]
    ]
    
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop", "canonical_id", "production_tonnes", "season_a_area_ha", "season_b_area_ha", "yield_t_ha_season_a", "source", "notes"])
        writer.writerows(data)
    print(f"Generated {out_path.name} with {len(data)} rows")

def generate_calibration_data(num_rows=2500):
    out_path = DATA_DIR / "plot_harvest_calibration.csv"
    factors = load_crop_factors()
    
    varieties = {
        "irish_potatoes": ["Kinigi", "Kirundo", "Cruza"],
        "corn": ["Hybrid WH", "Local", "Kigega"],
        "beans": ["Climbing", "Bush", "Mutiki"],
        "rice": ["Kigori", "Zhongeng", "Intsindagirabigega"],
        "cassava": ["Gisari", "Ndamirabana", "Gahene"]
    }
    
    fertilizers = ["NPK 17-17-17", "DAP+Urea", "Manure", "NPK+Manure", "None", "Urea"]
    
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "plot_id", "date", "season", "district", "crop_id", "variety", 
            "soil_texture", "lab_ph", "nitrogen", "phosphorus", "potassium", 
            "ec_us_cm", "moisture_vwc", "yield_t_ha", "fertilizer_applied"
        ])
        
        for i in range(num_rows):
            crop_id = random.choice(list(factors.keys()))
            crop = factors[crop_id]
            
            # Select valid season
            seasons = [s for s in crop["seasons"].split("|") if s]
            season = random.choice(seasons) if seasons else "A"
            
            # Select valid province/district
            provs = [p for p in crop["provinces"].split("|") if p]
            prov = random.choice(provs) if provs else "Southern"
            district = random.choice(DISTRICTS[prov])
            
            # Select valid soil
            soils = [s for s in crop["soils"].split("|") if s]
            soil = random.choice(soils) if soils else "loamy"
            
            # Variety
            variety = random.choice(varieties.get(crop_id, ["Local", "Improved"]))
            
            # Generate agronomic values around optimums with noise
            n_opt = float(crop["n_opt"])
            p_opt = float(crop["p_opt"])
            k_opt = float(crop["k_opt"])
            
            # Simulate different farm conditions (Good, Average, Poor)
            farm_quality = random.choices(["Good", "Average", "Poor"], weights=[0.3, 0.5, 0.2])[0]
            
            if farm_quality == "Good":
                n = n_opt * random.uniform(0.9, 1.2)
                p = p_opt * random.uniform(0.9, 1.2)
                k = k_opt * random.uniform(0.9, 1.2)
                fert = random.choice(["NPK 17-17-17", "DAP+Urea", "NPK+Manure"])
                yield_mult = random.uniform(0.9, 1.3)
            elif farm_quality == "Average":
                n = n_opt * random.uniform(0.6, 0.95)
                p = p_opt * random.uniform(0.6, 0.95)
                k = k_opt * random.uniform(0.6, 0.95)
                fert = random.choice(["Manure", "Urea", "DAP+Urea"])
                yield_mult = random.uniform(0.6, 0.95)
            else:
                n = n_opt * random.uniform(0.3, 0.6)
                p = p_opt * random.uniform(0.3, 0.6)
                k = k_opt * random.uniform(0.3, 0.6)
                fert = "None"
                yield_mult = random.uniform(0.3, 0.6)
                
            # pH
            ph_min = float(crop["ph_min"])
            ph_max = float(crop["ph_max"])
            ph = random.uniform(ph_min - 0.5, ph_max + 0.5)
            if ph < ph_min or ph > ph_max:
                yield_mult *= 0.8 # Penalty for bad pH
                
            # Moisture and EC
            moisture = random.uniform(30.0, 70.0)
            ec = random.uniform(100, 600)
            
            # Calculate final yield
            typical_yield = float(crop["typical_yield_t_ha"])
            final_yield = typical_yield * yield_mult
            
            # Date
            year = random.choice([2024, 2025, 2026])
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date_str = f"{year}-{month:02d}-{day:02d}"
            
            writer.writerow([
                f"PLT-{i+1:04d}", date_str, season, district, crop_id, variety,
                soil, round(ph, 2), round(n, 1), round(p, 1), round(k, 1),
                round(ec, 1), round(moisture, 1), round(final_yield, 2), fert
            ])
            
    print(f"Generated {out_path.name} with {num_rows} rows")

if __name__ == "__main__":
    generate_market_prices()
    generate_nisr_data()
    generate_calibration_data()
