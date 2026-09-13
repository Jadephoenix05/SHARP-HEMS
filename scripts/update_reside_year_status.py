"""Record the RESIDE-AC clock-year diagnosis in the master registry.

The CSV year (2019) is supported over the description year (2021) by an
independent outdoor-weather correlation test with a pre-registered decision
rule. This is statistical evidence about the clock, not a documentary
correction from the dataset authors, and it is labelled as such.

Nothing about units, sensor mapping, AC label construction or causal cooling
is settled by this, so those checks remain open.
"""
from pathlib import Path
from datetime import datetime
import json
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data_registry/sources.yaml"
DIAGNOSIS = ROOT / "reports/reside_year_hypothesis_v1.json"

report = json.loads(DIAGNOSIS.read_text(encoding="utf-8"))
if not report["tests_agree"] or report["primary_verdict"] != "YEAR_2019_SUPPORTED":
    raise ValueError(
        "The diagnosis does not support a single year; registry unchanged."
    )

update = {
    "status": "CLOCK_YEAR_DIAGNOSED_UNITS_AND_LABELS_PENDING_REVIEW",
    "observation_year_supported": 2019,
    "observation_year_basis": (
        "Daily indoor temperature correlates with Hyderabad 2019 outdoor "
        "temperature far better than with 2021 (AC-off mean r 0.524 versus "
        "0.126; 10 of 11 houses favour 2019). Statistical evidence about the "
        "clock, not an author correction."
    ),
    "observation_year_report": "reports/reside_year_hypothesis_v1.json",
    "site_weather_reference": (
        "data/raw/weather/nasa_power_hyderabad_telangana_hourly_2019_may.csv"
    ),
    "site_weather_basis": "CITY_GRID_CELL_REANALYSIS_NOT_HOUSE_SITE",
    "guntur_weather_used_for_reside": False,
    "remaining_checks": [
        "Measurement units",
        "Sensor-to-house mapping",
        "Meaning and construction of AC Status labels",
        "Possible Envilog clock drift of a few minutes within the day",
        "Authors have not been asked to confirm the 2019 versus 2021 year",
    ],
}

sources = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
if not isinstance(sources, dict):
    raise ValueError("Source registry must be a YAML mapping.")
if "reside_ac" not in sources:
    raise KeyError("Missing registry entry: reside_ac")

# Validate every referenced path before touching the registry.
for field in ["observation_year_report", "site_weather_reference"]:
    path = ROOT / update[field]
    if not path.exists():
        raise FileNotFoundError(f"{path}\nRegistry has not been changed.")

before = dict(sources["reside_ac"])
backup = REGISTRY.with_name(
    "sources_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".yaml"
)
shutil.copy2(REGISTRY, backup)

sources["reside_ac"].update(update)
sources["reside_ac"]["release_ready"] = False

temporary = REGISTRY.with_suffix(".yaml.partial")
temporary.write_text(
    yaml.safe_dump(sources, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
temporary.replace(REGISTRY)

print("RESIDE REGISTRY ENTRY UPDATED")
print("Backup:", backup.name)
print("status:", before["status"], "->", sources["reside_ac"]["status"])
print("observation_year_supported:", sources["reside_ac"]["observation_year_supported"])
print("release_ready:", sources["reside_ac"]["release_ready"])
print("Remaining checks:", len(sources["reside_ac"]["remaining_checks"]))
