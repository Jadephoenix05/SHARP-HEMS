from pathlib import Path
from datetime import datetime
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data_registry/sources.yaml"

updates = {
    "ires_2020": {
        "status": "PROCESSED_SELECTED_SEMANTIC_CHECKS_PASSED",
        "processed_path": (
            "data/interim/indian_priors/"
            "ires_household_priors_v1.parquet"
        ),
        "validation_report": (
            "reports/ires_household_priors_validation_v1.csv"
        ),
        "remaining_checks": [
            "Full semantic review and release suitability",
        ],
    },
    "tus_2024": {
        "status": "AP_DIARY_CANDIDATES_BUILT_PENDING_REVIEW",
        "processed_path": (
            "data/interim/occupancy_profiles/"
            "tus2024_ap_complete_location_diaries_v1.parquet"
        ),
        "validation_report": "reports/tus2024_ap_location_validation.csv",
        "remaining_checks": [
            "Survey weighting",
            "Activity-location mapping and overlapping activity interpretation",
            "Selection effects from retaining complete diaries",
            "Source attribution and redistribution terms",
        ],
    },
    "iawe": {
        "status": "DESCRIPTIVE_SUMMARIES_BUILT_PENDING_REVIEW",
        "processed_path": "data/interim/iawe/15min",
        "validation_report": "reports/iawe_15min_validation_v1.csv",
        "remaining_checks": [
            "Mirror provenance and redistribution terms",
            "Relationship between the two mains channels",
            "Calibration suitability of sparse appliance channels",
            "Irregular water-motor sampling",
        ],
    },
    "reside_ac": {
        "status": "TIME_ALIGNED_CANDIDATES_BUILT_PENDING_REVIEW",
        "processed_path": "data/interim/reside_ac_clean/timezone_aligned",
        "validation_report": "reports/reside_ac_time_validation_v1.csv",
        "remaining_checks": [
            "CSV year 2019 versus description year 2021",
            "Measurement units",
            "Sensor-to-house mapping",
            "Meaning and construction of AC Status labels",
        ],
    },
}

sources = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
if not isinstance(sources, dict):
    raise ValueError("Source registry must be a YAML mapping.")

# Validate all paths before changing the registry.
for source_id, update in updates.items():
    if source_id not in sources:
        raise KeyError(f"Missing registry entry: {source_id}")

    for field in ["processed_path", "validation_report"]:
        path = ROOT / update[field]
        if not path.exists():
            raise FileNotFoundError(
                f"{source_id}: {path}\n"
                "Registry has not been changed."
            )
        if path.is_dir() and not any(path.glob("*.parquet")):
            raise ValueError(f"No Parquet files found in {path}")

backup = REGISTRY.with_name(
    "sources_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".yaml"
)
shutil.copy2(REGISTRY, backup)

for source_id, update in updates.items():
    sources[source_id].update(update)
    sources[source_id]["release_ready"] = False

temporary = REGISTRY.with_suffix(".yaml.partial")
temporary.write_text(
    yaml.safe_dump(sources, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
temporary.replace(REGISTRY)

print("MASTER REGISTRY UPDATED")
print("Backup:", backup)
for source_id in updates:
    print(source_id, "->", sources[source_id]["status"])
print("\nOther source entries were preserved.")