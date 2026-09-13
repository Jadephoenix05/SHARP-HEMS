from pathlib import Path
import json
import yaml

folders = [
    "configs/tariffs",
    "data_registry/schemas",
    "data_registry/licenses",
    "release/SHARP_MASTER_DATASET_V1",
    "data/raw/bee_clasp",
    "data/raw/ires",
    "data/raw/emarc",
    "data/raw/iawe",
    "data/raw/reside_ac",
    "data/raw/tus_india/tus_2024",
    "data/raw/tariff_ap",
    "data/interim/indian_priors",
    "data/interim/emarc",
    "data/interim/iawe",
    "data/interim/reside_ac",
    "data/interim/occupancy_profiles",
    "data/processed/simulator_v1",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

sources = {
    "refit_cleaned": {
        "status": "PROCESSED_NEEDS_SCHEMA_UPGRADE",
        "role": "empirical_appliance_temporal_traces",
        "geography": "United Kingdom",
        "canonical_source": "University of Strathclyde",
        "doi": "10.15129/9ab14b0e-19ac-9f88a",
        "license": "CC-BY-4.0",
        "local_path": "data/raw/refit",
        "processed_path": "data/interim/refit_15min",
        "notes": [
            "20 houses processed",
            "921931 fifteen-minute intervals",
            "cleaned-release identity must be verified",
            "required canonical schema is not yet complete",
        ],
    },
    "bee_clasp_2024": {
        "status": "NOT_ACQUIRED",
        "role": "modern_indian_appliance_ownership_priors",
        "geography": "India",
        "canonical_source": "BEE and CLASP",
        "local_path": "data/raw/bee_clasp",
    },
    "ires_2020": {
        "status": "NOT_ACQUIRED",
        "role": "indian_household_access_reliability_priors",
        "geography": "India",
        "canonical_source": "Harvard Dataverse",
        "doi": "10.7910/DVN/U8NYUP",
        "local_path": "data/raw/ires",
    },
    "emarc": {
        "status": "NOT_ACQUIRED",
        "role": "indian_aggregate_load_calibration_validation",
        "geography": "India",
        "canonical_source": "Prayas Energy Group",
        "local_path": "data/raw/emarc",
    },
    "iced_hourly": {
        "status": "NOT_ACQUIRED",
        "role": "subdaily_grid_context",
        "geography": "Andhra Pradesh",
        "canonical_source": "India Climate and Energy Dashboard",
        "local_path": "data/raw/grid_india",
        "notes": [
            "Monthly peak and load-duration workbooks exist",
            "hourly demand curve is still required",
        ],
    },
    "nasa_power_guntur": {
        "status": "READY",
        "role": "weather_and_solar_context",
        "geography": "Guntur Andhra Pradesh",
        "canonical_source": "NASA POWER",
        "period": "2021-01-01 to 2025-12-31",
        "local_path": "data/raw/weather/nasa_power_guntur_ap_hourly_2021_2025.csv",
    },
    "aperc_apcpdcl_tariff": {
        "status": "PARTIAL",
        "role": "reward_cost_configuration",
        "geography": "Guntur Andhra Pradesh",
        "canonical_source": "Andhra Pradesh Electricity Regulatory Commission",
        "local_path": "data/raw/tariff_ap",
        "notes": [
            "ICED actual tariff workbook obtained",
            "official APERC tariff order and complete YAML still required",
        ],
    },
    "iawe": {
        "status": "NOT_ACQUIRED",
        "role": "indian_appliance_voltage_outage_calibration",
        "geography": "New Delhi India",
        "canonical_source": "iAWE",
        "local_path": "data/raw/iawe",
    },
    "reside_ac": {
        "status": "NOT_ACQUIRED",
        "role": "indian_ac_thermal_cycle_calibration",
        "geography": "Hyderabad India",
        "canonical_source": "Figshare",
        "license": "CC0",
        "local_path": "data/raw/reside_ac",
    },
    "tus_2024": {
        "status": "ACQUIRED_NOT_EXPORTED",
        "role": "occupancy_activity_attention_priors",
        "geography": "India",
        "canonical_source": "MoSPI NSO",
        "local_path": "data/raw/tus_india/tus_2024",
    },
    "grid_india_historical": {
        "status": "READY",
        "role": "grid_peak_shortage_validation",
        "geography": "Andhra Pradesh",
        "period": "2020-01-01 to 2025-03-31",
        "local_path": "data/raw/grid_india/India_Elec_data_(Jan2020-Mar2025).csv",
        "processed_path": "data/interim/grid_peak_labels/ap_grid_daily_ml_v1.parquet",
    },
    "sharp_rl_transitions": {
        "status": "PENDING_SIMULATOR",
        "role": "direct_bdq_training_data",
        "geography": "simulated_indian_households",
        "processed_path": "data/processed/simulator_v1",
    },
}

versions = {
    "dataset_name": "SHARP Master Dataset",
    "current_build_version": "0.1.0",
    "target_release_version": "1.0.0",
    "control_interval_minutes": 15,
    "random_seed": 42,
    "deployment_scenario": "APCPDCL_Guntur_Andhra_Pradesh",
    "release_status": "BUILDING",
}

manifest = {
    "dataset_name": "SHARP Master Dataset",
    "dataset_version": "1.0.0-draft",
    "control_interval_minutes": 15,
    "random_seed": 42,
    "deployment_scenario": "APCPDCL_Guntur_Andhra_Pradesh",
    "refit_release": "cleaned_pending_verification",
    "simulator_version": "not_built",
    "created_by": ["Supriya", "Pranitha"],
    "release_status": "building",
}

Path("data_registry/sources.yaml").write_text(
    yaml.safe_dump(sources, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

Path("data_registry/versions.yaml").write_text(
    yaml.safe_dump(versions, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

Path("data_registry/manifest_draft.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("MASTER REGISTRY INITIALIZED")
print("----------------------------------------")
print("Sources registered:", len(sources))
print("Build version: 0.1.0")
print("Target version: 1.0.0")
print("Next source: BEE/CLASP 2024")