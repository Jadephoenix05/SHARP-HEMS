from pathlib import Path
import pandas as pd
import yaml

SOURCE_URL = (
    "https://www.clasp.ngo/wp-content/uploads/2026/05/"
    "Report_Residential_Energy_Consumption_Survey_India.pdf"
)

OUTPUT = Path(
    r"data\interim\indian_priors\bee_clasp_national_priors_v1.csv"
)
REPORT = Path(r"reports\bee_clasp_priors_validation_v1.csv")
SOURCES_YAML = Path(r"data_registry\sources.yaml")

rows = [
    # Survey metadata
    {
        "variable": "survey_household_count",
        "appliance_or_end_use": "all",
        "value": 4321,
        "unit": "households",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 8,
        "source_figure": "Executive Summary",
        "notes": "Survey sample size",
    },
    {
        "variable": "survey_state_count",
        "appliance_or_end_use": "all",
        "value": 20,
        "unit": "states",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 8,
        "source_figure": "Executive Summary",
        "notes": "States represented in survey",
    },

    # Appliance ownership: Figure 1
    {
        "variable": "ownership_rate",
        "appliance_or_end_use": "led_lamp",
        "value": 100,
        "unit": "percent_households",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 1",
        "notes": "Reported as nearly all/all surveyed households",
    },
    {
        "variable": "ownership_rate",
        "appliance_or_end_use": "fan",
        "value": 99,
        "unit": "percent_households",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 1",
        "notes": "",
    },
    {
        "variable": "ownership_rate",
        "appliance_or_end_use": "television",
        "value": 55,
        "unit": "percent_households",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 1",
        "notes": "",
    },
    {
        "variable": "ownership_rate",
        "appliance_or_end_use": "washing_machine",
        "value": 22,
        "unit": "percent_households",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 1",
        "notes": "",
    },
    {
        "variable": "ownership_rate",
        "appliance_or_end_use": "air_conditioner",
        "value": 13,
        "unit": "percent_households",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 1",
        "notes": "",
    },
    {
        "variable": "ownership_rate",
        "appliance_or_end_use": "air_cooler",
        "value": 11,
        "unit": "percent_households",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 1",
        "notes": "",
    },
    {
        "variable": "ownership_rate",
        "appliance_or_end_use": "water_heater_geyser",
        "value": 11,
        "unit": "percent_households",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 1",
        "notes": "Report figure spells this as Gysers",
    },
    {
        "variable": "ownership_rate",
        "appliance_or_end_use": "induction_cooktop",
        "value": 4,
        "unit": "percent_households",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 1",
        "notes": "",
    },
    {
        "variable": "ownership_rate",
        "appliance_or_end_use": "microwave",
        "value": 4,
        "unit": "percent_households",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 1",
        "notes": "",
    },

    # Annual electricity use by end use: Figure 2
    {
        "variable": "annual_electricity_end_use_share",
        "appliance_or_end_use": "thermal_comfort",
        "value": 40,
        "unit": "percent_electricity",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 2",
        "notes": "",
    },
    {
        "variable": "annual_electricity_end_use_share",
        "appliance_or_end_use": "kitchen",
        "value": 28,
        "unit": "percent_electricity",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 2",
        "notes": "",
    },
    {
        "variable": "annual_electricity_end_use_share",
        "appliance_or_end_use": "lighting",
        "value": 11,
        "unit": "percent_electricity",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 2",
        "notes": "",
    },
    {
        "variable": "annual_electricity_end_use_share",
        "appliance_or_end_use": "other",
        "value": 21,
        "unit": "percent_electricity",
        "geography": "India",
        "survey_year": 2024,
        "source_page": 9,
        "source_figure": "Figure 2",
        "notes": "",
    },
]

df = pd.DataFrame(rows)
df.insert(0, "prior_id", [f"bee_clasp_{i:03d}" for i in range(1, len(df) + 1)])
df["source_title"] = (
    "Residential Energy Consumption Patterns and Appliance Ownership "
    "in India: Insights From a 2024 Household Survey"
)
df["source_url"] = SOURCE_URL
df["license"] = "CC-BY-SA-4.0"

ownership = df[df["variable"] == "ownership_rate"]
end_use = df[df["variable"] == "annual_electricity_end_use_share"]

checks = {
    "total_rows": len(df),
    "ownership_rows": len(ownership),
    "end_use_rows": len(end_use),
    "end_use_sum_percent": end_use["value"].sum(),
    "ownership_values_in_range": bool(
        ownership["value"].between(0, 100).all()
    ),
    "duplicate_prior_ids": int(df["prior_id"].duplicated().sum()),
    "missing_provenance": int(
        df[
            ["source_page", "source_figure", "source_url", "license"]
        ].isna().any(axis=1).sum()
    ),
}

status = (
    "PASS"
    if checks["total_rows"] == 15
    and checks["ownership_rows"] == 9
    and checks["end_use_rows"] == 4
    and checks["end_use_sum_percent"] == 100
    and checks["ownership_values_in_range"]
    and checks["duplicate_prior_ids"] == 0
    and checks["missing_provenance"] == 0
    else "FAIL"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT, index=False)

report_rows = [{"metric": "status", "value": status}]
report_rows.extend(
    {"metric": key, "value": value}
    for key, value in checks.items()
)
pd.DataFrame(report_rows).to_csv(REPORT, index=False)

with SOURCES_YAML.open("r", encoding="utf-8") as f:
    sources = yaml.safe_load(f)

sources["bee_clasp_2024"]["status"] = "BASE_PRIORS_READY"
sources["bee_clasp_2024"]["processed_path"] = str(OUTPUT).replace("\\", "/")
sources["bee_clasp_2024"]["license"] = "CC-BY-SA-4.0"
sources["bee_clasp_2024"]["notes"] = [
    "Contains published national statistics only",
    "Does not contain fabricated household microdata",
    "Climate and household-class conditional priors may be added later",
]

with SOURCES_YAML.open("w", encoding="utf-8") as f:
    yaml.safe_dump(sources, f, sort_keys=False, allow_unicode=True)

print("BEE/CLASP PRIORS BUILD")
print("-" * 45)
print("Status:", status)
print("Rows:", len(df))
print("Ownership priors:", len(ownership))
print("End-use share total:", end_use["value"].sum())
print("Output:", OUTPUT)
print("Report:", REPORT)