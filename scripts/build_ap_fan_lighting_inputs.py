from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/household_templates"
REPORTS = ROOT / "reports"

FIELDS = {
    "q405_a_incadescent_bulb_no": "incandescent_bulb_count",
    "q405_b_cfl_bulb_no": "cfl_bulb_count",
    "q405_c_led_bulb_no": "led_bulb_count",
    "q405_d_led_tube_light_no": "led_tube_count",
    "q405_e_cfl_tube_light_no": "cfl_tube_count",
    "q410_ceiling_fan_no": "ceiling_fan_count",
    "q411_celing_fan_use_months_no": "ceiling_fan_months_per_year",
    "q414_b_bee_ceiling_fans_no": "star_rated_ceiling_fan_count",
    "q416_table_fan_no": "table_fan_count",
    "q417_table_fan_use_months_no": "table_fan_months_per_year",
}
for number in range(1, 9):
    FIELDS[f"q412_fan_{number}_hrs"] = f"ceiling_fan_{number}_hours_daily"
for number in range(1, 4):
    FIELDS[f"q418_table_fan_{number}_hrs"] = f"table_fan_{number}_hours_daily"


def main():
    raw = pd.read_csv(
        ROOT / "data/raw/ires/ires_2020_data.tab",
        sep="\t",
        usecols=["hhid", "s_name"] + list(FIELDS),
        dtype={"hhid": "string"},
        low_memory=False,
    )
    raw = raw.loc[
        raw["s_name"].str.strip().str.casefold().eq("andhra pradesh")
    ].copy()

    if raw["hhid"].isna().any():
        raise ValueError("Missing household IDs")

    raw["profile_id"] = raw["hhid"].map(
        lambda value: "ires_" + hashlib.sha256(
            f"IRES2020:{value}".encode("utf-8")
        ).hexdigest()[:16]
    )

    templates = pd.read_parquet(
        OUT / "ap_household_templates_v2.parquet",
        columns=["profile_id", "template_id"],
    )
    if raw["profile_id"].duplicated().any():
        raise ValueError("Duplicate source profile IDs")
    if templates["profile_id"].duplicated().any():
        raise ValueError("Duplicate template profile IDs")
    if set(raw["profile_id"]) != set(templates["profile_id"]):
        raise ValueError("Source/template profile mismatch")

    result = raw[["profile_id"]].merge(
        templates, on="profile_id", validate="one_to_one"
    ).set_index("profile_id")
    raw = raw.set_index("profile_id")

    audit = []
    for source_column, field in FIELDS.items():
        source_values = raw[source_column]
        numeric = pd.to_numeric(source_values, errors="coerce")
        parse_failure = source_values.notna() & numeric.isna()

        if field.endswith("_hours_daily"):
            valid = numeric.between(0, 24)
            rule = "0 to 24 hours"
        elif field.endswith("_months_per_year"):
            valid = numeric.between(0, 12)
            rule = "0 to 12 months"
        else:
            valid = numeric.ge(0) & numeric.mod(1).eq(0)
            rule = "Nonnegative integer; special codes not yet verified"

        flagged = source_values.notna() & ~valid
        result[field + "_source"] = source_values
        result[field + "_candidate"] = numeric.where(valid)
        result[field + "_flagged"] = flagged

        audit.append({
            "field": field,
            "source_column": source_column,
            "nonmissing_source_rows": int(source_values.notna().sum()),
            "missing_source_rows": int(source_values.isna().sum()),
            "numeric_parse_failures": int(parse_failure.sum()),
            "flagged_rows": int(flagged.sum()),
            "candidate_min": numeric.where(valid).min(),
            "candidate_max": numeric.where(valid).max(),
            "range_rule": rule,
        })

    result = result.reset_index()
    result["source_dataset"] = "IRES 2020"
    result["is_synthetic"] = False
    result["semantic_status"] = "SOURCE_FIELDS_PRESERVED_RANGE_SCREEN_ONLY"

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    result.to_parquet(
        OUT / "ap_fan_lighting_survey_inputs_v1.parquet",
        index=False,
    )
    report = pd.DataFrame(audit)
    report.to_csv(
        REPORTS / "ap_fan_lighting_inputs_validation_v1.csv",
        index=False,
    )

    print("Matched household templates:", len(result))
    print(report.to_string(index=False))
    print("\nSaved: ap_fan_lighting_survey_inputs_v1.parquet")
    print("Missing responses preserved; no wattages or schedules assigned.")


if __name__ == "__main__":
    main()