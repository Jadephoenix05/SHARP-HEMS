from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/household_templates"
REPORTS = ROOT / "reports"


def main():
    source = pd.read_parquet(
        OUT / "ap_fan_lighting_survey_inputs_v1.parquet"
    )
    households = pd.read_parquet(
        OUT / "ap_household_templates_v2.parquet"
    )

    if source["profile_id"].duplicated().any():
        raise ValueError("Duplicate source profiles")
    if households["profile_id"].duplicated().any():
        raise ValueError("Duplicate household profiles")
    if set(source["profile_id"]) != set(households["profile_id"]):
        raise ValueError("Profile IDs do not match")

    corrected = households[[
        "profile_id",
        "ceiling_fan_count_for_profile",
        "table_fan_count_for_profile",
    ]]
    data = source.merge(
        corrected, on="profile_id", validate="one_to_one"
    )

    fan_rows = []
    lighting_rows = []
    issues = []

    for _, row in data.iterrows():
        for kind, maximum in [("ceiling_fan", 8), ("table_fan", 3)]:
            count = row[f"{kind}_count_for_profile"]
            known_count = (
                pd.notna(count)
                and count >= 0
                and float(count).is_integer()
                and count <= maximum
            )

            if not known_count:
                issues.append({
                    "profile_id": row["profile_id"],
                    "appliance_type": kind,
                    "issue": "Missing or unsupported count",
                })
                continue

            count = int(count)

            for number in range(1, maximum + 1):
                hours = row[f"{kind}_{number}_hours_daily_candidate"]

                if number > count:
                    if pd.notna(hours):
                        issues.append({
                            "profile_id": row["profile_id"],
                            "appliance_type": kind,
                            "issue": f"Hours reported for excess fan {number}",
                        })
                    continue

                if pd.isna(hours):
                    issues.append({
                        "profile_id": row["profile_id"],
                        "appliance_type": kind,
                        "issue": f"Missing hours for fan {number}",
                    })

                fan_rows.append({
                    "profile_id": row["profile_id"],
                    "template_id": row["template_id"],
                    "appliance_type": kind,
                    "survey_fan_number": number,
                    "reported_hours_daily": hours,
                    "reported_months_per_year": row[
                        f"{kind}_months_per_year_candidate"
                    ],
                    "usage_basis": "SURVEY_REPORTED",
                    "rated_power_w": None,
                    "schedule_status": "NOT_ASSIGNED",
                    "is_synthetic": False,
                })

        for kind in [
            "incandescent_bulb", "cfl_bulb", "led_bulb",
            "led_tube", "cfl_tube",
        ]:
            lighting_rows.append({
                "profile_id": row["profile_id"],
                "template_id": row["template_id"],
                "appliance_type": kind,
                "reported_count": row[f"{kind}_count_candidate"],
                "source_column_semantics": "NUMBER_IN_HOME",
                "rated_power_w": None,
                "schedule_status": "NOT_ASSIGNED",
                "is_synthetic": False,
            })

    fans = pd.DataFrame(fan_rows)
    lights = pd.DataFrame(lighting_rows)
    issue_table = pd.DataFrame(
        issues, columns=["profile_id", "appliance_type", "issue"]
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    issue_table.to_csv(
        REPORTS / "ap_fan_count_hours_issues_v1.csv", index=False
    )

    fans.to_parquet(OUT / "ap_fan_usage_units_v1.parquet", index=False)
    lights.to_parquet(
        OUT / "ap_lighting_inventory_v1.parquet", index=False
    )

    print("Households:", len(data))
    print("Fan usage records:", len(fans))
    print("Lighting category records:", len(lights))
    print("Count/hour issues:", len(issues))
    print("\nFan records by type:")
    print(fans.groupby("appliance_type").size().to_string())
    print("\nLighting counts summed across the AP sample:")
    print(
        lights.groupby("appliance_type")["reported_count"]
        .sum(min_count=1).to_string()
    )
    print("\nPower ratings and clock schedules remain unassigned.")


if __name__ == "__main__":
    main()