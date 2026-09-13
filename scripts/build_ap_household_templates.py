from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/indian_priors/ires_andhra_pradesh_priors_v1.parquet"
OUT = ROOT / "data/processed/household_templates"
REPORTS = ROOT / "reports"

# appliance: (ownership field, corrected quantity field)
APPLIANCES = {
    "ceiling_fan": ("ceiling_fan_available", "ceiling_fan_count_for_profile"),
    "table_fan": ("table_fan_available", "table_fan_count_for_profile"),
    "air_cooler": ("air_cooler_available", "air_cooler_count_for_profile"),
    "air_conditioner": ("air_conditioner_available", "air_conditioner_count_for_profile"),
    "television": ("television_available", "television_count_for_profile"),
    "refrigerator": ("refrigerator_available", "refrigerator_count_for_profile"),
    "led_bulb": (None, "led_bulb_count"),
    "led_tube": (None, "led_tube_count"),
    "geyser": (None, "geyser_count"),
    "desktop": ("desktop_available", None),
    "laptop_tablet": ("laptop_tablet_available", None),
    "modem_router": ("modem_router_available", None),
    "water_purifier": ("water_purifier_available", None),
    "mixer_grinder": ("mixer_grinder_available", None),
    "electric_kettle": ("electric_kettle_available", None),
    "washing_machine": ("washing_machine_available", None),
    "electric_iron": ("electric_iron_available", None),
    "water_pump": ("water_pump_available", None),
    "electric_coil": ("electric_coil_available", None),
    "induction_cooktop": ("induction_cooktop_available", None),
    "microwave": ("microwave_available", None),
    "electric_grill_toaster": ("electric_grill_toaster_available", None),
    "electric_rice_cooker": ("electric_rice_cooker_available", None),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    households = pd.read_parquet(SOURCE)

    if households["profile_id"].isna().any():
        raise ValueError("Missing profile IDs")
    if households["profile_id"].duplicated().any():
        raise ValueError("Duplicate profile IDs")
    if not households["state"].str.strip().str.casefold().eq(
        "andhra pradesh"
    ).all():
        raise ValueError("Unexpected state")
    if not households["is_synthetic"].eq(False).all():
        raise ValueError("Expected survey-derived profiles")

    households["template_id"] = "ires_ap_" + households["profile_id"].astype(str)
    households["template_version"] = "v1"
    households["template_basis"] = "SURVEY_DERIVED"
    households["target_location_assignment"] = "UNASSIGNED"
    households["simulator_ready"] = False

    records = []

    for _, household in households.iterrows():
        for appliance, (ownership_column, count_column) in APPLIANCES.items():
            ownership = (
                household[ownership_column]
                if ownership_column else pd.NA
            )
            raw_count = (
                household[count_column]
                if count_column else pd.NA
            )

            count_valid = (
                pd.notna(raw_count)
                and float(raw_count) >= 0
                and float(raw_count).is_integer()
            )
            quantity = int(raw_count) if count_valid else pd.NA
            invalid_count = pd.notna(raw_count) and not count_valid

            if pd.notna(ownership) and ownership in (0, 1):
                available = int(ownership)
                basis = "SOURCE_OWNERSHIP_FIELD"
            elif ownership_column is None and count_valid:
                available = int(quantity > 0)
                basis = "DERIVED_FROM_COUNT"
            else:
                available = pd.NA
                basis = "UNKNOWN"

            conflict = False
            if pd.notna(available) and count_valid:
                conflict = (
                    (available == 0 and quantity > 0)
                    or (available == 1 and quantity == 0)
                )

            quantity_basis = (
                "SOURCE_COUNT_FIELD" if count_valid else "UNKNOWN"
            )
            if pd.notna(available) and available == 0 and pd.isna(quantity):
                quantity = 0
                quantity_basis = "EXPLICIT_NONOWNERSHIP"

            if conflict or invalid_count:
                status = "REVIEW_REQUIRED"
            elif pd.isna(available):
                status = "OWNERSHIP_UNKNOWN"
            elif available == 0:
                status = "NOT_OWNED"
            elif pd.isna(quantity):
                status = "OWNED_QUANTITY_UNKNOWN"
            else:
                status = "OWNED_QUANTITY_KNOWN"

            records.append({
                "template_id": household["template_id"],
                "source_profile_id": household["profile_id"],
                "appliance_type": appliance,
                "available": available,
                "quantity": quantity,
                "source_count_value": raw_count,
                "ownership_basis": basis,
                "quantity_basis": quantity_basis,
                "ownership_source_column": ownership_column,
                "quantity_source_column": count_column,
                "ownership_count_conflict": conflict,
                "invalid_count": invalid_count,
                "inventory_status": status,
                "control_permission": "UNASSIGNED",
                "trace_assignment": "UNASSIGNED",
                "source_dataset": household["source_dataset"],
                "is_synthetic": False,
            })

    appliances = pd.DataFrame(records)
    appliances["available"] = appliances["available"].astype("Int8")
    appliances["quantity"] = appliances["quantity"].astype("Int64")
    appliances["source_count_value"] = pd.to_numeric(
        appliances["source_count_value"], errors="raise"
    ).astype("Float64")

    if appliances.duplicated(["template_id", "appliance_type"]).any():
        raise ValueError("Duplicate inventory keys")

    households.to_parquet(
        OUT / "ap_household_templates_v1.parquet", index=False
    )
    appliances.to_parquet(
        OUT / "ap_appliance_ownership_v1.parquet", index=False
    )

    review_count = int(
        appliances["inventory_status"].eq("REVIEW_REQUIRED").sum()
    )
    metrics = {
        "status": (
            "BUILT_WITH_REVIEW_FLAGS"
            if review_count else "TEMPLATES_BUILT"
        ),
        "household_templates": len(households),
        "appliance_types_per_template": len(APPLIANCES),
        "ownership_table_rows": len(appliances),
        "review_required_rows": review_count,
        "simulator_ready": False,
        "geographic_scope": "Andhra Pradesh survey sample",
        "guntur_household_assignment": "NOT_MADE",
    }
    for status, count in appliances["inventory_status"].value_counts().items():
        metrics[status.lower() + "_rows"] = int(count)

    pd.DataFrame(
        metrics.items(), columns=["metric", "value"]
    ).to_csv(
        REPORTS / "ap_household_templates_validation_v1.csv",
        index=False,
    )

    print("\nAP HOUSEHOLD TEMPLATES BUILT")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("Output:", OUT)


if __name__ == "__main__":
    main()