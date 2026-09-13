from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/household_templates"

SOURCE_COLUMNS = {
    "incandescent_bulb": "q405_a_incadescent_bulb_no",
    "cfl_bulb": "q405_b_cfl_bulb_no",
    "cfl_tube": "q405_e_cfl_tube_light_no",
}


def main():
    inventory = pd.read_parquet(OUT / "ap_appliance_ownership_v2.parquet")
    lights = pd.read_parquet(OUT / "ap_lighting_inventory_v1.parquet")
    templates = pd.read_parquet(OUT / "ap_household_templates_v2.parquet")

    keys = ["template_id", "appliance_type"]
    if inventory.duplicated(keys).any() or lights.duplicated(keys).any():
        raise ValueError("Duplicate inventory keys")
    if set(lights["template_id"]) != set(templates["template_id"]):
        raise ValueError("Lighting/template mismatch")

    # Check the two lighting categories already present.
    existing = inventory.loc[
        inventory["appliance_type"].isin(["led_bulb", "led_tube"]),
        keys + ["quantity"],
    ].merge(
        lights[keys + ["reported_count"]],
        on=keys, validate="one_to_one",
    )
    if len(existing) != len(templates) * 2:
        raise ValueError("Missing existing LED categories")
    if not existing["quantity"].eq(existing["reported_count"]).fillna(False).all():
        raise ValueError("Existing LED quantities disagree")

    selected = lights.loc[
        lights["appliance_type"].isin(SOURCE_COLUMNS)
    ].copy()
    count = pd.to_numeric(selected["reported_count"], errors="raise")
    if count.isna().any() or not (
        count.ge(0) & count.mod(1).eq(0)
    ).all():
        raise ValueError("Invalid lighting counts")

    # Obtain provenance through a validated profile join.
    selected = selected.merge(
        templates[["profile_id", "source_dataset"]],
        on="profile_id", validate="many_to_one",
    )

    additions = pd.DataFrame({
        "template_id": selected["template_id"],
        "source_profile_id": selected["profile_id"],
        "appliance_type": selected["appliance_type"],
        "available": selected["reported_count"].gt(0).astype("Int8"),
        "quantity": selected["reported_count"].astype("Int64"),
        "source_count_value": selected["reported_count"],
        "ownership_basis": "DERIVED_FROM_COUNT_IN_HOME",
        "quantity_basis": "SOURCE_COUNT_FIELD",
        "quantity_source_column": selected["appliance_type"].map(SOURCE_COLUMNS),
        "ownership_count_conflict": False,
        "invalid_count": False,
        "inventory_status": selected["reported_count"].map(
            lambda value: "OWNED_QUANTITY_KNOWN" if value > 0 else "NOT_OWNED"
        ),
        "control_permission": "UNASSIGNED",
        "trace_assignment": "UNASSIGNED",
        "source_dataset": selected["source_dataset"],
        "is_synthetic": False,
        "source_count_semantics": "NUMBER_IN_HOME",
    })

    result = pd.concat([inventory, additions], ignore_index=True)
    if result.duplicated(keys).any():
        raise ValueError("Duplicate consolidated keys")
    if not result.groupby("template_id").size().eq(26).all():
        raise ValueError("Expected 26 categories per template")

    result["available"] = result["available"].astype("Int8")
    result["quantity"] = result["quantity"].astype("Int64")
    destination = OUT / "ap_appliance_ownership_v3.parquet"
    result.to_parquet(destination, index=False)

    print("Household templates:", result["template_id"].nunique())
    print("Appliance types:", result["appliance_type"].nunique())
    print("Inventory rows:", len(result))
    print("Saved:", destination)


if __name__ == "__main__":
    main()