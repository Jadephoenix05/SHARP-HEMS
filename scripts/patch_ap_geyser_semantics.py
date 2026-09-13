from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/household_templates"
REPORTS = ROOT / "reports"


def main():
    columns = [
        "hhid", "s_name", "q450_hot_water_bath_yn",
        "q451_geyser_use_1", "q453_geyser_no",
    ]
    raw = pd.read_csv(
        ROOT / "data/raw/ires/ires_2020_data.tab",
        sep="\t",
        usecols=columns,
        dtype={"hhid": "string"},
        low_memory=False,
    )
    raw = raw.loc[
        raw["s_name"].str.strip().str.casefold().eq("andhra pradesh")
    ].copy()

    if raw["hhid"].isna().any():
        raise ValueError("Missing source household IDs")

    raw["profile_id"] = raw["hhid"].map(
        lambda value: "ires_" + hashlib.sha256(
            f"IRES2020:{value}".encode("utf-8")
        ).hexdigest()[:16]
    ).astype("string")

    if raw["profile_id"].duplicated().any():
        raise ValueError("Duplicate profile IDs")

    households = pd.read_parquet(OUT / "ap_household_templates_v1.parquet")
    inventory = pd.read_parquet(OUT / "ap_appliance_ownership_v1.parquet")

    if households["profile_id"].duplicated().any():
        raise ValueError("Duplicate template profile IDs")
    if set(raw["profile_id"]) != set(households["profile_id"]):
        raise ValueError("Source and template profile IDs do not match")

    for column in columns[2:]:
        raw[column] = pd.to_numeric(raw[column], errors="raise")

    hot = raw["q450_hot_water_bath_yn"]
    use = raw["q451_geyser_use_1"]
    count = raw["q453_geyser_no"]

    reported = hot.eq(1) & use.eq(1) & count.eq(1)
    nonuse = hot.eq(1) & use.eq(0) & count.isna()
    skipped = hot.eq(0) & use.isna() & count.isna()

    if not (reported | nonuse | skipped).all():
        raise ValueError("Unexpected response combinations; review before patching")

    responses = raw[["profile_id"]].copy()
    responses["hot_water_bathing_source_code"] = hot
    responses["geyser_use_source_code"] = use
    responses["geyser_count_used_source"] = count
    responses["geyser_use_status"] = "UNCLASSIFIED"
    responses.loc[reported, "geyser_use_status"] = "REPORTED_USE"
    responses.loc[nonuse, "geyser_use_status"] = "EXPLICIT_NONUSE"
    responses.loc[skipped, "geyser_use_status"] = (
        "NO_HOT_WATER_FOR_BATHING_GEYSER_RESPONSE_MISSING"
    )
    responses["geyser_reported_units_used"] = count.astype("Int64")
    responses["geyser_ownership_status"] = "NOT_ESTABLISHED"

    households = households.merge(
        responses, on="profile_id", how="left", validate="one_to_one"
    )
    households["template_version"] = "v2"
    households["simulator_ready"] = False

    mask = inventory["appliance_type"].eq("geyser")
    if (
        int(mask.sum()) != len(households)
        or inventory.loc[mask, "source_profile_id"].duplicated().any()
        or set(inventory.loc[mask, "source_profile_id"])
        != set(responses["profile_id"])
    ):
        raise ValueError("Geyser inventory/profile mismatch")

    # These fields describe ownership, which the use questions do not establish.
    inventory.loc[mask, "available"] = pd.NA
    inventory.loc[mask, "quantity"] = pd.NA
    inventory.loc[mask, "ownership_basis"] = "NOT_ESTABLISHED_BY_USE_QUESTION"
    inventory.loc[mask, "quantity_basis"] = "OWNED_QUANTITY_UNKNOWN"
    inventory.loc[mask, "inventory_status"] = "OWNERSHIP_UNKNOWN"

    # The original source_count_value remains preserved for provenance.
    inventory["source_count_semantics"] = "SEE_SOURCE_COLUMN"
    inventory.loc[mask, "source_count_semantics"] = "NUMBER_OF_GEYSERS_USED"

    inventory = inventory.merge(
        responses.rename(columns={"profile_id": "source_profile_id"}),
        on="source_profile_id",
        how="left",
        validate="many_to_one",
    )

    # Geyser-specific responses belong only on geyser inventory rows.
    response_columns = list(responses.columns.drop("profile_id"))
    inventory.loc[~mask, response_columns] = pd.NA

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    households.to_parquet(
        OUT / "ap_household_templates_v2.parquet", index=False
    )
    inventory.to_parquet(
        OUT / "ap_appliance_ownership_v2.parquet", index=False
    )
    responses.to_parquet(
        OUT / "ap_geyser_use_by_profile_v1.parquet", index=False
    )

    report = {
        "status": "GEYSER_SEMANTICS_PATCHED",
        "matched_households": len(households),
        "reported_geyser_use": int(reported.sum()),
        "explicit_geyser_nonuse": int(nonuse.sum()),
        "geyser_question_unanswered": int(skipped.sum()),
        "geyser_ownership_not_established": int(mask.sum()),
        "inventory_rows": len(inventory),
        "simulator_ready": False,
    }
    pd.DataFrame(
        report.items(), columns=["metric", "value"]
    ).to_csv(REPORTS / "ap_geyser_semantics_validation_v1.csv", index=False)

    for key, value in report.items():
        print(f"{key}: {value}")
    print("\nUse the v2 household and ownership files for subsequent steps.")


if __name__ == "__main__":
    main()