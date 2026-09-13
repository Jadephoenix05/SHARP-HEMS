from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

BLOCK_FILE = ROOT / "data/interim/emarc/emarc_load_blocks_native_v1.parquet"
ARCHIVE = ROOT / "data/raw/emarc/prayas-energy.zip"


def main():
    print("Reading block measurements...", flush=True)
    data = pd.read_parquet(
        BLOCK_FILE,
        columns=["deployment_id", "source_date", "block", "load_kw"],
    )

    keys = ["deployment_id", "source_date", "block"]
    duplicated = data.duplicated(keys, keep=False)
    duplicate_rows = int(duplicated.sum())

    if duplicate_rows:
        audit = (
            data.loc[duplicated]
            .groupby(keys, observed=True)
            .agg(
                rows=("load_kw", "size"),
                distinct_values=("load_kw", "nunique"),
                minimum_kw=("load_kw", "min"),
                maximum_kw=("load_kw", "max"),
            )
            .reset_index()
        )
    else:
        audit = pd.DataFrame(columns=keys + [
            "rows", "distinct_values", "minimum_kw", "maximum_kw"
        ])

    audit.to_csv(REPORTS / "emarc_duplicate_blocks_v1.csv", index=False)

    # Only unique-key days enter the energy comparison.
    # Duplicate records remain untouched in the source Parquet.
    data["duplicate_key"] = duplicated
    days = (
        data.groupby(["deployment_id", "source_date"], observed=True)
        .agg(
            rows=("block", "size"),
            distinct_blocks=("block", "nunique"),
            duplicate_key_rows=("duplicate_key", "sum"),
            sum_load_kw=("load_kw", "sum"),
        )
        .reset_index()
    )
    del data

    days["complete_unique_day"] = (
        days["rows"].eq(96)
        & days["distinct_blocks"].eq(96)
        & days["duplicate_key_rows"].eq(0)
    )
    days["candidate_energy_kwh"] = (
        days["sum_load_kw"] * 0.25
    ).where(days["complete_unique_day"])

    print("Reading supplied daily consumption...", flush=True)
    with ZipFile(ARCHIVE) as archive:
        with archive.open(
            "Prayas Energy/eMAR daily consumption.csv"
        ) as source:
            daily = pd.read_csv(
                source, dtype={"deployment_id": str}
            )

    daily.columns = daily.columns.str.strip()
    daily["deployment_id"] = daily["deployment_id"].str.strip()
    daily["source_date"] = pd.to_datetime(
        daily["Date"], format="%m/%d/%Y", errors="coerce"
    )
    daily["reported_energy_kwh"] = pd.to_numeric(
        daily["Daily consumption (kWh)"], errors="coerce"
    )

    invalid_dates = int(daily["source_date"].isna().sum())
    invalid_energy = (
        daily["reported_energy_kwh"].isna()
        | ~np.isfinite(daily["reported_energy_kwh"])
        | daily["reported_energy_kwh"].lt(0)
    )
    daily_keys = ["deployment_id", "source_date"]
    daily_duplicates = daily.duplicated(daily_keys, keep=False)

    daily.loc[daily_duplicates].to_csv(
        REPORTS / "emarc_duplicate_daily_records_v1.csv",
        index=False,
    )

    eligible = daily.loc[
        ~daily_duplicates
        & ~invalid_energy
        & daily["source_date"].notna(),
        daily_keys + ["reported_energy_kwh"],
    ]

    comparison = days.merge(
        eligible,
        on=daily_keys,
        how="left",
        validate="one_to_one",
    )
    paired = (
        comparison["complete_unique_day"]
        & comparison["reported_energy_kwh"].notna()
    )

    comparison["absolute_difference_kwh"] = (
        comparison["candidate_energy_kwh"]
        - comparison["reported_energy_kwh"]
    ).abs().where(paired)

    # Diagnostic tolerance, not an official source specification.
    tolerance = np.maximum(
        0.01, comparison["reported_energy_kwh"].abs() * 0.01
    )
    comparison["within_diagnostic_tolerance"] = (
        comparison["absolute_difference_kwh"] <= tolerance
    ).astype("boolean").where(paired)

    comparison.to_csv(
        REPORTS / "emarc_daily_reconciliation_v1.csv",
        index=False,
    )

    differences = comparison.loc[paired, "absolute_difference_kwh"]
    report = {
        "status": "AUDIT_COMPLETE_REVIEW_RESULTS",
        "duplicate_block_rows": duplicate_rows,
        "duplicate_block_keys": len(audit),
        "conflicting_block_keys": int(
            audit["distinct_values"].gt(1).sum()
        ),
        "observed_deployment_days": len(days),
        "complete_unique_days": int(days["complete_unique_day"].sum()),
        "daily_file_rows": len(daily),
        "daily_invalid_dates": invalid_dates,
        "daily_invalid_or_missing_energy": int(invalid_energy.sum()),
        "daily_duplicate_key_rows": int(daily_duplicates.sum()),
        "paired_complete_days": int(paired.sum()),
        "days_within_tolerance": int(
            comparison["within_diagnostic_tolerance"].fillna(False).sum()
        ),
        "median_absolute_difference_kwh": differences.median(),
        "maximum_absolute_difference_kwh": differences.max(),
        "tolerance": "max(0.01 kWh, 1 percent of reported daily kWh)",
        "clock_mapping": "NOT_ESTABLISHED_BY_THIS_CHECK",
    }

    pd.DataFrame(
        report.items(), columns=["metric", "value"]
    ).to_csv(REPORTS / "emarc_daily_validation_v1.csv", index=False)

    print("\nEMARC DAILY VALIDATION")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()