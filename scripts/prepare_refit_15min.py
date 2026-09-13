from pathlib import Path
import re
import pandas as pd

INPUT_DIR = Path("data/raw/refit")
OUTPUT_DIR = Path("data/interim/refit_15min")
REPORT_PATH = Path("reports/refit_15min_validation_v1.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

POWER_COLUMNS = [
    "Aggregate",
    "Appliance1",
    "Appliance2",
    "Appliance3",
    "Appliance4",
    "Appliance5",
    "Appliance6",
    "Appliance7",
    "Appliance8",
    "Appliance9",
]

COLUMN_NAMES = {
    "Aggregate": "aggregate_power_w",
    "Appliance1": "appliance_1_power_w",
    "Appliance2": "appliance_2_power_w",
    "Appliance3": "appliance_3_power_w",
    "Appliance4": "appliance_4_power_w",
    "Appliance5": "appliance_5_power_w",
    "Appliance6": "appliance_6_power_w",
    "Appliance7": "appliance_7_power_w",
    "Appliance8": "appliance_8_power_w",
    "Appliance9": "appliance_9_power_w",
}

files = sorted(
    INPUT_DIR.glob("House_*.csv"),
    key=lambda path: int(
        re.search(r"House_(\d+)", path.name).group(1)
    ),
)

if len(files) != 20:
    raise RuntimeError(
        f"Expected 20 canonical REFIT files, found {len(files)}."
    )

reports = []

for file_number, path in enumerate(files, start=1):
    house_id = int(
        re.search(r"House_(\d+)", path.name).group(1)
    )

    print(
        f"\n[{file_number}/20] Processing House {house_id}: "
        f"{path.name}"
    )

    partial_sums = []
    partial_counts = []

    raw_rows = 0
    invalid_timestamps = 0
    negative_values = 0

    reader = pd.read_csv(
        path,
        usecols=["Time"] + POWER_COLUMNS,
        chunksize=500_000,
        dtype={column: "float32" for column in POWER_COLUMNS},
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        raw_rows += len(chunk)

        chunk["timestamp"] = pd.to_datetime(
            chunk["Time"],
            errors="coerce",
        )

        invalid_timestamps += int(
            chunk["timestamp"].isna().sum()
        )

        chunk = chunk.dropna(subset=["timestamp"])

        negative_mask = chunk[POWER_COLUMNS] < 0
        negative_values += int(negative_mask.sum().sum())

        chunk[POWER_COLUMNS] = chunk[
            POWER_COLUMNS
        ].mask(negative_mask)

        chunk["interval"] = (
            chunk["timestamp"].dt.floor("15min")
        )

        grouped = chunk.groupby("interval")[POWER_COLUMNS]

        partial_sums.append(grouped.sum(min_count=1))
        partial_counts.append(grouped.count())

        print(
            f"\r  Chunks completed: {chunk_number} | "
            f"Rows read: {raw_rows:,}",
            end="",
        )

    print()

    total_sums = (
        pd.concat(partial_sums)
        .groupby(level=0)
        .sum(min_count=1)
    )

    total_counts = (
        pd.concat(partial_counts)
        .groupby(level=0)
        .sum()
    )

    means = total_sums.divide(
        total_counts.replace(0, pd.NA)
    )

    means = means.rename(columns=COLUMN_NAMES)
    means.index.name = "timestamp"
    result = means.reset_index()

    result.insert(1, "house_id", house_id)

    result["aggregate_energy_kwh"] = (
        result["aggregate_power_w"] * 0.25 / 1000
    )

    result["sample_count"] = (
        total_counts["Aggregate"]
        .reindex(means.index)
        .to_numpy()
    )

    result["data_quality_flag"] = "valid"

    result.loc[
        result["sample_count"] < 50,
        "data_quality_flag"
    ] = "low_sample_count"

    result["source"] = "REFIT_cleaned"

    output_path = (
        OUTPUT_DIR /
        f"refit_house_{house_id:02d}_15min.parquet"
    )

    result.to_parquet(output_path, index=False)

    reports.append({
        "house_id": house_id,
        "source_file": path.name,
        "raw_rows": raw_rows,
        "output_rows": len(result),
        "start_timestamp": result["timestamp"].min(),
        "end_timestamp": result["timestamp"].max(),
        "invalid_timestamps": invalid_timestamps,
        "negative_values_removed": negative_values,
        "low_sample_intervals": int(
            (result["data_quality_flag"] != "valid").sum()
        ),
        "output_file": output_path.name,
    })

    print(
        f"  Completed House {house_id}: "
        f"{len(result):,} intervals"
    )

report = pd.DataFrame(reports)
report.to_csv(REPORT_PATH, index=False)

print("\nREFIT 15-MINUTE PREPARATION COMPLETE")
print("-" * 50)
print("Houses processed:", len(report))
print("Raw rows processed:", f"{report['raw_rows'].sum():,}")
print("Output intervals:", f"{report['output_rows'].sum():,}")
print("Invalid timestamps:", report["invalid_timestamps"].sum())
print("Negative values removed:",
      report["negative_values_removed"].sum())
print("Report:", REPORT_PATH)