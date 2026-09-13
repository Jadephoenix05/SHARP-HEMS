from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/interim/refit_15min"
OUTPUT = ROOT / "data/processed/refit_canonical_v1"
REPORTS = ROOT / "reports"
MAPPING = ROOT / "data/registry/refit_appliance_mapping.csv"

EXPECTED_HOUSES = set(range(1, 22)) - {14}

OUTPUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def main():
    mapping = pd.read_csv(MAPPING)

    for column in ["house_id", "channel"]:
        values = pd.to_numeric(mapping[column], errors="raise")
        if values.isna().any() or not values.mod(1).eq(0).all():
            raise ValueError(f"Invalid mapping values: {column}")
        mapping[column] = values.astype(int)

    if mapping.duplicated(["house_id", "channel"]).any():
        raise ValueError("Duplicate household/channel mapping")

    if set(mapping["house_id"]) != EXPECTED_HOUSES:
        raise ValueError("Unexpected or missing houses in mapping")

    for house_id, group in mapping.groupby("house_id"):
        if set(group["channel"]) != set(range(10)):
            raise ValueError(f"House {house_id}: incomplete mapping")

    if mapping["appliance_name"].isna().any():
        raise ValueError("Missing appliance names")

    files = sorted(INPUT.glob("refit_house_*_15min.parquet"))
    found_houses = {
        int(path.stem.split("_")[2]) for path in files
    }
    if found_houses != EXPECTED_HOUSES or len(files) != 20:
        raise ValueError("Expected exactly 20 REFIT household files")

    reports = []

    for path in files:
        house = int(path.stem.split("_")[2])
        data = pd.read_parquet(path)

        if data.empty:
            raise ValueError(f"House {house}: empty input")
        if not data["house_id"].eq(house).all():
            raise ValueError(f"House {house}: inconsistent house IDs")

        timestamps = pd.to_datetime(data["timestamp"], errors="raise")
        if timestamps.isna().any() or timestamps.duplicated().any():
            raise ValueError(f"House {house}: missing/duplicate timestamps")
        if not timestamps.eq(timestamps.dt.floor("15min")).all():
            raise ValueError(f"House {house}: timestamps off 15-minute grid")

        data = data.assign(timestamp=timestamps).sort_values(
            "timestamp"
        ).reset_index(drop=True)

        destination = OUTPUT / f"refit_house_{house:02d}_channels_v1.parquet"
        temporary = destination.with_suffix(".parquet.partial")
        writer = None

        missing_power = 0
        negative_power = 0
        nonfinite_power = 0

        try:
            for channel in range(10):
                power_column = (
                    "aggregate_power_w"
                    if channel == 0
                    else f"appliance_{channel}_power_w"
                )

                entry = mapping.loc[
                    mapping["house_id"].eq(house)
                    & mapping["channel"].eq(channel)
                ].iloc[0]

                power = pd.to_numeric(data[power_column], errors="raise")
                missing = power.isna()
                negative = power.lt(0)
                nonfinite = power.notna() & ~np.isfinite(power)

                missing_power += int(missing.sum())
                negative_power += int(negative.sum())
                nonfinite_power += int(nonfinite.sum())

                table = pd.DataFrame({
                    "schema_version": "refit_channels_v1",
                    "source_dataset": "REFIT",
                    "source_house_id": house,
                    "household_id": f"refit_house_{house:02d}",
                    "channel_id": channel,
                    "channel_uid": f"refit_house_{house:02d}_channel_{channel:02d}",
                    "channel_type": str(entry["channel_type"]),
                    "appliance_name_source": str(entry["appliance_name"]),
                    "source_column": str(entry["source_column"]),
                    "timestamp_source": data["timestamp"],
                    "timestamp_timezone_status": "UNVERIFIED",
                    "interval_minutes": 15,
                    "power_w": power,
                    "power_statistic": "ARITHMETIC_MEAN_OF_AVAILABLE_SAMPLES",
                    "power_missing": missing,
                    "power_negative": negative,
                    "power_nonfinite": nonfinite,
                    "source_aggregate_sample_count": data["sample_count"],
                    "channel_coverage_known": False,
                    "source_data_quality_flag": data["data_quality_flag"],
                    "source_label": data["source"],
                    "aggregate_energy_kwh_source_estimate": (
                        pd.to_numeric(
                            data["aggregate_energy_kwh"], errors="raise"
                        )
                        if channel == 0
                        else np.full(len(data), np.nan)
                    ),
                    "source_file": path.relative_to(ROOT).as_posix(),
                    "source_geography": "United Kingdom",
                    "is_synthetic": False,
                })

                arrow = pa.Table.from_pandas(
                    table, preserve_index=False
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, arrow.schema, compression="snappy"
                    )
                writer.write_table(arrow)

            writer.close()
            writer = None

            rows = pq.ParquetFile(temporary).metadata.num_rows
            if rows != len(data) * 10:
                raise ValueError(f"House {house}: output row mismatch")

            temporary.replace(destination)

        finally:
            if writer is not None:
                writer.close()
            if temporary.exists():
                temporary.unlink()

        reports.append({
            "house_id": house,
            "input_intervals": len(data),
            "output_channel_rows": rows,
            "missing_power_rows": missing_power,
            "negative_power_rows": negative_power,
            "nonfinite_power_rows": nonfinite_power,
            "start_source_timestamp": str(data["timestamp"].min()),
            "end_source_timestamp": str(data["timestamp"].max()),
            "status": "SCHEMA_UPGRADED_PENDING_SOURCE_METHOD_REVIEW",
        })

        print(
            f"House {house:02d}: {len(data):,} intervals "
            f"-> {rows:,} channel rows",
            flush=True,
        )

    report = pd.DataFrame(reports)
    report.to_csv(
        REPORTS / "refit_schema_upgrade_v1.csv", index=False
    )

    notes = {
        "schema_version": "refit_channels_v1",
        "layout": "One row per household, channel and source interval",
        "source_mapping": MAPPING.relative_to(ROOT).as_posix(),
        "preservation": [
            "Original interim files unchanged",
            "Power values and source quality flags preserved",
            "No gap filling or timezone conversion",
            "No new appliance energy estimates",
            "No Indian household identity or control eligibility assigned",
        ],
        "remaining_checks": [
            "Verify cleaned-release identity and redistribution terms",
            "Review original aggregation script and timestamp basis",
            "Review within-interval coverage and energy estimation",
            "Carry forward channel changes and mixed-appliance notes",
            "Account for House 21 solar-affected aggregate",
        ],
        "release_ready": False,
    }

    (OUTPUT / "schema_notes.json").write_text(
        json.dumps(notes, indent=2), encoding="utf-8"
    )

    print("\nREFIT SCHEMA UPGRADE COMPLETE")
    print("Houses:", len(report))
    print("Input intervals:", int(report["input_intervals"].sum()))
    print("Channel rows:", int(report["output_channel_rows"].sum()))
    print("Output:", OUTPUT)
    print("Report:", REPORTS / "refit_schema_upgrade_v1.csv")


if __name__ == "__main__":
    main()