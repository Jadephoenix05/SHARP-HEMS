from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/reside_ac/extracted/dataset"
OUT = ROOT / "data/interim/reside_ac_clean"
REPORT = ROOT / "reports/reside_ac_preparation_v1.csv"
DUPLICATE_DIR = ROOT / "reports/reside_ac_duplicates"


def read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def prepare_sensor(path, mapping):
    raw = read_csv(path)

    required = ["datetime"] + list(mapping)
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    data = raw.rename(columns=mapping).copy()
    data["source_row"] = np.arange(1, len(data) + 1)
    data["datetime_source"] = raw["datetime"].astype("string")
    data["timestamp_source_clock"] = pd.to_datetime(
        raw["datetime"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    invalid_times = int(data["timestamp_source_clock"].isna().sum())
    parse_failures = 0
    infinite_values = 0
    fields = list(mapping.values())

    for field in fields:
        original = data[field].copy()
        numeric = pd.to_numeric(original, errors="coerce")

        parse_failures += int(
            (original.notna() & numeric.isna()).sum()
        )
        infinite_values += int(
            np.isinf(numeric.to_numpy(dtype=float)).sum()
        )
        data[field] = numeric

    if invalid_times or parse_failures or infinite_values:
        raise ValueError(
            f"{path.name}: invalid_times={invalid_times}, "
            f"numeric_parse_failures={parse_failures}, "
            f"infinite_values={infinite_values}"
        )

    # Validate source status codes before duplicate consolidation.
    if "ac_status_source" in data.columns:
        unexpected = (
            data["ac_status_source"].notna()
            & ~data["ac_status_source"].isin([0, 1])
        )
        if unexpected.any():
            raise ValueError(
                f"{path.name}: unexpected AC status codes "
                f"{data.loc[unexpected, 'ac_status_source'].unique().tolist()}"
            )

    time_column = "timestamp_source_clock"
    duplicate_mask = data[time_column].duplicated(keep=False)

    DUPLICATE_DIR.mkdir(parents=True, exist_ok=True)

    # Retain all readings involved in duplicate timestamps.
    data.loc[duplicate_mask].to_csv(
        DUPLICATE_DIR / f"{path.stem}_duplicate_readings.csv",
        index=False,
    )

    grouped = data.groupby(time_column, sort=True)
    result = grouped[fields].first()
    result["source_rows_at_timestamp"] = grouped.size()

    for field in fields:
        distinct = grouped[field].nunique(dropna=True)
        conflict = distinct.gt(1)

        # Keep a value only when available nonmissing readings agree.
        # Conflicting values become missing, never averaged.
        result[field] = result[field].mask(conflict)
        result[f"{field}_conflict"] = conflict

    result["datetime_source"] = grouped["datetime_source"].first()
    result = result.reset_index()

    assert result[time_column].is_unique
    assert result[time_column].is_monotonic_increasing
    assert int(result["source_rows_at_timestamp"].sum()) == len(raw)

    conflict_columns = [f"{field}_conflict" for field in fields]
    conflict_timestamps = int(
        result[conflict_columns].any(axis=1).sum()
    )

    stats = {
        "source_rows": len(raw),
        "unique_timestamps": len(result),
        "duplicate_rows": int(duplicate_mask.sum()),
        "rows_consolidated": len(raw) - len(result),
        "conflicting_timestamps": conflict_timestamps,
    }

    if duplicate_mask.any():
        print(
            f"{path.name}: {stats['duplicate_rows']} duplicate rows; "
            f"{conflict_timestamps} conflicting timestamps",
            flush=True,
        )

    return result, stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    ac = read_csv(
        SOURCE / "Household information/ACDetails.csv"
    )
    building = read_csv(
        SOURCE / "Household information/buildingdetails.csv"
    )
    dwelling = read_csv(
        SOURCE / "Household information/DwellingDetails.csv"
    )

    expected_ids = set(range(1, 12))

    for name, table in [
        ("ac_details", ac),
        ("building_details", building),
        ("dwelling_details", dwelling),
    ]:
        if (
            table["HouseID"].isna().any()
            or table["HouseID"].duplicated().any()
        ):
            raise ValueError(f"{name}: invalid or duplicate HouseID")

        if set(table["HouseID"]) != expected_ids:
            raise ValueError(
                f"{name}: expected HouseIDs 1 through 11"
            )

    metadata = ac.merge(
        building,
        on="HouseID",
        validate="one_to_one",
    ).merge(
        dwelling,
        on="HouseID",
        validate="one_to_one",
    )

    metadata.to_parquet(
        OUT / "household_metadata_v1.parquet",
        index=False,
    )

    weather = read_csv(SOURCE / "WeatherData.csv")
    weather["date_source"] = weather["Date"].astype("string")
    weather["date"] = pd.to_datetime(
        weather["Date"],
        format="%d-%m-%Y",
        errors="raise",
    )

    if weather["date"].isna().any():
        raise ValueError("Missing weather dates")

    if weather["date"].duplicated().any():
        raise ValueError("Duplicate weather dates")

    weather.to_parquet(
        OUT / "weather_daily_source_v1.parquet",
        index=False,
    )

    records = []

    for house in range(1, 12):
        garud, garud_stats = prepare_sensor(
            SOURCE / "Garud" / f"G{house:02d}.csv",
            {
                "R": "phase_r_source_value",
                "Y": "phase_y_source_value",
                "B": "phase_b_source_value",
                "AC Status": "ac_status_source",
            },
        )

        env, env_stats = prepare_sensor(
            SOURCE / "Envilog" / f"E{house:02d}.csv",
            {
                "Temperature": "temperature_source_value",
                "RelativeHumidity": "relative_humidity_source_value",
            },
        )

        for sensor, table in [
            ("garud", garud),
            ("envilog", env),
        ]:
            table["candidate_house_id"] = house
            table["source_dataset"] = "RESIDE_AC"
            table["timezone_status"] = "UNVERIFIED_SOURCE_CLOCK"
            table["mapping_status"] = (
                "FILENAME_SUFFIX_PENDING_VERIFICATION"
            )
            table["processing_status"] = (
                "DUPLICATES_CONSOLIDATED_CONFLICTS_MASKED"
            )

            # Retains existing output filenames for compatibility.
            # Original, unmodified readings remain in data/raw.
            table.to_parquet(
                OUT / f"house_{house:02d}_{sensor}_native_v1.parquet",
                index=False,
            )

        garud_bins = (
            garud.set_index("timestamp_source_clock")
            .resample("15min", closed="left", label="left")
            .agg(
                garud_rows=("ac_status_source", "size"),
                garud_source_rows=("source_rows_at_timestamp", "sum"),
                ac_status_valid_samples=("ac_status_source", "count"),
                ac_status_sample_mean=("ac_status_source", "mean"),
                ac_status_conflict_timestamps=(
                    "ac_status_source_conflict", "sum"
                ),
                phase_r_sample_mean_source_units=(
                    "phase_r_source_value", "mean"
                ),
                phase_y_sample_mean_source_units=(
                    "phase_y_source_value", "mean"
                ),
                phase_b_sample_mean_source_units=(
                    "phase_b_source_value", "mean"
                ),
                phase_r_valid_samples=("phase_r_source_value", "count"),
                phase_y_valid_samples=("phase_y_source_value", "count"),
                phase_b_valid_samples=("phase_b_source_value", "count"),
                phase_r_conflict_timestamps=(
                    "phase_r_source_value_conflict", "sum"
                ),
                phase_y_conflict_timestamps=(
                    "phase_y_source_value_conflict", "sum"
                ),
                phase_b_conflict_timestamps=(
                    "phase_b_source_value_conflict", "sum"
                ),
            )
        )

        env_bins = (
            env.set_index("timestamp_source_clock")
            .resample("15min", closed="left", label="left")
            .agg(
                envilog_rows=("temperature_source_value", "size"),
                envilog_source_rows=("source_rows_at_timestamp", "sum"),
                temperature_valid_samples=(
                    "temperature_source_value", "count"
                ),
                humidity_valid_samples=(
                    "relative_humidity_source_value", "count"
                ),
                temperature_sample_mean_source_units=(
                    "temperature_source_value", "mean"
                ),
                humidity_sample_mean_source_units=(
                    "relative_humidity_source_value", "mean"
                ),
                temperature_conflict_timestamps=(
                    "temperature_source_value_conflict", "sum"
                ),
                humidity_conflict_timestamps=(
                    "relative_humidity_source_value_conflict", "sum"
                ),
            )
        )

        joined = garud_bins.join(env_bins, how="outer")

        assert int(joined["garud_rows"].sum()) == len(garud)
        assert int(joined["envilog_rows"].sum()) == len(env)
        assert (
            int(joined["garud_source_rows"].sum())
            == garud_stats["source_rows"]
        )
        assert (
            int(joined["envilog_source_rows"].sum())
            == env_stats["source_rows"]
        )

        joined["candidate_house_id"] = house
        joined["mapping_status"] = (
            "FILENAME_SUFFIX_PENDING_VERIFICATION"
        )
        joined["timezone_status"] = "UNVERIFIED_SOURCE_CLOCK"
        joined["measurement_units_status"] = (
            "PENDING_DOCUMENTATION_CHECK"
        )
        joined["energy_estimated"] = False
        joined.index.name = "interval_start_source_clock"

        joined.reset_index().to_parquet(
            OUT / f"house_{house:02d}_15min_candidate_v1.parquet",
            index=False,
        )

        phase = str(
            metadata.loc[
                metadata["HouseID"].eq(house),
                "phase on which primary AC is connected",
            ].iloc[0]
        ).strip()

        record = {
            "candidate_house_id": house,
            "garud_rows": len(garud),
            "envilog_rows": len(env),
            "intervals_15min": len(joined),
            "garud_start": garud["timestamp_source_clock"].min(),
            "garud_end": garud["timestamp_source_clock"].max(),
            "missing_ac_status": int(
                garud["ac_status_source"].isna().sum()
            ),
            "missing_temperature": int(
                env["temperature_source_value"].isna().sum()
            ),
            "missing_humidity": int(
                env["relative_humidity_source_value"].isna().sum()
            ),
            "bins_with_both_streams": int(
                (
                    joined["garud_rows"].fillna(0).gt(0)
                    & joined["envilog_rows"].fillna(0).gt(0)
                ).sum()
            ),
            "primary_ac_phase_from_metadata": phase,
            "status": "PREPARED_PENDING_SEMANTIC_QA",
        }

        for prefix, stats in [
            ("garud", garud_stats),
            ("envilog", env_stats),
        ]:
            for name, value in stats.items():
                record[f"{prefix}_{name}"] = value

        records.append(record)
        pd.DataFrame(records).to_csv(REPORT, index=False)

        print(
            f"House {house:02d}: "
            f"{len(garud):,} Garud timestamps, "
            f"{len(env):,} Envilog timestamps, "
            f"{len(joined):,} intervals",
            flush=True,
        )

    print("\nRESIDE-AC PREPARATION COMPLETE")
    print("Houses processed:", len(records))
    print("Household metadata rows:", len(metadata))
    print("Daily weather rows:", len(weather))
    print("Duplicate audits:", DUPLICATE_DIR)
    print("Report:", REPORT)


if __name__ == "__main__":
    main()