from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/iawe/extracted/iawe/iawe/electricity/electricity"
OUT = ROOT / "data/interim/iawe"
REPORT = ROOT / "reports/iawe_channels_validation_v1.csv"

MEASUREMENTS = {
    "W": "active_power_w",
    "VAR": "reactive_power_var",
    "VA": "apparent_power_va",
    "f": "frequency_hz",
    "PF": "power_factor",
    "A": "current_a",
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    labels = {}
    for line in (SOURCE / "labels.dat").read_text().splitlines():
        channel, name = line.strip().split(maxsplit=1)
        labels[int(channel)] = name

    reports = []

    for channel, appliance in sorted(labels.items()):
        path = SOURCE / f"{channel}.csv"
        output = OUT / f"iawe_channel_{channel:02d}_native_v1.parquet"
        temporary = output.with_suffix(".parquet.partial")

        writer = None
        rows = 0
        bad_timestamps = 0
        parse_failures = 0
        nonfinite_values = 0
        negative_power = 0
        backward_steps = 0
        adjacent_duplicates = 0
        previous = None
        earliest = None
        latest = None
        voltage_present = False

        try:
            for chunk in pd.read_csv(
                path, chunksize=100000, low_memory=False
            ):
                result = pd.DataFrame(index=chunk.index)
                result["source_row"] = np.arange(
                    rows + 1, rows + len(chunk) + 1, dtype=np.int64
                )
                result["channel_id"] = np.int16(channel)
                result["appliance_name"] = appliance
                result["source_dataset"] = "iAWE"
                result["timestamp_source"] = chunk["timestamp"].astype("string")

                numeric_time = pd.to_numeric(
                    chunk["timestamp"], errors="coerce"
                )
                stamp = pd.to_datetime(
                    numeric_time, unit="s", utc=True, errors="coerce"
                )
                result["timestamp_utc"] = stamp
                bad_timestamps += int(stamp.isna().sum())

                valid = stamp.dropna()
                if not valid.empty:
                    delta = valid.diff()
                    backward_steps += int(
                        delta.lt(pd.Timedelta(0)).sum()
                    )
                    adjacent_duplicates += int(
                        delta.eq(pd.Timedelta(0)).sum()
                    )
                    if previous is not None:
                        backward_steps += int(valid.iloc[0] < previous)
                        adjacent_duplicates += int(valid.iloc[0] == previous)
                    previous = valid.iloc[-1]
                    earliest = (
                        valid.min() if earliest is None
                        else min(earliest, valid.min())
                    )
                    latest = (
                        valid.max() if latest is None
                        else max(latest, valid.max())
                    )

                voltage = (
                    "VLN" if "VLN" in chunk.columns
                    else "V" if "V" in chunk.columns else None
                )
                voltage_present |= voltage is not None

                mapping = dict(MEASUREMENTS)
                if voltage:
                    mapping[voltage] = "voltage_v"

                for source, target in mapping.items():
                    if source in chunk.columns:
                        numeric = pd.to_numeric(
                            chunk[source], errors="coerce"
                        )
                        parse_failures += int(
                            (chunk[source].notna() & numeric.isna()).sum()
                        )
                        nonfinite_values += int(
                            np.isinf(numeric.to_numpy(dtype=float)).sum()
                        )
                        # Preserve signs, zero readings and source magnitudes.
                        result[target] = numeric.astype("float64")
                    else:
                        result[target] = np.nan

                if "voltage_v" not in result:
                    result["voltage_v"] = np.nan

                negative_power += int(result["active_power_w"].lt(0).sum())

                table = pa.Table.from_pandas(result, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="snappy"
                    )
                writer.write_table(table)
                rows += len(chunk)

            if writer is not None:
                writer.close()
                writer = None

            if rows == 0:
                raise RuntimeError(f"Empty channel: {channel}")

            temporary.replace(output)

        finally:
            if writer is not None:
                writer.close()

        reports.append({
            "channel_id": channel,
            "appliance_name": appliance,
            "rows": rows,
            "start_utc": earliest,
            "end_utc": latest,
            "invalid_timestamps": bad_timestamps,
            "measurement_parse_failures": parse_failures,
            "infinite_measurement_values": nonfinite_values,
            "negative_power_rows_preserved": negative_power,
            "backward_timestamp_steps": backward_steps,
            "adjacent_duplicate_timestamps": adjacent_duplicates,
            "voltage_column_available": voltage_present,
            "status": "CONVERTED_PENDING_QA",
            "full_duplicate_check": "PENDING",
        })

        # Save progress after every completed channel.
        pd.DataFrame(reports).to_csv(REPORT, index=False)
        print(
            f"Channel {channel:02d}: {appliance} — {rows:,} rows",
            flush=True
        )

    print("\nIAWE NATIVE CHANNEL CONVERSION COMPLETE")
    print("Channels:", len(reports))
    print("Total rows:", sum(r["rows"] for r in reports))
    print("Report:", REPORT)
    print("Calibration and release approval remain pending.")

if __name__ == "__main__":
    main()