from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/iawe"
OUT = SOURCE / "calibration"
REPORT = ROOT / "reports/iawe_calibration_validation_v1.csv"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []

    for channel in range(1, 13):
        path = SOURCE / f"iawe_channel_{channel:02d}_native_v1.parquet"
        data = pd.read_parquet(path, columns=[
            "timestamp_utc", "active_power_w", "voltage_v"
        ])

        if data["timestamp_utc"].isna().any():
            raise RuntimeError(f"Channel {channel}: missing timestamps")

        duplicate_rows = int(
            data.duplicated("timestamp_utc", keep=False).sum()
        )

        # Preserve evidence of source anomalies before deriving clean values.
        data["negative_power"] = data["active_power_w"].lt(0)
        data["nonfinite_power"] = np.isinf(data["active_power_w"])
        data["nonfinite_voltage"] = np.isinf(data["voltage_v"])

        grouped = data.groupby("timestamp_utc", sort=True)
        result = grouped.agg(
            source_rows=("active_power_w", "size"),
            power_nonmissing_count=("active_power_w", "count"),
            power_distinct_values=("active_power_w", "nunique"),
            power_source_min_w=("active_power_w", "min"),
            power_source_max_w=("active_power_w", "max"),
            voltage_nonmissing_count=("voltage_v", "count"),
            voltage_distinct_values=("voltage_v", "nunique"),
            voltage_source_min_v=("voltage_v", "min"),
            voltage_source_max_v=("voltage_v", "max"),
            negative_power_seen=("negative_power", "max"),
            nonfinite_power_seen=("nonfinite_power", "max"),
            nonfinite_voltage_seen=("nonfinite_voltage", "max"),
        )

        result["power_conflict"] = result["power_distinct_values"].gt(1)
        result["voltage_conflict"] = result["voltage_distinct_values"].gt(1)

        # At a duplicate timestamp, use a value only when all available
        # nonmissing readings agree. Never average conflicting readings.
        result["active_power_w"] = result["power_source_min_w"].where(
            result["power_distinct_values"].eq(1)
            & ~result["negative_power_seen"]
            & ~result["nonfinite_power_seen"]
        )
        result["voltage_v"] = result["voltage_source_min_v"].where(
            result["voltage_distinct_values"].eq(1)
            & ~result["nonfinite_voltage_seen"]
        )

        result = result.reset_index()
        result["channel_id"] = channel

        assert result["timestamp_utc"].is_unique
        assert result["timestamp_utc"].is_monotonic_increasing
        assert int(result["source_rows"].sum()) == len(data)

        output = OUT / f"iawe_channel_{channel:02d}_power_voltage_v1.parquet"
        result.to_parquet(output, index=False)

        reports.append({
            "channel_id": channel,
            "input_rows": len(data),
            "unique_timestamps": len(result),
            "duplicate_timestamp_rows": duplicate_rows,
            "rows_consolidated": len(data) - len(result),
            "power_conflict_timestamps": int(result["power_conflict"].sum()),
            "voltage_conflict_timestamps": int(result["voltage_conflict"].sum()),
            "negative_power_timestamps": int(result["negative_power_seen"].sum()),
            "missing_derived_power": int(result["active_power_w"].isna().sum()),
            "missing_derived_voltage": int(result["voltage_v"].isna().sum()),
            "status": "CLEANED_POWER_VOLTAGE_PENDING_COVERAGE_QA",
        })

        pd.DataFrame(reports).to_csv(REPORT, index=False)
        print(
            f"Channel {channel:02d}: {len(result):,} timestamps; "
            f"power conflicts={reports[-1]['power_conflict_timestamps']}; "
            f"voltage conflicts={reports[-1]['voltage_conflict_timestamps']}",
            flush=True,
        )

    print("\nSaved report:", REPORT)

if __name__ == "__main__":
    main()