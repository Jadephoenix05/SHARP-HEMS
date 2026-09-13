from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/iawe/calibration"
OUT = ROOT / "data/interim/iawe/15min"
REPORT = ROOT / "reports/iawe_15min_validation_v1.csv"

MIN_COMPLETENESS = 0.80

LABELS = {
    1: "mains", 2: "mains", 3: "fridge",
    4: "air conditioner", 5: "air conditioner",
    6: "washing machine", 7: "laptop computer",
    8: "iron", 9: "kitchen outlets",
    10: "television", 11: "water filter", 12: "water motor",
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    reports = []

    for channel, appliance in LABELS.items():
        path = SOURCE / f"iawe_channel_{channel:02d}_power_voltage_v1.parquet"
        data = pd.read_parquet(path, columns=[
            "timestamp_utc", "active_power_w", "voltage_v",
            "power_conflict", "voltage_conflict",
        ]).set_index("timestamp_utc").sort_index()

        assert data.index.is_unique
        assert not data.index.hasnans

        # Means below are sample means, not time-weighted means.
        bins = data.resample("15min", closed="left", label="left")
        result = bins.agg(
            power_sample_mean_w=("active_power_w", "mean"),
            power_sample_median_w=("active_power_w", "median"),
            power_sample_min_w=("active_power_w", "min"),
            power_sample_max_w=("active_power_w", "max"),
            power_valid_samples=("active_power_w", "count"),
            timestamp_count=("active_power_w", "size"),
            voltage_sample_mean_v=("voltage_v", "mean"),
            voltage_sample_min_v=("voltage_v", "min"),
            voltage_sample_max_v=("voltage_v", "max"),
            voltage_valid_samples=("voltage_v", "count"),
            power_conflict_timestamps=("power_conflict", "sum"),
            voltage_conflict_timestamps=("voltage_conflict", "sum"),
        )

        assert int(result["timestamp_count"].sum()) == len(data)
        assert int(result["power_valid_samples"].sum()) == int(
            data["active_power_w"].notna().sum()
        )

        if channel != 12:
            # Verify timestamps sit on whole seconds before using count/900.
            assert (
                data.index == data.index.floor("s")
            ).all()
            assert result["timestamp_count"].le(900).all()

            result["nominal_power_sample_completeness"] = (
                result["power_valid_samples"] / 900.0
            )
            result["passes_sample_count_screen"] = (
                result["nominal_power_sample_completeness"]
                .ge(MIN_COMPLETENESS)
                .astype("boolean")
            )
            result["sampling_assumption"] = "nominal_1_second"
        else:
            result["nominal_power_sample_completeness"] = np.nan
            result["passes_sample_count_screen"] = pd.Series(
                pd.NA, index=result.index, dtype="boolean"
            )
            result["sampling_assumption"] = "irregular_not_scored"

        result["channel_id"] = channel
        result["appliance_name"] = appliance
        result["source_dataset"] = "iAWE"
        result["source_geography"] = "New Delhi, India"
        result["interval_seconds"] = 900
        result["sample_count_threshold"] = MIN_COMPLETENESS
        result["summary_status"] = "DESCRIPTIVE_NOT_ENERGY_VALIDATED"

        result.index.name = "interval_start_utc"
        result = result.reset_index()

        output = OUT / f"iawe_channel_{channel:02d}_15min_v1.parquet"
        result.to_parquet(output, index=False)

        reports.append({
            "channel_id": channel,
            "appliance_name": appliance,
            "intervals": len(result),
            "bins_with_power": int(result["power_valid_samples"].gt(0).sum()),
            "bins_without_power": int(result["power_valid_samples"].eq(0).sum()),
            "bins_passing_sample_count_screen": (
                int(result["passes_sample_count_screen"].sum())
                if channel != 12 else None
            ),
            "sampling_completeness_scored": channel != 12,
            "energy_estimated": False,
        })
        pd.DataFrame(reports).to_csv(REPORT, index=False)
        print(f"Channel {channel:02d}: {len(result):,} intervals", flush=True)

    print("\nIAWE 15-MINUTE DESCRIPTIVE SUMMARIES COMPLETE")
    print("Report:", REPORT)

if __name__ == "__main__":
    main()