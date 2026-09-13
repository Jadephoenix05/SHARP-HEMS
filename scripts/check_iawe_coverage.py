from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/iawe/calibration"
REPORT = ROOT / "reports/iawe_sampling_coverage_v1.csv"

rows = []

for channel in range(1, 13):
    path = SOURCE / f"iawe_channel_{channel:02d}_power_voltage_v1.parquet"
    data = pd.read_parquet(
        path, columns=["timestamp_utc", "active_power_w"]
    )
    times = data["timestamp_utc"]
    gaps = times.diff().dt.total_seconds().dropna()
    positive = gaps[gaps > 0]

    power = data.set_index("timestamp_utc")["active_power_w"]
    counts = power.resample("15min").count()

    rows.append({
        "channel_id": channel,
        "timestamps": len(data),
        "median_gap_seconds": positive.median(),
        "most_common_gap_seconds": (
            positive.mode().iloc[0] if not positive.empty else None
        ),
        "largest_gap_seconds": positive.max(),
        "gaps_over_60_seconds": int(positive.gt(60).sum()),
        "gaps_over_900_seconds": int(positive.gt(900).sum()),
        "bins_15min_in_observed_span": len(counts),
        "bins_without_valid_power": int(counts.eq(0).sum()),
        "median_valid_samples_per_nonempty_bin": (
            counts[counts > 0].median()
        ),
        "max_valid_samples_per_bin": counts.max(),
    })

    print(f"Checked channel {channel:02d}", flush=True)

pd.DataFrame(rows).to_csv(REPORT, index=False)
print("\n" + pd.DataFrame(rows).to_string(index=False))
print("\nSaved:", REPORT)