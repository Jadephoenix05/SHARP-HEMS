from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/refit_canonical_v1"
OUTPUT = ROOT / "data/processed/appliance_inputs_v1/refit_trace_library"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Exact source labels only. Mixed appliance sites remain excluded.
LABELS = {
    "Fridge": "refrigerator",
    "Fridge(garage)": "refrigerator",
    "Washing Machine": "washing_machine",
    "Washing Machine (1)": "washing_machine",
    "Washing Machine (2)": "washing_machine",
    "Kettle": "electric_kettle",
    "Microwave": "microwave",
    "Television": "television",
    "Router": "modem_router",
    "Desktop Computer": "desktop",
}

split_path = (
    ROOT / "data/processed/appliance_inputs_v1/household_splits_v1.csv"
)
splits = pd.read_csv(split_path, dtype=str)
splits = splits.loc[splits["source"].eq("REFIT")]

if splits["household_id"].duplicated().any():
    raise ValueError("Duplicate REFIT household split assignments")

split_map = splits.set_index("household_id")["split"].to_dict()

files = sorted(SOURCE.glob("refit_house_*_channels_v1.parquet"))
if len(files) != 20:
    raise ValueError(f"Expected 20 REFIT files; found {len(files)}")

report = []
manifest = []

for path in files:
    data = pd.read_parquet(path, columns=[
        "source_house_id",
        "channel_id",
        "appliance_name_source",
        "timestamp_source",
        "power_w",
        "source_aggregate_sample_count",
        "channel_coverage_known",
        "source_data_quality_flag",
    ])

    houses = data["source_house_id"].unique()
    if len(houses) != 1:
        raise ValueError(f"Multiple households in {path.name}")

    house = int(houses[0])
    split = split_map[str(house)]
    parts = []

    data = data.loc[data["appliance_name_source"].isin(LABELS)].copy()

    for channel, group in data.groupby("channel_id"):
        group = group.copy()
        group["timestamp_source"] = pd.to_datetime(
            group["timestamp_source"], errors="raise"
        )

        if group["timestamp_source"].isna().any():
            raise ValueError(f"Missing timestamp: house {house}, channel {channel}")
        if group["timestamp_source"].duplicated().any():
            raise ValueError(f"Duplicate timestamp: house {house}, channel {channel}")

        group["power_w"] = pd.to_numeric(group["power_w"], errors="coerce")
        valid = np.isfinite(group["power_w"]) & group["power_w"].ge(0)
        excluded = int((~valid).sum())
        group = group.loc[valid].sort_values("timestamp_source").copy()

        if group.empty:
            continue

        # Missing intervals break segments; never fill them with zeros.
        segment = group["timestamp_source"].diff().ne(
            pd.Timedelta(minutes=15)
        ).cumsum()

        group["segment_id"] = [
            f"refit_h{house:02d}_c{int(channel):02d}_s{int(s):06d}"
            for s in segment
        ]
        group["appliance_type_candidate"] = (
            group["appliance_name_source"].map(LABELS)
        )
        group["split"] = split
        group["source_dataset"] = "REFIT"
        group["source_geography"] = "United Kingdom"
        group["timestamp_timezone_status"] = "UNVERIFIED"
        group["power_statistic"] = "ARITHMETIC_MEAN_OF_AVAILABLE_SAMPLES"
        group["usage_scope"] = "candidate_temporal_trace"
        group["is_synthetic"] = False
        group["physical_off_labels_verified"] = False
        group["complete_appliance_cycles_verified"] = False

        lengths = group.groupby("segment_id").size()
        report.append({
            "house": house,
            "channel": int(channel),
            "appliance": group["appliance_type_candidate"].iloc[0],
            "split": split,
            "intervals": len(group),
            "excluded_invalid_power": excluded,
            "segments": len(lengths),
            "longest_segment_intervals": int(lengths.max()),
        })
        parts.append(group)

    if parts:
        result = pd.concat(parts, ignore_index=True)
        target = OUTPUT / f"refit_house_{house:02d}_candidate_traces.parquet"
        result.to_parquet(target, index=False, compression="zstd")
        manifest.append({
            "house": house,
            "split": split,
            "rows": len(result),
            "path": target.relative_to(ROOT).as_posix(),
        })
        print(f"House {house:02d}: {len(result):,} intervals ({split})", flush=True)

audit = pd.DataFrame(report)
if audit.empty:
    raise ValueError("No matching appliance traces found")

audit.to_csv(ROOT / "reports/refit_trace_library_v1.csv", index=False)
pd.DataFrame(manifest).to_csv(OUTPUT / "manifest_v1.csv", index=False)

print("\nREFIT CANDIDATE TRACE LIBRARY BUILT")
print(audit.groupby(["appliance", "split"])["intervals"].sum().to_string())
print("\nOutput:", OUTPUT)
print("Sampling coverage and complete appliance cycles remain unverified.")
print("No traces assigned to AP households yet.")