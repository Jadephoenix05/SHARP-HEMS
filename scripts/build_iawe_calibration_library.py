from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/iawe/15min"
OUTPUT = ROOT / "data/processed/appliance_inputs_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

CHANNELS = {
    3: "refrigerator",
    4: "air_conditioner",
    5: "air_conditioner",
    7: "laptop",
    10: "television",
    11: "water_filter",
}

parts = []
summary = []

for channel, appliance in CHANNELS.items():
    path = SOURCE / f"iawe_channel_{channel:02d}_15min_v1.parquet"
    data = pd.read_parquet(path)

    power = pd.to_numeric(data["power_sample_mean_w"], errors="coerce")
    keep = (
        data["passes_sample_count_screen"].fillna(False)
        & np.isfinite(power)
        & power.ge(0)
        & data["power_conflict_timestamps"].eq(0)
    )

    selected = data.loc[keep].copy()
    selected["interval_start_utc"] = pd.to_datetime(
        selected["interval_start_utc"], utc=True, errors="raise"
    )
    selected = selected.sort_values("interval_start_utc")

    if selected.empty:
        raise ValueError(f"No usable records for channel {channel}")

    if selected["interval_start_utc"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in channel {channel}")

    # Gaps break segments: do not stitch separated observations together.
    gap = selected["interval_start_utc"].diff()
    segment = gap.ne(pd.Timedelta(minutes=15)).cumsum()

    selected["segment_id"] = [
        f"iawe_c{channel:02d}_s{int(value):05d}"
        for value in segment
    ]
    selected["appliance_type_candidate"] = appliance
    selected["source_household_id"] = "iawe_01"
    selected["usage_scope"] = "calibration_only"
    selected["is_synthetic"] = False
    selected["continuous_coverage_verified"] = False

    parts.append(selected)
    lengths = selected.groupby("segment_id").size()

    summary.append({
        "channel_id": channel,
        "appliance": appliance,
        "selected_intervals": len(selected),
        "segments": len(lengths),
        "longest_segment_intervals": int(lengths.max()),
        "segments_at_least_4_intervals": int(lengths.ge(4).sum()),
    })

library = pd.concat(parts, ignore_index=True)
assert not library.duplicated(
    ["channel_id", "interval_start_utc"]
).any()

library.to_parquet(
    OUTPUT / "iawe_screened_calibration_intervals_v1.parquet",
    index=False,
)

report = {
    "status": "CALIBRATION_INTERVALS_BUILT",
    "rows": len(library),
    "channels": summary,
    "limitations": [
        "One measured household; not population-representative.",
        "Sample-count screening does not establish continuous coverage.",
        "Selected segments may contain OFF or standby observations.",
        "Water-filter readings do not represent every purifier type.",
        "No gap filling, energy estimation or rated-power inference.",
        "No assignments to AP households have been made.",
    ],
    "master_release_ready": False,
}

report_path = ROOT / "reports/iawe_calibration_library_v1.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(pd.DataFrame(summary).to_string(index=False))
print(f"\nSaved calibration intervals: {len(library):,}")
print("Output:", OUTPUT / "iawe_screened_calibration_intervals_v1.parquet")