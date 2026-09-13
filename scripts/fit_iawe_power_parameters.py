from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/appliance_inputs_v1"

data = pd.read_parquet(
    BASE / "iawe_screened_calibration_intervals_v1.parquet"
)

rows = []

for channel, group in data.groupby("channel_id"):
    power = pd.to_numeric(
        group["power_sample_mean_w"], errors="coerce"
    )
    power = power[np.isfinite(power) & power.ge(0)]

    if power.empty:
        raise ValueError(f"No valid power for channel {channel}")

    # Engineering threshold used only to separate low-power observations.
    # It does not establish physical ON/OFF or identify complete cycles.
    threshold_w = 5.0
    above = power[power.gt(threshold_w)]
    segments = group.groupby("segment_id").size()

    rows.append({
        "source_dataset": "iAWE",
        "source_household_id": "iawe_01",
        "channel_id": int(channel),
        "appliance_type_candidate":
            group["appliance_type_candidate"].iloc[0],
        "calibration_intervals": len(power),
        "observed_mean_w": float(power.mean()),
        "observed_p10_w": float(power.quantile(0.10)),
        "observed_p50_w": float(power.quantile(0.50)),
        "observed_p90_w": float(power.quantile(0.90)),
        "observed_p95_w": float(power.quantile(0.95)),
        "observed_max_w": float(power.max()),
        "analysis_threshold_w": threshold_w,
        "above_threshold_intervals": len(above),
        "above_threshold_median_w":
            float(above.median()) if len(above) else None,
        "above_threshold_p95_w":
            float(above.quantile(0.95)) if len(above) else None,
        "observed_below_or_equal_threshold_fraction":
            float(power.le(threshold_w).mean()),
        "segments": len(segments),
        "longest_segment_intervals": int(segments.max()),
        "rated_power_established": False,
        "daily_duty_cycle_established": False,
        "usage_scope": "shared_calibration_evidence",
    })

result = pd.DataFrame(rows)
assert len(result) == 6
assert result["channel_id"].is_unique

output = BASE / "iawe_empirical_power_parameters_v1.csv"
result.to_csv(output, index=False)

notes = {
    "status": "EMPIRICAL_POWER_PARAMETERS_BUILT",
    "source_households": 1,
    "interpretation": [
        "Statistics describe retained 15-minute interval means.",
        "Separate AC channels remain separate calibration examples.",
        "Observed threshold fractions are not daily duty cycles.",
        "Five watts is an explicit analysis assumption.",
        "These parameters do not establish appliance rated power.",
        "Calibration selection may favour particular operating states.",
        "Independent Indian-household evaluation is not established."
    ],
    "master_release_ready": False
}

(BASE / "iawe_empirical_power_parameters_v1.json").write_text(
    json.dumps(notes, indent=2), encoding="utf-8"
)

print("INDIAN POWER CALIBRATION PARAMETERS BUILT")
print(result[[
    "channel_id",
    "appliance_type_candidate",
    "calibration_intervals",
    "above_threshold_intervals",
    "above_threshold_median_w",
]].to_string(index=False))
print("\nSaved:", output)