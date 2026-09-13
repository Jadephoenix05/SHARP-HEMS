from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/occupancy_profiles/tus2024_ap_activities.parquet"
OUT = ROOT / "data/interim/occupancy_profiles"
REPORT = ROOT / "reports/tus2024_ap_location_validation.csv"

# Composite grouping key; do not rely on the mirror's person_id alone.
KEYS = [
    "survey_year", "schedule_id", "fsu_serial_no", "schedule",
    "sector", "nss_region", "district", "stratum", "sub_stratum",
    "sub_round", "fod_sub_region", "sample_household_no",
    "person_serial_no",
]
META = ["age", "day_of_week", "day_type", "mult", "nsc"]

def minute(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not re.fullmatch(r"\d{1,2}:\d{2}", text):
        return None
    h, m = map(int, text.split(":"))
    if h == 24 and m == 0:
        return 1440
    if 0 <= h < 24 and 0 <= m < 60:
        return 60 * h + m
    return None

def main():
    columns = KEYS + META + [
        "person_id", "time_from", "time_to", "activity_location"
    ]
    data = pd.read_parquet(SOURCE, columns=columns)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    profiles = []
    excluded = []
    missing_key_rows = int(data[KEYS].isna().any(axis=1).sum())
    usable = data.loc[~data[KEYS].isna().any(axis=1)].copy()

    for number, (key, group) in enumerate(
        usable.groupby(KEYS, dropna=False, sort=False), start=1
    ):
        text = json.dumps([str(v) for v in key])
        profile_id = "tus_" + hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()[:24]

        age = pd.to_numeric(group["age"], errors="coerce")
        if age.notna().all() and age.lt(6).all():
            excluded.append({
                "profile_id": profile_id,
                "reason": "under_6_no_profile_created",
                "source_rows": len(group),
            })
            continue

        # Bit flags per clock minute:
        # 1 = reported home, 2 = reported non-home, 4 = unknown.
        flags = np.zeros(1440, dtype=np.uint8)
        invalid_intervals = 0

        for row in group.itertuples(index=False):
            start = minute(row.time_from)
            end = minute(row.time_to)

            # Equal endpoints are ambiguous; do not assume a full day.
            if start is None or end is None or start == end:
                invalid_intervals += 1
                continue

            duration = end - start
            if duration < 0:
                duration += 1440
            if not 0 < duration <= 1440:
                invalid_intervals += 1
                continue

            location = str(row.activity_location).strip().lower()
            if location == "home":
                flag = 1
            elif location in {"fixed location", "non-fixed location"}:
                flag = 2
            else:
                flag = 4

            indices = (np.arange(duration) + start) % 1440
            flags[indices] |= flag

        # Conservative: unknown or conflicting reports stay unresolved.
        home = flags == 1
        nonhome = flags == 2
        known = home | nonhome
        conflict = (flags & 3) == 3

        metadata_conflict = any(
            group[c].nunique(dropna=False) > 1 for c in META
        )
        first = group.iloc[0]

        record = {
            "profile_id": profile_id,
            "source_person_id": first["person_id"],
            "sector": first["sector"],
            "age": first["age"],
            "day_of_week": first["day_of_week"],
            "day_type": first["day_type"],
            "source_mult": first["mult"],
            "source_nsc": first["nsc"],
            "source_rows": len(group),
            "invalid_intervals": invalid_intervals,
            "metadata_conflict": metadata_conflict,
            "reported_home_minutes": int(home.sum()),
            "reported_nonhome_minutes": int(nonhome.sum()),
            "unknown_minutes": int((~known).sum()),
            "conflict_minutes": int(conflict.sum()),
            "complete_location_diary": bool(
                known.all()
                and invalid_intervals == 0
                and not metadata_conflict
                and age.notna().all()
                and age.ge(6).all()
            ),
            "profile_type": "reported_location_from_survey",
            "weighting_status": "not_applied_pending_verification",
        }

        # Clock-time slots: 0000, 0015, ..., 2345.
        # A fraction is provided only if all 15 minutes are resolved.
        for start in range(0, 1440, 15):
            label = f"{start // 60:02d}{start % 60:02d}"
            resolved = known[start:start + 15]
            record[f"home_fraction_{label}"] = (
                float(home[start:start + 15].mean())
                if resolved.all() else np.nan
            )

        profiles.append(record)
        if number % 2000 == 0:
            print(f"Processed {number:,} person groups", flush=True)

    result = pd.DataFrame(profiles)
    if result.empty:
        raise RuntimeError("No profiles produced; source review required.")

    assert (
        result["reported_home_minutes"]
        + result["reported_nonhome_minutes"]
        + result["unknown_minutes"]
    ).eq(1440).all()
    assert not result["profile_id"].duplicated().any()

    output = OUT / "tus2024_ap_reported_location_profiles_v1.parquet"
    result.to_parquet(output, index=False)

    pd.DataFrame(
        excluded, columns=["profile_id", "reason", "source_rows"]
    ).to_csv(OUT / "tus2024_ap_excluded_profile_log.csv", index=False)

    metrics = {
        "status": "BUILT_PENDING_REVIEW",
        "source_activity_rows": len(data),
        "missing_key_rows_excluded": missing_key_rows,
        "under_6_groups_excluded": len(excluded),
        "profiles_created": len(result),
        "complete_location_diaries": int(
            result["complete_location_diary"].sum()
        ),
        "profiles_with_unknown_minutes": int(
            result["unknown_minutes"].gt(0).sum()
        ),
        "profiles_with_location_conflicts": int(
            result["conflict_minutes"].gt(0).sum()
        ),
        "invalid_intervals": int(result["invalid_intervals"].sum()),
        "metadata_conflict_profiles": int(
            result["metadata_conflict"].sum()
        ),
        "survey_weighting": "PENDING",
        "release_ready": False,
    }
    pd.DataFrame(
        metrics.items(), columns=["metric", "value"]
    ).to_csv(REPORT, index=False)

    print("\nAP REPORTED-LOCATION PROFILES BUILT")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print("Output:", output)
    print("Report:", REPORT)

if __name__ == "__main__":
    main()