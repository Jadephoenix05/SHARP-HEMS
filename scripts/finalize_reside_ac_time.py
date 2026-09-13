from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/reside_ac_clean"
OUT = SOURCE / "timezone_aligned"
REPORT = ROOT / "reports/reside_ac_time_validation_v1.csv"

OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)
records = []

for house in range(1, 12):
    path = SOURCE / f"house_{house:02d}_15min_candidate_v1.parquet"
    data = pd.read_parquet(path)

    clock = pd.to_datetime(
        data["interval_start_source_clock"], errors="raise"
    )
    if clock.isna().any() or clock.duplicated().any():
        raise ValueError(f"House {house}: missing or duplicate times")

    data["interval_start_ist"] = clock.dt.tz_localize("Asia/Kolkata")
    data["interval_start_utc"] = (
        data["interval_start_ist"].dt.tz_convert("UTC")
    )
    data["timezone_status"] = "IST_CONFIRMED_BY_FIGSHARE_DESCRIPTION"
    data["source_timestamp_year"] = clock.dt.year
    data["description_claimed_year"] = 2021
    data["source_year_discrepancy"] = clock.dt.year.ne(2021)
    data["date_resolution_status"] = (
        "CSV_YEAR_PRESERVED_DESCRIPTION_DISCREPANCY_UNRESOLVED"
    )
    data["release_ready"] = False

    output = OUT / f"house_{house:02d}_15min_v1.parquet"
    data.to_parquet(output, index=False)

    records.append({
        "house_id": house,
        "rows": len(data),
        "start_ist": data["interval_start_ist"].min(),
        "end_ist": data["interval_start_ist"].max(),
        "start_utc": data["interval_start_utc"].min(),
        "csv_years": ",".join(
            str(year) for year in sorted(clock.dt.year.unique())
        ),
        "description_year": 2021,
        "year_discrepancy": bool(data["source_year_discrepancy"].any()),
        "units_and_mapping_review": "PENDING",
    })

pd.DataFrame(records).to_csv(REPORT, index=False)
print("Timezone-aligned houses:", len(records))
print("Total intervals:", sum(row["rows"] for row in records))
print("CSV years preserved; description discrepancy recorded.")
print("Report:", REPORT)