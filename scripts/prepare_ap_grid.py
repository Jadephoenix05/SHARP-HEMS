from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/raw/grid_india/"
    "India_Elec_data_(Jan2020-Mar2025).csv"
)

OUTPUT = Path(
    "data/interim/grid_peak_labels/"
    "ap_grid_daily_ml_v1.parquet"
)

REPORT = Path(
    "reports/ap_grid_daily_validation_v1.csv"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

# Load raw dataset
df = pd.read_csv(INPUT)

# Standardize column names
df = df.rename(columns={
    "Date": "date",
    "State": "state",
    "Max Demand Met": "max_demand_met_mw",
    "Shortage During Peak": "peak_shortage_mw",
    "Energy Met": "energy_met_mu",
    "Drawl Schedule": "drawl_schedule",
    "OD(+) / UD(-)": "overdraw_underdraw",
    "Max OD": "max_overdraw_mw",
    "Energy Shortage": "energy_shortage_mu",
})

# Select Andhra Pradesh only
df["state"] = df["state"].astype(str).str.strip()

df = df[
    df["state"].str.casefold() == "andhra pradesh"
].copy()

# Parse and sort dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date")

# Remove duplicate dates
duplicate_count = int(df.duplicated(subset=["date"]).sum())
df = df.drop_duplicates(subset=["date"], keep="last")

numeric_columns = [
    "max_demand_met_mw",
    "peak_shortage_mw",
    "energy_met_mu",
    "drawl_schedule",
    "overdraw_underdraw",
    "max_overdraw_mw",
    "energy_shortage_mu",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Calendar features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear
df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

# Shortage ratios
df["peak_shortage_pct"] = (
    100 * df["peak_shortage_mw"]
    / df["max_demand_met_mw"].replace(0, pd.NA)
)

df["energy_shortage_pct"] = (
    100 * df["energy_shortage_mu"]
    / df["energy_met_mu"].replace(0, pd.NA)
)

# Historical demand features
for lag in [1, 7, 14, 28]:
    df[f"demand_lag_{lag}d"] = df["max_demand_met_mw"].shift(lag)

df["demand_rolling_mean_7d"] = (
    df["max_demand_met_mw"]
    .shift(1)
    .rolling(7, min_periods=3)
    .mean()
)

df["demand_rolling_mean_30d"] = (
    df["max_demand_met_mw"]
    .shift(1)
    .rolling(30, min_periods=7)
    .mean()
)

df["demand_rolling_std_7d"] = (
    df["max_demand_met_mw"]
    .shift(1)
    .rolling(7, min_periods=3)
    .std()
)

# Prediction targets
df["target_next_day_demand_mw"] = (
    df["max_demand_met_mw"].shift(-1)
)

df["target_next_day_peak_shortage_mw"] = (
    df["peak_shortage_mw"].shift(-1)
)

# Chronological split
df["split"] = "train"
df.loc[df["date"] >= "2024-01-01", "split"] = "validation"
df.loc[df["date"] >= "2025-01-01", "split"] = "test"

# Data-quality flag
df["data_quality_flag"] = "valid"

df.loc[
    df[numeric_columns].isna().any(axis=1),
    "data_quality_flag"
] = "missing_numeric_value"

df.loc[
    df["max_demand_met_mw"] <= 0,
    "data_quality_flag"
] = "invalid_demand"
    
# Save processed table
df.to_parquet(OUTPUT, index=False)

# Validation report
report = pd.DataFrame({
    "metric": [
        "rows",
        "duplicate_dates_removed",
        "start_date",
        "end_date",
        "missing_demand",
        "invalid_demand",
        "train_rows",
        "validation_rows",
        "test_rows",
    ],
    "value": [
        len(df),
        duplicate_count,
        str(df["date"].min().date()),
        str(df["date"].max().date()),
        int(df["max_demand_met_mw"].isna().sum()),
        int((df["max_demand_met_mw"] <= 0).sum()),
        int((df["split"] == "train").sum()),
        int((df["split"] == "validation").sum()),
        int((df["split"] == "test").sum()),
    ],
})

report.to_csv(REPORT, index=False)

print("\nANDHRA PRADESH GRID DATA")
print("-" * 45)
print("Rows:", len(df))
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Duplicate dates removed:", duplicate_count)
print("Missing demand:", df["max_demand_met_mw"].isna().sum())
print("\nSplit distribution:")
print(df["split"].value_counts())
print("\nOutput:", OUTPUT)
print("Report:", REPORT)