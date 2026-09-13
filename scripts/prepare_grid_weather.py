from pathlib import Path
import pandas as pd

WEATHER_INPUT = Path(
    "data/raw/weather/"
    "nasa_power_guntur_ap_hourly_2021_2025.csv"
)

GRID_INPUT = Path(
    "data/interim/grid_peak_labels/"
    "ap_grid_daily_ml_v1.parquet"
)

WEATHER_OUTPUT = Path(
    "data/interim/weather_aligned/"
    "guntur_weather_daily_2021_2025.parquet"
)

COMBINED_OUTPUT = Path(
    "data/processed/grid_forecast/"
    "ap_grid_weather_daily_v1.parquet"
)

REPORT_OUTPUT = Path(
    "reports/grid_weather_join_validation_v1.csv"
)

for path in [WEATHER_OUTPUT, COMBINED_OUTPUT, REPORT_OUTPUT]:
    path.parent.mkdir(parents=True, exist_ok=True)

# Dynamically locate the NASA CSV header
with WEATHER_INPUT.open("r", encoding="utf-8") as file:
    lines = file.readlines()

header_end = next(
    i for i, line in enumerate(lines)
    if line.strip() == "-END HEADER-"
)

weather = pd.read_csv(
    WEATHER_INPUT,
    skiprows=header_end + 1
)

weather = weather.rename(columns={
    "YEAR": "year",
    "MO": "month",
    "DY": "day",
    "HR": "hour",
    "T2M": "temperature_c",
    "RH2M": "humidity_pct",
    "ALLSKY_SFC_SW_DWN": "solar_irradiance_wh_m2",
    "WS10M": "wind_speed_m_s",
})

numeric_columns = [
    "year",
    "month",
    "day",
    "hour",
    "temperature_c",
    "humidity_pct",
    "solar_irradiance_wh_m2",
    "wind_speed_m_s",
]

for column in numeric_columns:
    weather[column] = pd.to_numeric(
        weather[column],
        errors="coerce"
    )

# NASA missing-value marker
weather = weather.replace(-999, pd.NA)

weather["timestamp"] = pd.to_datetime(
    weather[["year", "month", "day", "hour"]],
    errors="coerce"
)

weather = weather.dropna(subset=["timestamp"])
weather = weather.sort_values("timestamp")
weather["date"] = weather["timestamp"].dt.normalize()

# Cooling stress above 24°C
weather["cooling_degree"] = (
    weather["temperature_c"] - 24
).clip(lower=0)

# Aggregate hourly weather into daily features
daily_weather = (
    weather.groupby("date", as_index=False)
    .agg(
        temperature_mean_c=("temperature_c", "mean"),
        temperature_min_c=("temperature_c", "min"),
        temperature_max_c=("temperature_c", "max"),
        humidity_mean_pct=("humidity_pct", "mean"),
        humidity_max_pct=("humidity_pct", "max"),
        solar_energy_wh_m2_day=(
            "solar_irradiance_wh_m2", "sum"
        ),
        wind_speed_mean_m_s=("wind_speed_m_s", "mean"),
        cooling_degree_hours=("cooling_degree", "sum"),
        weather_hour_count=("hour", "count"),
    )
)

daily_weather["solar_energy_kwh_m2_day"] = (
    daily_weather["solar_energy_wh_m2_day"] / 1000
)

daily_weather["weather_quality_flag"] = "valid"

daily_weather.loc[
    daily_weather["weather_hour_count"] < 24,
    "weather_quality_flag"
] = "incomplete_day"

daily_weather.to_parquet(
    WEATHER_OUTPUT,
    index=False
)

# Load processed Andhra Pradesh grid data
grid = pd.read_parquet(GRID_INPUT)
grid["date"] = pd.to_datetime(grid["date"]).dt.normalize()

# Join actual grid demand with Guntur weather
combined = grid.merge(
    daily_weather,
    on="date",
    how="inner",
    validate="one_to_one"
)

combined = combined.sort_values("date").reset_index(drop=True)

combined.to_parquet(
    COMBINED_OUTPUT,
    index=False
)

# ML rows require a known next-day target
ml_data = combined.dropna(
    subset=["target_next_day_demand_mw"]
).copy()

for split in ["train", "validation", "test"]:
    split_data = ml_data[ml_data["split"] == split]

    split_output = Path(
        f"data/processed/{split}/"
        f"ap_grid_weather_{split}_v1.parquet"
    )

    split_output.parent.mkdir(parents=True, exist_ok=True)
    split_data.to_parquet(split_output, index=False)

# Validation report
report = pd.DataFrame({
    "metric": [
        "hourly_weather_rows",
        "daily_weather_rows",
        "weather_start_date",
        "weather_end_date",
        "incomplete_weather_days",
        "grid_weather_joined_rows",
        "joined_start_date",
        "joined_end_date",
        "ml_usable_rows",
        "train_rows",
        "validation_rows",
        "test_rows",
    ],
    "value": [
        len(weather),
        len(daily_weather),
        str(daily_weather["date"].min().date()),
        str(daily_weather["date"].max().date()),
        int(
            (
                daily_weather["weather_quality_flag"]
                != "valid"
            ).sum()
        ),
        len(combined),
        str(combined["date"].min().date()),
        str(combined["date"].max().date()),
        len(ml_data),
        int((ml_data["split"] == "train").sum()),
        int((ml_data["split"] == "validation").sum()),
        int((ml_data["split"] == "test").sum()),
    ],
})

report.to_csv(REPORT_OUTPUT, index=False)

print("\nGRID + GUNTUR WEATHER PREPARATION")
print("-" * 48)
print("Hourly weather rows:", len(weather))
print("Daily weather rows:", len(daily_weather))
print("Incomplete weather days:",
      (daily_weather["weather_quality_flag"] != "valid").sum())
print("Joined rows:", len(combined))
print("Joined date range:",
      combined["date"].min(), "to", combined["date"].max())
print("\nML split distribution:")
print(ml_data["split"].value_counts())
print("\nWeather output:", WEATHER_OUTPUT)
print("Combined output:", COMBINED_OUTPUT)
print("Report:", REPORT_OUTPUT)