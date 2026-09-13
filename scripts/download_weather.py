from pathlib import Path
import requests

# Guntur, Andhra Pradesh
LATITUDE = 16.3067
LONGITUDE = 80.4365

output = Path(
    "data/raw/weather/"
    "nasa_power_guntur_ap_hourly_2021_2025.csv"
)
output.parent.mkdir(parents=True, exist_ok=True)

url = "https://power.larc.nasa.gov/api/temporal/hourly/point"

params = {
    "parameters": "T2M,RH2M,ALLSKY_SFC_SW_DWN,WS10M",
    "community": "RE",
    "longitude": LONGITUDE,
    "latitude": LATITUDE,
    "start": "20210101",
    "end": "20251231",
    "format": "CSV",
    "time-standard": "LST"
}

print("Downloading recent weather for Guntur, Andhra Pradesh...")

response = requests.get(url, params=params, timeout=300)
response.raise_for_status()

if len(response.content) < 1000:
    raise RuntimeError(
        "The downloaded response is unexpectedly small:\n"
        + response.text[:500]
    )

output.write_bytes(response.content)

print("Saved:", output)
print("Size:", round(output.stat().st_size / 1024 / 1024, 2), "MB")