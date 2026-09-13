"""NASA POWER hourly weather at the RESIDE-AC city for BOTH candidate years.

RESIDE-AC is documented as Hyderabad; CSV timestamps read 2019 while the
dataset description states 2021. That discrepancy is unresolved upstream, so
both years are downloaded and neither is assumed correct here.

This is city-level MERRA-2 reanalysis on a 0.5 x 0.625 degree grid. It is NOT
a site measurement at the RESIDE houses and must never be labelled as one.
Guntur weather is the deployment-scenario input and is NOT used to fit RESIDE.
"""
from pathlib import Path
import argparse
import hashlib
import json
import requests

# Hyderabad, Telangana. RESIDE does not publish per-house coordinates;
# NASA POWER resolves to a ~0.5 degree cell, so this is a city-cell proxy.
LATITUDE = 17.3850
LONGITUDE = 78.4867
CITY = "hyderabad_telangana"

URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS = "T2M,RH2M,ALLSKY_SFC_SW_DWN,WS10M"

# Padded either side of the 10-28 May RESIDE window.
WINDOWS = {2019: ("20190501", "20190605"), 2021: ("20210501", "20210605")}


def fetch(year, start, end, out_dir, root):
    params = {
        "parameters": PARAMETERS,
        "community": "RE",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": start,
        "end": end,
        "format": "CSV",
        "time-standard": "LST",
    }
    print(f"Downloading {CITY} {year} ({start}-{end})...", flush=True)
    response = requests.get(URL, params=params, timeout=300)
    response.raise_for_status()
    body = response.content
    if len(body) < 1000:
        raise RuntimeError(f"{year}: response too small:\n{response.text[:500]}")
    if b"-END HEADER-" not in body:
        raise RuntimeError(f"{year}: NASA POWER header missing; unexpected payload")
    path = out_dir / f"nasa_power_{CITY}_hourly_{year}_may.csv"
    path.write_bytes(body)
    rows = body.decode("utf-8").split("-END HEADER-", 1)[1].strip().splitlines()
    # The first line after the header is the column row.
    data_rows = len(rows) - 1
    expected = 36 * 24  # 1 May through 5 June inclusive
    if data_rows != expected:
        raise RuntimeError(f"{year}: expected {expected} hourly rows, got {data_rows}")
    print(f"  saved {path.name}: {data_rows} hourly rows", flush=True)
    return {
        "year": year,
        "path": path.relative_to(root).as_posix(),
        "hourly_rows": data_rows,
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = root / "data/raw/weather"
    out_dir.mkdir(parents=True, exist_ok=True)

    downloads = [fetch(y, *w, out_dir, root) for y, w in sorted(WINDOWS.items())]

    report = {
        "status": "RESIDE_SITE_CANDIDATE_WEATHER_DOWNLOADED",
        "city": CITY,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "source": "NASA POWER hourly MERRA-2 / CERES, LST",
        "downloads": downloads,
        "year_discrepancy_resolved": False,
        "site_measured_weather": False,
        "spatial_basis": "CITY_GRID_CELL_REANALYSIS_NOT_HOUSE_SITE",
        "used_to_fit_thermal_model": False,
        "guntur_weather_used_for_reside_fitting": False,
        "full_simulator_ready": False,
        "master_release_ready": False,
    }
    path = root / "reports/reside_site_weather_download_v1.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nRESIDE SITE CANDIDATE WEATHER DOWNLOADED")
    print("City:", CITY, f"({LATITUDE}, {LONGITUDE})")
    print("Years:", ", ".join(str(d["year"]) for d in downloads))
    print("Year discrepancy resolved: False")
    print("Report:", path)


if __name__ == "__main__":
    main()
