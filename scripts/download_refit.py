from pathlib import Path
import requests

RECORD_ID = "5063428"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
OUT_DIR = Path("data/raw/refit")
CHUNK_SIZE = 1024 * 1024  # 1 MB

OUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "SHARP-Research-Dataset/1.0"
})

print("Reading REFIT file information...")

response = session.get(API_URL, timeout=60)
response.raise_for_status()
record = response.json()

files = record.get("files", [])

if not files:
    raise RuntimeError("No files were found in the Zenodo record.")

print(f"Found {len(files)} file(s).")

for item in files:
    name = item["key"]
    url = item["links"]["self"]
    destination = OUT_DIR / name
    expected_size = item.get("size", 0)

    if destination.exists() and destination.stat().st_size == expected_size:
        print(f"Already complete: {name}")
        continue

    print(f"\nDownloading: {name}")
    print(f"Destination: {destination}")

    downloaded = 0

    with session.get(url, stream=True, timeout=(30, 300)) as download:
        download.raise_for_status()

        with destination.open("wb") as file:
            for chunk in download.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue

                file.write(chunk)
                downloaded += len(chunk)

                if expected_size:
                    percentage = downloaded * 100 / expected_size
                    print(
                        f"\rProgress: {percentage:6.2f}% "
                        f"({downloaded / 1_073_741_824:.2f} GB)",
                        end="",
                    )

    print(f"\nCompleted: {name}")

print("\nREFIT download complete!")