import json
from pathlib import Path

path = Path(r"data\raw\ires\ires_dataset_metadata.json")

with path.open("r", encoding="utf-8-sig") as f:
    response = json.load(f)

if response.get("status") != "OK":
    raise RuntimeError(f"Dataverse response failed: {response}")

version = response["data"]["latestVersion"]
files = version.get("files", [])

print("IRES 2020 DATASET INVENTORY")
print("-" * 90)
print("Dataset version:", version.get("versionNumber"))
print("Release date:", version.get("releaseTime"))
print("File count:", len(files))
print()

for item in files:
    data_file = item["dataFile"]
    file_id = data_file.get("id")
    name = data_file.get("filename")
    size_mb = data_file.get("filesize", 0) / (1024 * 1024)
    file_type = data_file.get("contentType")
    restricted = item.get("restricted", False)

    print(
        f"ID={file_id:<10} "
        f"SIZE={size_mb:8.2f} MB  "
        f"RESTRICTED={str(restricted):<5}  "
        f"TYPE={file_type}  "
        f"NAME={name}"
    )