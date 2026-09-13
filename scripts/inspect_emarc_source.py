from pathlib import Path
import json
from collections import Counter
from zipfile import ZipFile

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/emarc"
REPORT = ROOT / "reports/emarc_source_and_blocks.txt"

sections = []

# Retrieve the original repository's description, terms and file inventory.
url = "https://dataverse.harvard.edu/api/datasets/:persistentId/"
try:
    response = requests.get(
        url,
        params={"persistentId": "doi:10.7910/DVN/YJ5SP1"},
        timeout=60,
    )
    response.raise_for_status()
    metadata = response.json()
    if metadata.get("status") != "OK":
        raise ValueError("Dataverse did not return an OK response")

    (RAW / "harvard_source_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    version = metadata["data"]["latestVersion"]
    sections.append(
        "ORIGINAL SOURCE METADATA\n"
        + json.dumps(
            {
                key: version.get(key)
                for key in [
                    "versionNumber", "versionMinorNumber",
                    "versionState", "releaseTime", "license",
                    "termsOfUse", "termsOfAccess",
                ]
            },
            indent=2,
        )
    )

    for field in version.get("metadataBlocks", {}).get(
        "citation", {}
    ).get("fields", []):
        if field.get("typeName") in [
            "title", "author", "dsDescription",
            "publication", "notes",
        ]:
            sections.append(json.dumps(field, indent=2))

    sections.append("\nORIGINAL FILE INVENTORY")
    for item in version.get("files", []):
        file = item["dataFile"]
        sections.append(json.dumps({
            "id": file.get("id"),
            "name": file.get("filename"),
            "description": item.get("description"),
            "restricted": item.get("restricted"),
            "size": file.get("filesize"),
        }))

except Exception as error:
    sections.append(f"SOURCE METADATA UNAVAILABLE: {error}")

# Inspect every block value without loading the full CSV into memory.
counts = Counter()
total_rows = 0
missing_blocks = 0

with ZipFile(RAW / "prayas-energy.zip") as archive:
    with archive.open("Prayas Energy/eMARC load blocks.csv") as stream:
        for chunk in pd.read_csv(
            stream,
            usecols=["block"],
            dtype={"block": "string"},
            chunksize=250000,
        ):
            total_rows += len(chunk)
            missing_blocks += int(chunk["block"].isna().sum())
            counts.update(chunk["block"].dropna().str.strip())

sections.append(f"\nLOAD-BLOCK ROWS: {total_rows}")
sections.append(f"MISSING BLOCK VALUES: {missing_blocks}")
sections.append("BLOCK VALUES AND ROW COUNTS")
sections.extend(
    f"{value}: {count}"
    for value, count in sorted(counts.items())
)

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(sections), encoding="utf-8")

print("Load-block rows:", total_rows)
print("Distinct block values:", len(counts))
print("Missing block values:", missing_blocks)
print("Saved:", REPORT)