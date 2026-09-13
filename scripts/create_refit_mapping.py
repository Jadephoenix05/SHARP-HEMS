import re
from pathlib import Path
import pandas as pd

readme_path = Path(r"data\raw\refit\REFIT_Readme.txt")
output_path = Path(r"data\registry\refit_appliance_mapping.csv")
report_path = Path(r"reports\refit_appliance_mapping_validation.csv")

expected_houses = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 15, 16, 17, 18, 19, 20, 21
}

text = readme_path.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

records = []
current_house = None
mapping_lines = []
collecting = False


def process_house(house_id, lines_to_process):
    if house_id is None or not lines_to_process:
        return []

    combined = " ".join(lines_to_process)
    combined = re.sub(r"\s+", " ", combined).strip()
    combined = combined.replace(
    "2.Tumble Dryer 3.Washing Machine",
    "2.Tumble Dryer, 3.Washing Machine"
)

    matches = re.findall(
        r"(?:^|,\s*)(\d+)\.\s*(.*?)(?=,\s*\d+\.|$)",
        combined
    )

    house_records = []

    for channel_text, appliance_name in matches:
        channel = int(channel_text)
        appliance_name = appliance_name.strip(" ,")

        if 0 <= channel <= 9:
            house_records.append({
                "house_id": house_id,
                "channel": channel,
                "source_column": (
                    "Aggregate" if channel == 0
                    else f"Appliance{channel}"
                ),
                "appliance_name": appliance_name,
                "channel_type": (
                    "aggregate" if channel == 0
                    else "submeter"
                )
            })

    return house_records


for line in lines:
    stripped = line.strip()
    house_match = re.fullmatch(r"House\s+(\d+)", stripped, flags=re.I)

    if house_match:
        if collecting:
            records.extend(process_house(current_house, mapping_lines))

        current_house = int(house_match.group(1))
        mapping_lines = []
        collecting = current_house in expected_houses
        continue

    if collecting:
        if stripped.upper().startswith("!NOTES"):
            records.extend(process_house(current_house, mapping_lines))
            current_house = None
            mapping_lines = []
            collecting = False
        else:
            mapping_lines.append(stripped)

if collecting:
    records.extend(process_house(current_house, mapping_lines))

mapping = pd.DataFrame(records)

mapping = (
    mapping[
        mapping["house_id"].isin(expected_houses)
        & mapping["channel"].between(0, 9)
    ]
    .drop_duplicates(subset=["house_id", "channel"])
    .sort_values(["house_id", "channel"])
    .reset_index(drop=True)
)

found_houses = set(mapping["house_id"].unique())
missing_houses = sorted(expected_houses - found_houses)

counts = mapping.groupby("house_id")["channel"].nunique()
incomplete_houses = counts[counts != 10].index.tolist()

duplicate_count = int(
    mapping.duplicated(["house_id", "channel"]).sum()
)

status = (
    "PASS"
    if (
        len(mapping) == 200
        and not missing_houses
        and not incomplete_houses
        and duplicate_count == 0
    )
    else "FAIL"
)

output_path.parent.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)

mapping.to_csv(output_path, index=False)

report = pd.DataFrame([
    {"metric": "status", "value": status},
    {"metric": "expected_houses", "value": 20},
    {"metric": "houses_found", "value": len(found_houses)},
    {"metric": "expected_mapping_rows", "value": 200},
    {"metric": "mapping_rows", "value": len(mapping)},
    {"metric": "duplicate_channels", "value": duplicate_count},
    {
        "metric": "missing_houses",
        "value": ",".join(map(str, missing_houses)) or "none"
    },
    {
        "metric": "incomplete_houses",
        "value": ",".join(map(str, incomplete_houses)) or "none"
    }
])

report.to_csv(report_path, index=False)

print("\nREFIT APPLIANCE MAPPING")
print("-" * 50)
print(f"Status: {status}")
print(f"Houses found: {len(found_houses)}/20")
print(f"Mapping rows: {len(mapping)}/200")
print(f"Duplicate channels: {duplicate_count}")
print(f"Missing houses: {missing_houses or 'None'}")
print(f"Incomplete houses: {incomplete_houses or 'None'}")
print(f"Output: {output_path}")
print(f"Report: {report_path}")