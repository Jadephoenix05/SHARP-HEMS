from pathlib import Path
import re
import pandas as pd

source = Path("data/raw/refit")
output = Path("reports/refit_raw_inventory.csv")

files = list(source.glob("House_*.csv"))
files.sort(key=lambda p: int(re.search(r"House_(\d+)", p.name).group(1)))

expected_columns = [
    "Time", "Unix", "Aggregate",
    "Appliance1", "Appliance2", "Appliance3",
    "Appliance4", "Appliance5", "Appliance6",
    "Appliance7", "Appliance8", "Appliance9"
]

records = []

for file in files:
    house_id = int(re.search(r"House_(\d+)", file.name).group(1))
    header = pd.read_csv(file, nrows=0).columns.tolist()

    records.append({
        "house_id": house_id,
        "file_name": file.name,
        "size_mb": round(file.stat().st_size / (1024 ** 2), 2),
        "column_count": len(header),
        "schema_valid": header == expected_columns,
        "columns": "|".join(header)
    })

inventory = pd.DataFrame(records)
output.parent.mkdir(parents=True, exist_ok=True)
inventory.to_csv(output, index=False)

expected_houses = set(range(1, 22)) - {14}
actual_houses = set(inventory["house_id"])

print("\nREFIT RAW-DATA AUDIT")
print("-" * 40)
print("Files found:", len(inventory))
print("Total size (GB):", round(inventory["size_mb"].sum() / 1024, 2))
print("Valid schemas:", int(inventory["schema_valid"].sum()))
print("Missing houses:", sorted(expected_houses - actual_houses))
print("Unexpected houses:", sorted(actual_houses - expected_houses))
print("Duplicate house IDs:", int(inventory["house_id"].duplicated().sum()))
print("Report:", output)

if (
    len(inventory) == 20
    and inventory["schema_valid"].all()
    and actual_houses == expected_houses
):
    print("\nSTATUS: PASS")
else:
    print("\nSTATUS: CHECK REQUIRED")