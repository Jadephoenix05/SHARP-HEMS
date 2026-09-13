from pathlib import Path
from collections import Counter

import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/tus_india/tus_2024/timeuse.parquet"
OUTPUT = ROOT / "data/interim/occupancy_profiles/tus2024_ap_activities.parquet"
REPORT = ROOT / "reports/tus2024_ap_code_audit.txt"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)
temporary = OUTPUT.with_suffix(".parquet.partial")

fields = [
    "survey_year", "sector", "activity_location",
    "multiple_activity", "simultaneous_activity",
    "is_major_activity", "day_of_week", "day_type",
]
counts = {name: Counter() for name in fields}
states = Counter()
scanned = 0
selected = 0

source = pq.ParquetFile(SOURCE)

# Preserve every source column, including identifiers and survey weights.
with pq.ParquetWriter(temporary, source.schema_arrow,
                      compression="snappy") as writer:
    for batch in source.iter_batches(batch_size=50000):
        scanned += batch.num_rows
        state = pc.utf8_lower(pc.utf8_trim_whitespace(batch.column("state")))
        states.update(state.to_pylist())

        mask = pc.fill_null(pc.equal(state, "andhra pradesh"), False)
        ap = batch.filter(mask)

        if ap.num_rows:
            writer.write_batch(ap)
            selected += ap.num_rows

            for name in fields:
                counts[name].update(ap.column(name).to_pylist())

        if scanned % 500000 == 0:
            print(f"Scanned {scanned:,}; AP activities {selected:,}",
                  flush=True)

lines = [
    "TUS 2024 ANDHRA PRADESH EXTRACTION",
    f"Source rows scanned: {scanned}",
    f"AP activity rows: {selected}",
    "Scope: extraction and code inventory; occupancy QA pending",
    "",
    "Source state labels and activity-row counts:",
]
for label, count in sorted(states.items(), key=lambda item: str(item[0])):
    lines.append(f"{label}: {count}")

for name, values in counts.items():
    lines.extend(["", name])
    for value, count in sorted(values.items(), key=lambda item: str(item[0])):
        lines.append(f"  {value!r}: {count}")

REPORT.write_text("\n".join(lines), encoding="utf-8")

if selected == 0:
    temporary.unlink(missing_ok=True)
    raise RuntimeError(f"No AP rows matched. Inspect {REPORT}")

temporary.replace(OUTPUT)
print(f"\nSaved {selected:,} AP activity records")
print("Output:", OUTPUT)
print("Report:", REPORT)