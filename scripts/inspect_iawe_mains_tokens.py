from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/iawe/extracted/iawe/iawe/electricity/electricity"
OUTPUT = ROOT / "reports/iawe_mains_parse_tokens.csv"

records = []

for channel in [1, 2]:
    counts = Counter()

    for chunk in pd.read_csv(
        SOURCE / f"{channel}.csv",
        chunksize=100000,
        low_memory=False,
    ):
        for column in chunk.columns:
            if column == "timestamp":
                continue

            numeric = pd.to_numeric(chunk[column], errors="coerce")
            bad = chunk[column].notna() & numeric.isna()

            counts.update(
                (column, str(value))
                for value in chunk.loc[bad, column]
            )

    for (column, token), count in counts.items():
        records.append({
            "channel": channel,
            "column": column,
            "source_token": token,
            "count": count,
        })

report = pd.DataFrame(
    records, columns=["channel", "column", "source_token", "count"]
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(OUTPUT, index=False)
print(report.to_string(index=False))
print("\nSaved:", OUTPUT)