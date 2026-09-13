from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/reside_ac/extracted/dataset"
REPORT = ROOT / "reports/reside_ac_preview.txt"

sections = []

for path in sorted(SOURCE.rglob("*.csv")):
    print("Inspecting:", path.name, flush=True)

    try:
        data = pd.read_csv(path, encoding="utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        data = pd.read_csv(path, encoding="cp1252")
        encoding = "cp1252"

    sections.append(
        f"\nFILE: {path.relative_to(SOURCE)}\n"
        f"Encoding used: {encoding}\n"
        f"Shape: {data.shape}\n"
        f"Columns: {data.columns.tolist()}\n"
    )

    if path.parent.name in {"Envilog", "Garud"}:
        sections.append(data.head(3).to_string(index=False))
        sections.append(
            "\nLast row:\n" + data.tail(1).to_string(index=False)
        )
    else:
        sections.append(data.to_string(index=False))

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(sections), encoding="utf-8")
print("\nSaved:", REPORT)