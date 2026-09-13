from pathlib import Path
from zipfile import ZipFile
from io import TextIOWrapper, BytesIO
from itertools import islice

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/emarc/prayas-energy.zip"
REPORT = ROOT / "reports/emarc_archive_preview.txt"

sections = []

with ZipFile(ARCHIVE) as archive:
    for name in archive.namelist():
        if name.endswith("/"):
            continue

        suffix = Path(name).suffix.lower()

        if suffix == ".txt":
            text = archive.read(name).decode("utf-8-sig", errors="replace")
            sections.append(f"\nFILE: {name}\n{text}")

        elif suffix == ".csv":
            with archive.open(name) as stream:
                with TextIOWrapper(
                    stream, encoding="utf-8-sig", errors="replace"
                ) as text:
                    preview = "".join(islice(text, 6))
            sections.append(f"\nFILE: {name}\n{preview}")

        elif suffix == ".xlsx":
            with pd.ExcelFile(BytesIO(archive.read(name))) as workbook:
                sections.append(
                    f"\nFILE: {name}\nSheets: {workbook.sheet_names}"
                )
                for sheet in workbook.sheet_names:
                    data = pd.read_excel(
                        workbook, sheet_name=sheet, header=None, nrows=8
                    )
                    sections.append(
                        f"\nSHEET: {sheet}\n"
                        + data.to_string(index=False, header=False)
                    )

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(sections), encoding="utf-8")
print("Saved:", REPORT)