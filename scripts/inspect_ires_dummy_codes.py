import ast
from pathlib import Path
import pandas as pd

builder_path = Path(r"scripts\build_ires_household_priors.py")
source = builder_path.read_text(encoding="utf-8")
tree = ast.parse(source)

dummy_columns = None

for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "dummy_sources":
                dummy_columns = ast.literal_eval(node.value)

if dummy_columns is None:
    raise RuntimeError("Could not find dummy_sources in builder script")

data = pd.read_csv(
    r"data\raw\ires\ires_2020_data.tab",
    sep="\t",
    usecols=dummy_columns,
    low_memory=False,
)

print("IRES DUMMY CODE AUDIT")
print("-" * 70)

total_bad = 0

for column in dummy_columns:
    values = pd.to_numeric(data[column], errors="coerce")
    counts = values.value_counts(dropna=False).sort_index()
    bad = values[~values.isna() & ~values.isin([0, 1, 99])]

    if len(bad):
        total_bad += len(bad)
        print(f"\n{column}")
        print("Unexpected count:", len(bad))
        print("Unexpected codes:")
        print(bad.value_counts().sort_index().to_string())
        print("All codes:")
        print(counts.to_string())

print("\nTotal unexpected values:", total_bad)