"""Collect small integration inputs. Does not change source tables or delete data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import hashlib
import json
import tempfile
import pandas as pd
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "data_registry/ap_household_bundle_v1.json"
    if not manifest_path.is_file():
        raise SystemExit("Run from SHARP_Master_Dataset or pass --root.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = set(manifest["files"].values())
    paths.update([
        "data/interim/indian_priors/bee_clasp_national_priors_v1.csv",
        "data/registry/refit_appliance_mapping.csv",
        "data/interim/emarc/emarc_deployment_registry_v1.csv",
        "data/interim/occupancy_profiles/tus2024_ap_complete_location_diaries_v1.parquet",
        "data/raw/weather/nasa_power_guntur_ap_hourly_2021_2025.csv",
        "data/processed/grid_forecast/ap_grid_weather_daily_v1.parquet",
    ])
    patterns = [
        "data/processed/refit_canonical_v1/*.parquet",
        "data/interim/iawe/15min/*.parquet",
        "data/interim/reside_ac_clean/timezone_aligned/*.parquet",
        "data/interim/reside_ac_clean/*metadata*.parquet",
    ]
    for pattern in patterns:
        paths.update(p.relative_to(root).as_posix() for p in root.glob(pattern))
    destination = root / "sharp_integration_inputs.zip"
    partial = root / "sharp_integration_inputs.zip.partial"
    entries = []
    full_budget = 24 * 1024**2
    with tempfile.TemporaryDirectory() as temp:
        try:
            with ZipFile(partial, "w", ZIP_DEFLATED) as archive:
                for relative in sorted(paths):
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError("Input outside project")
                    entry = {"source_path": relative}
                    if not path.is_file():
                        entry["status"] = "MISSING"
                        entries.append(entry)
                        continue
                    size = path.stat().st_size
                    entry["source_bytes"] = size
                    # Always sample electrical traces; their full files stay local.
                    trace = any(s in relative for s in (
                        "refit_canonical", "/iawe/", "/timezone_aligned/"
                    ))
                    if path.suffix == ".parquet":
                        pf = pq.ParquetFile(path)
                        entry["rows"] = pf.metadata.num_rows
                        entry["schema"] = str(pf.schema_arrow)
                        if trace or size > min(8 * 1024**2, full_budget):
                            sample = next(pf.iter_batches(batch_size=192), None)
                            if sample is not None:
                                output = Path(temp) / "sample.parquet"
                                pq.write_table(
                                    __import__("pyarrow").Table.from_batches([sample]), output
                                )
                                name = "samples/" + relative
                                archive.write(output, name)
                                entry.update(status="SAMPLE_ONLY", archive_path=name,
                                             sample_rows=sample.num_rows)
                            else:
                                entry["status"] = "EMPTY"
                            entries.append(entry)
                            continue
                    if size <= min(8 * 1024**2, full_budget):
                        name = "inputs/" + relative
                        archive.write(path, name)
                        entry.update(
                            status="FULL_COPY", archive_path=name,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                        )
                        full_budget -= size
                    else:
                        entry["status"] = "METADATA_ONLY_SIZE_LIMIT"
                    entries.append(entry)
                # Only known validation reports, not arbitrary scripts or credentials.
                report_names = [
                    "refit_schema_upgrade_v1.csv",
                    "iawe_15min_validation_v1.csv",
                    "reside_ac_time_validation_v1.csv",
                    "emarc_daily_validation_v1.csv",
                    "tus2024_ap_location_validation.csv",
                    "ires_household_priors_validation_v1.csv",
                ]
                for name in report_names:
                    p = root / "reports" / name
                    if p.is_file() and p.stat().st_size < 1024**2:
                        archive.write(p, "reports/" + name)
                archive.writestr("integration_inventory.json", json.dumps({
                    "purpose": "Implementation inputs, not a release or validation dataset",
                    "sample_policy": "First 192 rows of electrical files; not representative",
                    "household_manifest": manifest,
                    "entries": entries,
                }, indent=2))
            with ZipFile(partial) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError("Archive CRC failure: " + bad)
            partial.replace(destination)
        finally:
            if partial.exists():
                partial.unlink()
    print("Saved:", destination)
    print("ZIP MiB:", round(destination.stat().st_size / 1024**2, 2))
    print(pd.Series([x["status"] for x in entries]).value_counts().to_string())
    missing = [x["source_path"] for x in entries if x["status"] == "MISSING"]
    if missing:
        print("Missing inputs:", *missing, sep="\n")
    print("Source files unchanged. Upload this ZIP for integration.")

if __name__ == "__main__":
    main()
