from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/emarc/prayas-energy.zip"
OUT = ROOT / "data/interim/emarc"
REPORTS = ROOT / "reports"

PREFIX = "Prayas Energy/"
BLOCKS = PREFIX + "eMARC load blocks.csv"
METADATA = PREFIX + "Household-Deployment basic info.xlsx"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def main():
    target = OUT / "emarc_load_blocks_native_v1.parquet"
    temporary = OUT / "emarc_load_blocks_native_v1.parquet.partial"
    writer = None

    total = 0
    counts = {}
    deployments = set()
    households = set()
    first_date = None
    last_date = None

    try:
        with ZipFile(ARCHIVE) as archive:
            metadata = pd.read_excel(
                BytesIO(archive.read(METADATA)),
                dtype=str,
            )
            metadata.columns = metadata.columns.str.strip()

            required = [
                "deployment_id",
                "household_id",
                "Deployment type",
                "Region",
                "Household type",
            ]
            missing = set(required) - set(metadata.columns)
            if missing:
                raise ValueError(f"Missing metadata columns: {missing}")

            metadata = metadata[required].copy()
            for column in required:
                metadata[column] = metadata[column].str.strip()
                if (
                    metadata[column].isna().any()
                    or metadata[column].eq("").any()
                ):
                    raise ValueError(f"Missing metadata: {column}")

            if metadata["deployment_id"].duplicated().any():
                raise ValueError("Duplicate deployment IDs in metadata")

            metadata = metadata.rename(columns={
                "Deployment type": "deployment_type_source",
                "Region": "region_source",
                "Household type": "household_type_source",
            })

            metadata["meter_role"] = (
                metadata["deployment_type_source"]
                .str.casefold()
                .map({
                    "mainline": "mainline",
                    "appliance": "appliance",
                })
            )
            if metadata["meter_role"].isna().any():
                unknown = metadata.loc[
                    metadata["meter_role"].isna(),
                    "deployment_type_source",
                ].unique()
                raise ValueError(f"Unknown deployment types: {unknown}")

            with archive.open(BLOCKS) as source:
                reader = pd.read_csv(
                    source,
                    chunksize=500_000,
                    dtype={"deployment_id": str, "date": str},
                )

                for chunk in reader:
                    chunk.columns = chunk.columns.str.strip()
                    chunk["deployment_id"] = (
                        chunk["deployment_id"].str.strip()
                    )

                    dates = pd.to_datetime(
                        chunk["date"],
                        format="%m/%d/%Y",
                        errors="coerce",
                    )
                    block = pd.to_numeric(
                        chunk["block"], errors="coerce"
                    )
                    load = pd.to_numeric(
                        chunk["Load (kW)"], errors="coerce"
                    )

                    if dates.isna().any():
                        raise ValueError("Unparseable source dates")
                    if (
                        block.isna().any()
                        or not block.between(0, 95).all()
                        or not block.mod(1).eq(0).all()
                    ):
                        raise ValueError("Invalid block values")
                    if (
                        load.isna().any()
                        or not np.isfinite(load).all()
                        or load.lt(0).any()
                    ):
                        raise ValueError("Invalid load measurements")

                    data = pd.DataFrame({
                        # One-based data-row number, excluding CSV header.
                        "source_row": np.arange(
                            total + 1,
                            total + len(chunk) + 1,
                            dtype=np.int64,
                        ),
                        "deployment_id": chunk["deployment_id"].to_numpy(),
                        "source_date_text": chunk["date"].to_numpy(),
                        "source_date": dates.to_numpy(),
                        "block": block.astype("int16").to_numpy(),
                        "load_kw": load.astype("float64").to_numpy(),
                    })

                    data = data.merge(
                        metadata,
                        on="deployment_id",
                        how="left",
                        validate="many_to_one",
                        indicator=True,
                        sort=False,
                    )

                    unmatched = data["_merge"].ne("both")
                    if unmatched.any():
                        ids = data.loc[
                            unmatched, "deployment_id"
                        ].unique()
                        raise ValueError(
                            f"Deployments missing metadata: {ids[:20]}"
                        )
                    data = data.drop(columns="_merge")

                    table = pa.Table.from_pandas(
                        data, preserve_index=False
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary,
                            table.schema,
                            compression="snappy",
                        )
                    writer.write_table(table)

                    total += len(data)
                    deployments.update(data["deployment_id"].unique())
                    households.update(data["household_id"].unique())

                    for role, count in data["meter_role"].value_counts().items():
                        counts[role] = counts.get(role, 0) + int(count)

                    start = dates.min()
                    end = dates.max()
                    first_date = start if first_date is None else min(first_date, start)
                    last_date = end if last_date is None else max(last_date, end)

                    print(f"Prepared {total:,} rows", flush=True)

            if writer is None:
                raise ValueError("No load records found")

            writer.close()
            writer = None

            saved_rows = pq.ParquetFile(temporary).metadata.num_rows
            if saved_rows != total:
                raise ValueError("Parquet row count mismatch")

            temporary.replace(target)
            metadata.to_csv(
                OUT / "emarc_deployment_registry_v1.csv",
                index=False,
            )

        report = {
            "status": "PREPARED_PENDING_TEMPORAL_QA",
            "source_rows": total,
            "output_rows": saved_rows,
            "observed_deployments": len(deployments),
            "observed_households": len(households),
            "mainline_rows": counts.get("mainline", 0),
            "appliance_rows": counts.get("appliance", 0),
            "first_source_date": str(first_date.date()),
            "last_source_date": str(last_date.date()),
            "unmatched_deployment_rows": 0,
            "duplicate_key_check": "PENDING",
            "daily_consumption_reconciliation": "PENDING",
            "block_clock_mapping": "UNCONFIRMED",
            "timezone": "UNCONFIRMED",
            "redistribution_license": "UNVERIFIED",
            "release_ready": False,
        }

        pd.DataFrame(
            report.items(), columns=["metric", "value"]
        ).to_csv(
            REPORTS / "emarc_preparation_v1.csv", index=False
        )

        provenance = {
            "archive": ARCHIVE.relative_to(ROOT).as_posix(),
            "archive_member": BLOCKS,
            "mirror": "basu1999/prayas-energy",
            "referenced_source_doi": "10.7910/DVN/YJ5SP1",
            "measurement": "Load (kW), preserved from source",
            "processing": [
                "Joined deployment metadata with many-to-one validation",
                "Preserved all source rows and duplicate keys",
                "Did not fill missing intervals or infer clock timestamps",
                "Did not combine mainline and appliance loads",
            ],
            "validation": report,
        }
        (OUT / "emarc_provenance_v1.json").write_text(
            json.dumps(provenance, indent=2),
            encoding="utf-8",
        )

        print("\nEMARC PREPARATION COMPLETE")
        for key, value in report.items():
            print(f"{key}: {value}")
        print("Output:", target)

    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()