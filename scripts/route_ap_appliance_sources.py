from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/appliance_inputs_v1"

inventory = pd.read_parquet(
    BASE / "ap_inventory_with_splits_v1.parquet"
)
catalog = pd.read_csv(
    ROOT / "reports/refit_trace_library_v1.csv"
)

# Candidate source labels that match the inventory categories.
REFIT = {
    "refrigerator": "refrigerator",
    "washing_machine": "washing_machine",
    "electric_kettle": "electric_kettle",
    "microwave": "microwave",
    "television": "television",
    "desktop": "desktop",
    "modem_router": "modem_router",
}

# Shared calibration evidence, not independent evaluation traces.
IAWE = {
    "refrigerator": "3",
    "air_conditioner": "4,5",
    "laptop_tablet": "7",
    "television": "10",
    "water_purifier": "11",
}

routes = []

for row in inventory.itertuples(index=False):
    appliance = row.appliance_type
    source_type = REFIT.get(appliance)

    candidates = catalog.loc[
        catalog["appliance"].eq(source_type)
        & catalog["split"].eq(row.split)
    ]

    references = [
        f"refit_h{int(h):02d}_c{int(c):02d}"
        for h, c in zip(candidates["house"], candidates["channel"])
    ]

    if row.inventory_status == "NOT_OWNED":
        status = "INACTIVE_REPORTED_NOT_OWNED"
        references = []
    elif references:
        status = "SAME_SPLIT_TRACE_CANDIDATES"
    else:
        status = "DOCUMENTED_SIMULATION_MODEL_REQUIRED"

    routes.append({
        "template_id": row.template_id,
        "appliance_type": appliance,
        "split": row.split,
        "inventory_status": row.inventory_status,
        "source_route": status,
        "refit_candidate_channels": "|".join(references),
        "iawe_calibration_channels": IAWE.get(appliance, ""),
        "specific_trace_assigned": False,
        "quantity_resolved": row.inventory_status in {
            "NOT_OWNED", "OWNED_QUANTITY_KNOWN"
        },
        "model_parameters_fitted": False,
        "simulator_ready": False,
    })

result = pd.DataFrame(routes)

assert len(result) == len(inventory)
assert not result.duplicated(["template_id", "appliance_type"]).any()

path = BASE / "ap_appliance_source_routes_v1.parquet"
result.to_parquet(path, index=False)

summary = (
    result.groupby(["appliance_type", "source_route"])
    .size()
    .rename("inventory_rows")
    .reset_index()
)
summary.to_csv(
    ROOT / "reports/ap_appliance_source_routes_v1.csv",
    index=False,
)

print("AP APPLIANCE SOURCE ROUTES BUILT")
print("Inventory rows:", len(result))
print(result["source_route"].value_counts().to_string())
print("\nModels needed for these potentially active categories:")
missing = result.loc[
    result.source_route.eq("DOCUMENTED_SIMULATION_MODEL_REQUIRED")
]
print(
    missing.groupby("appliance_type")
    .size()
    .rename("inventory_rows")
    .to_string()
)
print("\nSaved:", path)