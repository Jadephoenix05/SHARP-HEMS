from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/household_templates/ap_appliance_ownership_v2.parquet"
OUT = ROOT / "data/external_config/appliances"
REPORTS = ROOT / "reports"

# Candidate sources only; these are not approved simulator assignments.
# appliance: (Indian candidate, supplementary REFIT candidate, model requirement)
MODELS = {
    "air_conditioner": (
        "iAWE channels 4 and 5; RESIDE-AC events and temperature",
        "",
        "Power and thermal model; verify events, units and coverage",
    ),
    "air_cooler": ("", "", "Simulated power and usage model required"),
    "ceiling_fan": ("", "", "Simulated speed-dependent power and usage model required"),
    "desktop": (
        "", "Desktop Computer",
        "Review channel notes and transfer assumptions",
    ),
    "electric_coil": ("", "", "Simulated cooking-use model required if included"),
    "electric_grill_toaster": (
        "", "Toaster",
        "Toaster-only candidate; does not represent all grills or ovens",
    ),
    "electric_iron": (
        "iAWE channel 8", "",
        "Sparse observations; cycle model requires further support",
    ),
    "electric_kettle": (
        "", "Kettle",
        "Review channel notes; simulated usage schedule required",
    ),
    "electric_rice_cooker": ("", "", "Simulated cooking-cycle model required"),
    "geyser": (
        "", "",
        "Water-heating model required; survey reports use, not ownership",
    ),
    "induction_cooktop": ("", "", "Simulated cooking-use model required if included"),
    "laptop_tablet": (
        "iAWE channel 7", "",
        "Laptop-only candidate; does not establish tablet power",
    ),
    "led_bulb": ("", "", "Documented wattage and lighting schedule required"),
    "led_tube": ("", "", "Documented wattage and lighting schedule required"),
    "microwave": (
        "", "Microwave",
        "Review channel notes and transfer assumptions if included",
    ),
    "mixer_grinder": (
        "", "Food Mixer; Magimix (Blender); Kenwood KMix",
        "Related appliances only; Indian mixer-grinder model needs validation",
    ),
    "modem_router": (
        "", "Router",
        "Review channel notes and transfer assumptions",
    ),
    "refrigerator": (
        "iAWE channel 3", "Fridge",
        "Review power cycles and coverage; fridge-freezers are separate candidates",
    ),
    "table_fan": ("", "", "Simulated speed-dependent power and usage model required"),
    "television": (
        "iAWE channel 10", "Television",
        "Review coverage; exclude Television Site as an isolated-TV trace",
    ),
    "washing_machine": (
        "iAWE channel 6", "Washing Machine",
        "Extract validated cycles; iAWE coverage is sparse",
    ),
    "water_pump": (
        "iAWE channel 12", "",
        "Resolve sampling limitations and model water-service requirements",
    ),
    "water_purifier": (
        "iAWE channel 11", "",
        "Confirm water-filter equipment matches intended purifier",
    ),
}


def main():
    inventory = pd.read_parquet(INPUT)
    found = set(inventory["appliance_type"])

    if found != set(MODELS):
        raise ValueError(
            f"Appliance mismatch: missing={found - set(MODELS)}, "
            f"extra={set(MODELS) - found}"
        )

    if inventory.duplicated(["template_id", "appliance_type"]).any():
        raise ValueError("Duplicate household/appliance entries")

    rows = []
    for appliance, (indian, refit, requirement) in MODELS.items():
        group = inventory.loc[inventory["appliance_type"].eq(appliance)]

        # Geyser use is deliberately separate from ownership.
        geyser_users = (
            int(group["geyser_use_status"].eq("REPORTED_USE").sum())
            if appliance == "geyser" else 0
        )

        rows.append({
            "appliance_type": appliance,
            "survey_templates": group["template_id"].nunique(),
            "inventory_available_profiles": int(group["available"].eq(1).sum()),
            "inventory_unavailable_profiles": int(group["available"].eq(0).sum()),
            "inventory_availability_unknown_profiles": int(group["available"].isna().sum()),
            "reported_geyser_use_profiles": geyser_users,
            "indian_candidate_source": indian,
            "supplementary_uk_candidate": refit,
            "model_requirement": requirement,
            "model_status": (
                "CANDIDATE_SOURCE_REVIEW_REQUIRED"
                if indian or refit else "MODEL_REQUIRED"
            ),
            "assigned_trace": "",
            "rated_power_w": None,
            "control_permission": "UNASSIGNED",
            "automatic_control_enabled": False,
            "simulator_ready": False,
        })

    plan = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    destination = OUT / "ap_appliance_model_plan_v1.csv"
    plan.to_csv(destination, index=False)

    summary = plan[[
        "appliance_type",
        "inventory_available_profiles",
        "reported_geyser_use_profiles",
        "model_status",
    ]]
    summary.to_csv(
        REPORTS / "ap_appliance_model_plan_summary_v1.csv",
        index=False,
    )

    print(summary.to_string(index=False))
    print("\nAppliance types:", len(plan))
    print("Saved:", destination)
    print("No trace assignments or power ratings have been invented.")


if __name__ == "__main__":
    main()