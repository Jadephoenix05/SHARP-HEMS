from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data/external_config/appliances"
OUT = ROOT / "data/processed/fan_lighting_scenarios"
REPORTS = ROOT / "reports"

# Experimental low/base/high power assumptions.
# Fan values represent assumed average power while running.
# Lighting values represent assumed power per illuminated unit.
WATTS = {
    "ceiling_fan": (30, 60, 80),
    "table_fan": (30, 50, 70),
    "incandescent_bulb": (40, 60, 100),
    "cfl_bulb": (9, 15, 23),
    "cfl_tube": (20, 36, 40),
    "led_bulb": (5, 9, 15),
    "led_tube": (10, 20, 24),
}


def main():
    manifest = json.loads(
        (ROOT / "data_registry/ap_household_bundle_v1.json")
        .read_text(encoding="utf-8")
    )
    files = manifest["files"]
    fans = pd.read_parquet(ROOT / files["fan_usage"])
    lights = pd.read_parquet(ROOT / files["lighting_inventory"])
    templates = pd.read_parquet(ROOT / files["household_templates"])

    if templates["template_id"].duplicated().any():
        raise ValueError("Duplicate template IDs")
    ids = set(templates["template_id"])

    if not set(fans["template_id"]).issubset(ids):
        raise ValueError("Unknown fan template IDs")
    if set(lights["template_id"]) != ids:
        raise ValueError("Lighting/template mismatch")
    if fans.duplicated(
        ["template_id", "appliance_type", "survey_fan_number"]
    ).any():
        raise ValueError("Duplicate fan records")
    if lights.duplicated(["template_id", "appliance_type"]).any():
        raise ValueError("Duplicate lighting categories")

    hours = pd.to_numeric(fans["reported_hours_daily"], errors="raise")
    counts = pd.to_numeric(lights["reported_count"], errors="raise")
    if hours.isna().any() or not hours.between(0, 24).all():
        raise ValueError("Missing or invalid fan hours")
    if counts.isna().any() or not (
        np.isfinite(counts) & counts.ge(0) & counts.mod(1).eq(0)
    ).all():
        raise ValueError("Invalid lighting counts")

    parameter_rows = []
    fan_outputs = []
    light_outputs = []

    for index, scenario in enumerate(["low", "base", "high"]):
        scenario_id = f"experimental_power_{scenario}_v1"

        for appliance, values in WATTS.items():
            parameter_rows.append({
                "scenario_id": scenario_id,
                "appliance_type": appliance,
                "assumed_operating_power_w": values[index],
                "parameter_basis": "EXPERIMENTAL_ASSUMPTION",
                "empirically_calibrated": False,
                "interpretation": "Sensitivity scenario; not a confidence interval",
            })

        f = fans.copy()
        f["scenario_id"] = scenario_id
        f["assumed_operating_power_w"] = f["appliance_type"].map(
            {key: value[index] for key, value in WATTS.items()}
        )
        if f["assumed_operating_power_w"].isna().any():
            raise ValueError("Missing fan power assumption")

        f["modeled_energy_per_reported_usage_day_kwh"] = (
            f["reported_hours_daily"]
            * f["assumed_operating_power_w"]
            / 1000
        )
        f["parameter_basis"] = "EXPERIMENTAL_ASSUMPTION"
        f["is_synthetic"] = True
        fan_outputs.append(f)

        light = lights.copy()
        light["scenario_id"] = scenario_id
        light["assumed_power_per_unit_w"] = light["appliance_type"].map(
            {key: value[index] for key, value in WATTS.items()}
        )
        if light["assumed_power_per_unit_w"].isna().any():
            raise ValueError("Missing lighting power assumption")

        light["modeled_all_units_on_power_w"] = (
            light["reported_count"] * light["assumed_power_per_unit_w"]
        )
        light["parameter_basis"] = "EXPERIMENTAL_ASSUMPTION"
        light["is_synthetic"] = True
        light_outputs.append(light)

    fan_result = pd.concat(fan_outputs, ignore_index=True)
    light_result = pd.concat(light_outputs, ignore_index=True)

    for directory in [CONFIG, OUT, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(parameter_rows).to_csv(
        CONFIG / "fan_lighting_power_assumptions_v1.csv", index=False
    )
    fan_result.to_parquet(
        OUT / "fan_usage_day_energy_scenarios_v1.parquet", index=False
    )
    light_result.to_parquet(
        OUT / "lighting_connected_power_scenarios_v1.parquet", index=False
    )

    notes = {
        "status": "EXPERIMENTAL_SCENARIOS_BUILT",
        "empirically_calibrated": False,
        "interpretation": [
            "Survey inputs remain unchanged in their original files.",
            "All wattages are experimental assumptions.",
            "Fan energy applies to the reported usage duration.",
            "Fan energy is not a calendar-day or annual prediction.",
            "Lighting power assumes all reported units of a category are on.",
            "Lighting energy is not calculated without usage duration.",
            "No clock schedules, control actions or RL transitions generated.",
            "These scenarios do not describe measured Guntur households.",
        ],
    }
    (OUT / "scenario_notes_v1.json").write_text(
        json.dumps(notes, indent=2), encoding="utf-8"
    )

    report = {
        "status": notes["status"],
        "scenarios": 3,
        "parameter_rows": len(parameter_rows),
        "fan_scenario_rows": len(fan_result),
        "lighting_scenario_rows": len(light_result),
        "empirically_calibrated": False,
        "rl_transitions_generated": False,
    }
    pd.DataFrame(report.items(), columns=["metric", "value"]).to_csv(
        REPORTS / "fan_lighting_scenarios_v1.csv", index=False
    )
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()