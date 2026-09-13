"""Experimental RESIDE temperature dynamics; not validated cooling physics."""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def temperature_step(
    model,
    temperature_c,
    ac_on_fraction,
    minute_of_day,
    *,
    allow_associational_scenario=False,
):
    """
    Advance temperature by exactly 15 minutes.

    ac_on_fraction is simulated compressor operation during this interval,
    not a thermostat setpoint or a fraction of electrical power.
    """
    if not allow_associational_scenario:
        raise ValueError(
            "Explicit associational-scenario opt-in is required."
        )

    values = [temperature_c, ac_on_fraction, minute_of_day]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Inputs must be finite.")
    if not 0 <= ac_on_fraction <= 1:
        raise ValueError("AC fraction must be between 0 and 1.")
    if not 0 <= minute_of_day < 1440:
        raise ValueError("Minute of day must be between 0 and 1439.")

    c = model["coefficients"]
    names = [
        "intercept",
        "temperature_t_c",
        "ac_label_fraction",
        "sin_clock",
        "cos_clock",
    ]
    if not all(math.isfinite(float(c[k])) for k in names):
        raise ValueError("Model coefficients must be finite.")
    if not 0 <= c["temperature_t_c"] < 1:
        raise ValueError("Temperature persistence coefficient is unsupported.")
    if c["ac_label_fraction"] >= 0:
        raise ValueError("Model lacks a negative AC association.")

    angle = 2 * math.pi * minute_of_day / 1440
    next_temperature = (
        c["intercept"]
        + c["temperature_t_c"] * temperature_c
        + c["ac_label_fraction"] * ac_on_fraction
        + c["sin_clock"] * math.sin(angle)
        + c["cos_clock"] * math.cos(angle)
    )

    if not math.isfinite(next_temperature):
        raise ValueError("Nonfinite predicted temperature.")

    # Do not clip: clipping could hide unstable or unsuitable dynamics.
    return {
        "next_temperature_c": next_temperature,
        "temperature_change_c": next_temperature - temperature_c,
        "interval_minutes": 15,
        "model_basis": "EXPLICIT_ASSOCIATIONAL_SIMULATION_SCENARIO",
        "causal_cooling_effect_established": False,
        "outdoor_weather_integrated": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--house", type=int, required=True)
    parser.add_argument("--allow-associational-scenario", action="store_true")
    args = parser.parse_args()

    path = (
        ROOT / "data/processed/reside_temperature_baseline_v1"
        / f"house_{args.house:02d}_model.json"
    )
    model = json.loads(path.read_text(encoding="utf-8"))

    results = {}
    for name, fraction in [("ac_off", 0.0), ("ac_on", 1.0)]:
        results[name] = temperature_step(
            model,
            temperature_c=30.0,
            ac_on_fraction=fraction,
            minute_of_day=720,
            allow_associational_scenario=args.allow_associational_scenario,
        )

    difference = (
        results["ac_on"]["next_temperature_c"]
        - results["ac_off"]["next_temperature_c"]
    )
    if not math.isclose(
        difference,
        model["coefficients"]["ac_label_fraction"],
        abs_tol=1e-10,
    ):
        raise AssertionError("AC coefficient arithmetic failed.")

    print(json.dumps({
        "status": "EXPERIMENTAL_THERMAL_STEP_CHECK_PASSED",
        "house": args.house,
        "initial_temperature_c": 30.0,
        "clock": "12:00 IST",
        "results": results,
        "on_minus_off_temperature_c": difference,
        "validation_scope": "arithmetic_only",
        "household_assignment_made": False,
        "full_simulator_ready": False,
        "master_release_ready": False,
    }, indent=2))


if __name__ == "__main__":
    main()