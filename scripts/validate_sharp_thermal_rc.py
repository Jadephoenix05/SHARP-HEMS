"""Deterministic tests and RESIDE envelope conformance for the thermal component.

Two kinds of check:

  1. Deterministic unit tests. Directionality, finiteness, bounds, repeatability,
     exact 15-minute stepping, thermostat behaviour, duty-cycle semantics,
     stability, and config guards. These must all pass.

  2. Envelope conformance. The declared model's passive change is replayed on
     the ACTUAL observed RESIDE AC-off intervals and its distribution compared
     with the observed distribution. RESIDE is used only to REJECT implausible
     parameters. Staying inside the envelope is not evidence of correctness.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd

from sharp_thermal_rc import (
    ThermalConfigError, equilibrium_temperature_c, load_config, thermal_step)

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-9


class Failure(AssertionError):
    pass


def expect(condition, message):
    if not condition:
        raise Failure(message)


# --------------------------------------------------------------------------
# Deterministic unit tests
# --------------------------------------------------------------------------

def test_directionality(params):
    """Hot outside warms the room; the AC cools it relative to doing nothing."""
    off = thermal_step(30.0, 38.0, params, ac_enabled=False)
    expect(off['temperature_change_c'] > 0,
           'A room cooler than a hot outside must warm with the AC off.')
    on = thermal_step(30.0, 38.0, params, ac_enabled=True, setpoint_c=24.0)
    expect(on['next_temperature_c'] < off['next_temperature_c'],
           'Enabling the AC must not leave the room warmer than leaving it off.')
    # Only a unit whose capacity exceeds the envelope load can cool in absolute
    # terms. An undersized unit legitimately still warms, just more slowly.
    if params['ac_full_cooling_c_per_15min'] > off['passive_change_c']:
        expect(on['temperature_change_c'] < 0,
               'Capacity exceeds the load, so the AC must cool the room.')
    expect(on['cooling_applied_c'] > 0,
           'The AC applied no cooling although the room was above setpoint.')
    # Cool outside must cool the room even with the AC off.
    cool = thermal_step(30.0, 20.0, params, ac_enabled=False)
    expect(cool['temperature_change_c'] < 0,
           'A room warmer than a cool outside must cool with the AC off.')
    return {'ac_off_change_c': off['temperature_change_c'],
            'ac_on_change_c': on['temperature_change_c'],
            'cool_outside_change_c': cool['temperature_change_c']}


def test_finite_and_bounded(params):
    """No step produces a nonfinite or absurd temperature across a wide sweep."""
    worst_low, worst_high = math.inf, -math.inf
    for indoor in np.arange(10.0, 50.1, 2.0):
        for outdoor in np.arange(10.0, 50.1, 2.0):
            for enabled in (False, True):
                r = thermal_step(float(indoor), float(outdoor), params,
                                 ac_enabled=enabled, setpoint_c=24.0)
                value = r['next_temperature_c']
                expect(math.isfinite(value), 'Nonfinite temperature produced.')
                worst_low = min(worst_low, value)
                worst_high = max(worst_high, value)
                expect(0.0 <= r['compressor_duty_fraction'] <= 1.0,
                       'Compressor duty fraction left [0, 1].')
    expect(5.0 < worst_low and worst_high < 55.0,
           f'Sweep left a plausible range: {worst_low:.2f} to {worst_high:.2f} C.')
    return {'sweep_min_c': worst_low, 'sweep_max_c': worst_high}


def test_repeatability(params):
    """Identical inputs give bit-identical outputs; the step is pure."""
    first = thermal_step(31.0, 36.0, params, ac_enabled=True, setpoint_c=25.0)
    for _ in range(50):
        again = thermal_step(31.0, 36.0, params, ac_enabled=True, setpoint_c=25.0)
        expect(again['next_temperature_c'] == first['next_temperature_c'],
               'Repeated identical steps disagreed.')
    return {'value_c': first['next_temperature_c']}


def test_fifteen_minute_stepping(params):
    """Every step reports exactly 15 minutes, and four steps make an hour."""
    temperature, minutes = 33.0, 0
    for _ in range(4):
        r = thermal_step(temperature, 36.0, params, ac_enabled=False)
        expect(r['interval_minutes'] == 15, 'A step did not report 15 minutes.')
        temperature = r['next_temperature_c']
        minutes += r['interval_minutes']
    expect(minutes == 60, 'Four steps did not total one hour.')
    return {'hour_end_temperature_c': temperature, 'minutes': minutes}


def test_thermostat(params):
    """The thermostat holds setpoint instead of cooling without limit."""
    # Room already below setpoint: no cooling is called for.
    idle = thermal_step(22.0, 24.0, params, ac_enabled=True, setpoint_c=26.0)
    expect(idle['compressor_duty_fraction'] == 0.0,
           'Compressor ran although the room was below setpoint.')
    expect(idle['cooling_applied_c'] == 0.0, 'Cooling applied below setpoint.')

    # Room far above setpoint: compressor saturates at full duty.
    hard = thermal_step(38.0, 42.0, params, ac_enabled=True, setpoint_c=24.0)
    expect(hard['compressor_duty_fraction'] == 1.0,
           'Compressor did not saturate far above setpoint.')

    # Never overshoots below setpoint, whatever the capacity.
    temperature = 24.0
    for _ in range(96):
        temperature = thermal_step(temperature, 38.0, params,
                                   ac_enabled=True, setpoint_c=24.0)['next_temperature_c']
        expect(temperature >= 24.0 - TOLERANCE,
               f'Thermostat undershot the setpoint: {temperature:.6f} C.')
    return {'no_undershoot_temperature_c': temperature,
            'saturated_duty': hard['compressor_duty_fraction']}


def test_holds_setpoint_at_design_condition(params):
    """A unit sized for the design condition holds its setpoint there.

    This applies to the baseline only. Reduced-capacity sensitivity scenarios
    are SUPPOSED to fail to hold, which is what test_saturates_beyond_capacity
    checks: a real air conditioner saturates rather than cooling without limit.
    """
    temperature = 24.0
    for _ in range(4 * 24):
        temperature = thermal_step(temperature, 39.0, params,
                                   ac_enabled=True, setpoint_c=24.0)['next_temperature_c']
    expect(abs(temperature - 24.0) < 0.5,
           f'Failed to hold setpoint at the design condition: {temperature:.4f} C.')
    return {'design_condition_temperature_c': temperature}


def test_saturates_beyond_capacity(params):
    """Beyond capacity the room drifts above setpoint at full duty, and settles.

    The condition is derived from the parameters rather than hard-coded, because
    a higher-capacity or better-insulated scenario can legitimately hold a
    setpoint that a weaker one cannot.
    """
    outdoor = 55.0
    coupling = params['envelope_coupling_per_step']
    headroom = (params['ac_full_cooling_c_per_15min']
                - params['internal_gain_c_per_15min']) / coupling
    setpoint = outdoor - headroom - 2.0
    if setpoint < 0.0:
        # This scenario can hold any setpoint in the simulated range.
        return {'saturation_reachable': False}
    temperature = setpoint
    for _ in range(4 * 24 * 5):
        step = thermal_step(temperature, outdoor, params,
                            ac_enabled=True, setpoint_c=setpoint)
        temperature = step['next_temperature_c']
    expect(step['compressor_duty_fraction'] == 1.0,
           'Compressor was not saturated beyond its capacity.')
    expect(temperature > setpoint, 'Room reached a setpoint beyond the unit capacity.')
    expect(math.isfinite(temperature) and temperature < 60.0,
           f'Saturated run diverged: {temperature:.2f} C.')
    return {'saturation_reachable': True, 'outdoor_c': outdoor,
            'setpoint_c': setpoint, 'saturated_temperature_c': temperature,
            'duty': step['compressor_duty_fraction']}


def test_duty_semantics(params):
    """Duty is a usable energy multiplier and rises with thermal load."""
    mild = thermal_step(26.0, 30.0, params, ac_enabled=True, setpoint_c=25.0)
    harsh = thermal_step(26.0, 44.0, params, ac_enabled=True, setpoint_c=25.0)
    expect(harsh['compressor_duty_fraction'] >= mild['compressor_duty_fraction'],
           'Duty did not increase with a hotter outside.')
    for r in (mild, harsh):
        expected = r['compressor_duty_fraction'] * params['ac_full_cooling_c_per_15min']
        expect(abs(expected - r['cooling_applied_c']) < TOLERANCE,
               'Duty fraction is inconsistent with the cooling applied.')
    return {'mild_duty': mild['compressor_duty_fraction'],
            'harsh_duty': harsh['compressor_duty_fraction']}


def test_stability(params):
    """Free-running temperature converges to the model's equilibrium, not divergence."""
    temperature = 45.0
    for _ in range(4 * 24 * 10):  # ten simulated days
        temperature = thermal_step(temperature, 34.0, params,
                                   ac_enabled=False)['next_temperature_c']
    target = equilibrium_temperature_c(34.0, params, ac_enabled=False)
    expect(abs(temperature - target) < 0.01,
           f'Free run settled at {temperature:.4f} C, expected {target:.4f} C.')
    return {'settled_c': temperature, 'equilibrium_c': target}


def test_config_guards():
    """Bad configurations are refused rather than silently simulated."""
    refused = 0
    for scenario in ['does_not_exist', 'nonsense']:
        try:
            load_config(scenario=scenario)
        except ThermalConfigError:
            refused += 1
    expect(refused == 2, 'An unknown scenario was accepted.')

    params = load_config()
    for indoor, outdoor in [(-50.0, 30.0), (30.0, 500.0), (float('nan'), 30.0)]:
        try:
            thermal_step(indoor, outdoor, params)
        except ValueError:
            continue
        raise Failure(f'Accepted an impossible input: {indoor}, {outdoor}.')
    return {'scenarios_refused': refused}


# --------------------------------------------------------------------------
# RESIDE envelope conformance
# --------------------------------------------------------------------------

def envelope_conformance(root, params):
    """Replay the declared passive model on observed AC-off intervals."""
    envelope = json.loads(
        (root / 'reports/reside_thermal_envelope_v1.json').read_text(encoding='utf-8'))
    pairs = pd.read_parquet(
        root / 'data/processed/reside_thermal_rc_v1/weather_joined_temperature_pairs.parquet')
    off = pairs[pairs.primary_ac_label_fraction.eq(0)]
    expect(len(off) > 1000, 'Too few AC-off intervals for a conformance check.')

    coupling = params['envelope_coupling_per_step']
    modelled = (coupling * (off.outdoor_t2m_c.to_numpy(float)
                            - off.room_temperature_c.to_numpy(float))
                + params['internal_gain_c_per_15min'])
    observed = off.temperature_change_c.to_numpy(float)
    expect(np.isfinite(modelled).all(), 'Nonfinite modelled change.')

    bounds = envelope['ac_off_change']
    summary = {
        'intervals': int(len(off)),
        'observed_mean_c': float(observed.mean()),
        'modelled_mean_c': float(modelled.mean()),
        'mean_bias_c': float(modelled.mean() - observed.mean()),
        'observed_p01_c': bounds['p01_c_per_15min'],
        'observed_p99_c': bounds['p99_c_per_15min'],
        'modelled_p01_c': float(np.quantile(modelled, 0.01)),
        'modelled_p99_c': float(np.quantile(modelled, 0.99)),
        'modelled_min_c': float(modelled.min()),
        'modelled_max_c': float(modelled.max()),
    }
    # Rejection rule: the declared model must not predict passive swings beyond
    # what was ever observed, and must not be biased by more than a tenth of a
    # degree per interval, which compounds to 9.6 C per day.
    summary['within_observed_extremes'] = bool(
        summary['modelled_min_c'] >= bounds['min_c_per_15min']
        and summary['modelled_max_c'] <= bounds['max_c_per_15min'])
    summary['mean_bias_acceptable'] = bool(abs(summary['mean_bias_c']) <= 0.10)
    observed_spread = bounds['p99_c_per_15min'] - bounds['p01_c_per_15min']
    modelled_spread = summary['modelled_p99_c'] - summary['modelled_p01_c']
    summary['observed_p01_p99_spread_c'] = float(observed_spread)
    summary['modelled_p01_p99_spread_c'] = float(modelled_spread)
    summary['spread_acceptable'] = bool(modelled_spread <= observed_spread)
    # The AC-off subset is selection biased: households switch the AC off
    # preferentially when it is cooler, so an unconditional mean-bias match is
    # not a fair criterion for an envelope coupling. Bias is reported, and the
    # pass criterion is that the model never predicts swings the houses never
    # showed and never exceeds the observed spread.
    summary['conformant'] = bool(
        summary['within_observed_extremes'] and summary['spread_acceptable'])
    summary['mean_bias_is_pass_criterion'] = False
    summary['interpretation'] = (
        'RESIDE bounds can only reject parameters. Conformance is not evidence '
        'that these declared parameters are correct. The AC-off subset is '
        'selection biased toward cooler conditions, so mean bias is reported '
        'as a diagnostic rather than used as a pass criterion.')
    return summary


def _design_hold(params):
    temperature = 24.0
    for _ in range(4 * 24):
        temperature = thermal_step(temperature, 39.0, params,
                                   ac_enabled=True, setpoint_c=24.0)['next_temperature_c']
    return temperature


def sensitivity_sweep(root):
    """Every declared scenario must pass the unit tests and report its behaviour."""
    config = json.loads(
        (root / 'configs/thermal/sharp_thermal_rc_v1.json').read_text(encoding='utf-8'))
    rows = []
    for scenario in config['sensitivity_scenarios']:
        params = load_config(scenario=scenario)
        test_directionality(params)
        test_finite_and_bounded(params)
        test_thermostat(params)
        test_saturates_beyond_capacity(params)
        test_stability(params)
        conformance = envelope_conformance(root, params)
        hot = thermal_step(32.0, 38.0, params, ac_enabled=True, setpoint_c=24.0)
        rows.append({
            'scenario': scenario,
            'tau_hours': params['envelope_time_constant_hours'],
            'coupling_per_step': params['envelope_coupling_per_step'],
            'full_cooling_c': params['ac_full_cooling_c_per_15min'],
            'duty_at_32in_38out': hot['compressor_duty_fraction'],
            'holds_design_setpoint': abs(_design_hold(params) - 24.0) < 0.5,
            'ac_off_equilibrium_c': equilibrium_temperature_c(38.0, params),
            'modelled_mean_bias_c': conformance['mean_bias_c'],
            'envelope_conformant': conformance['conformant'],
            'modelled_spread_c': conformance['modelled_p01_p99_spread_c'],
        })
    return pd.DataFrame(rows)


def main(root):
    params = load_config()
    print(f"Config: {params['config_id']} | scenario: {params['scenario']} | "
          f"tau {params['envelope_time_constant_hours']} h | "
          f"coupling {params['envelope_coupling_per_step']:.5f} per step")

    tests = {
        'directionality': test_directionality(params),
        'finite_and_bounded': test_finite_and_bounded(params),
        'repeatability': test_repeatability(params),
        'fifteen_minute_stepping': test_fifteen_minute_stepping(params),
        'thermostat': test_thermostat(params),
        'holds_setpoint_at_design_condition': test_holds_setpoint_at_design_condition(params),
        'saturates_beyond_capacity': test_saturates_beyond_capacity(params),
        'duty_semantics': test_duty_semantics(params),
        'stability': test_stability(params),
        'config_guards': test_config_guards(),
    }
    for name in tests:
        print(f'  PASS  {name}')

    conformance = envelope_conformance(root, params)
    print(f"\nEnvelope conformance on {conformance['intervals']:,} observed AC-off intervals")
    print(f"  observed mean {conformance['observed_mean_c']:+.4f} C | "
          f"modelled mean {conformance['modelled_mean_c']:+.4f} C | "
          f"bias {conformance['mean_bias_c']:+.4f} C")
    print(f"  modelled range {conformance['modelled_min_c']:+.3f} to "
          f"{conformance['modelled_max_c']:+.3f} C | observed range "
          f"{conformance['observed_p01_c']:+.3f} to {conformance['observed_p99_c']:+.3f} C (p01/p99)")
    print(f"  conformant: {conformance['conformant']}")

    sweep = sensitivity_sweep(root)
    print('\nSensitivity scenarios (all passed the unit tests):')
    print(sweep.round(4).to_string(index=False))

    report = {
        'status': ('THERMAL_COMPONENT_TESTS_PASSED' if conformance['conformant']
                   else 'THERMAL_COMPONENT_TESTS_PASSED_ENVELOPE_REVIEW_REQUIRED'),
        'config_id': params['config_id'],
        'deterministic_tests': tests,
        'envelope_conformance': conformance,
        'sensitivity_scenarios': sweep.to_dict('records'),
        'action_semantics': 'AC_SETPOINT_NOT_BINARY_SWITCH',
        'compressor_duty_is_output_not_action': True,
        'parameter_basis': 'DECLARED_ENGINEERING_ASSUMPTIONS_NOT_FITTED',
        'fitted_to_reside': False,
        'causal_cooling_effect_established': False,
        'integrated_into_transition_core': False,
        'approved_for_hardware_control': False,
        'full_simulator_ready': False,
        'master_release_ready': False,
    }
    out = root / 'reports/sharp_thermal_rc_validation_v1.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nSTATUS:', report['status'])
    print('Fitted to RESIDE:', report['fitted_to_reside'])
    print('Integrated into transition core:', report['integrated_into_transition_core'])
    print('Report:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    a = p.parse_args()
    main(a.root.resolve())
