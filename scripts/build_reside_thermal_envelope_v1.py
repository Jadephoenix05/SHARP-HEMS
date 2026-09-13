"""Observed plausibility envelope for indoor temperature dynamics.

This is the ONLY role RESIDE-AC plays in the SHARP simulator thermal component.
It does not supply a cooling coefficient, a time constant or any fitted
parameter. It supplies observed bounds that a declared-assumption model must
not contradict: how fast indoor temperature actually moved in 15 minutes, and
what indoor temperatures actually occurred.

Rejecting a parameter set that leaves this envelope is sound. Claiming a
parameter set is correct because it stays inside it is not.

Eleven Hyderabad houses, nineteen May 2019 days. Not a national distribution.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def check(condition, message):
    if not condition:
        raise ValueError(message)


def describe(change):
    values = change.to_numpy(float)
    check(np.isfinite(values).all(), 'Nonfinite temperature changes')
    return {
        'pairs': int(len(values)),
        'mean_c_per_15min': float(values.mean()),
        'sd_c_per_15min': float(values.std(ddof=1)),
        'p01_c_per_15min': float(np.quantile(values, 0.01)),
        'p50_c_per_15min': float(np.median(values)),
        'p99_c_per_15min': float(np.quantile(values, 0.99)),
        'min_c_per_15min': float(values.min()),
        'max_c_per_15min': float(values.max()),
    }


def build(root):
    source = (root / 'data/processed/reside_thermal_inputs_v1'
              / 'observed_temperature_response_pairs.parquet')
    check(source.exists(), f'Missing {source}')
    d = pd.read_parquet(source)
    check(d.candidate_house_id.nunique() == 11, 'Expected eleven houses')

    off = d[d.primary_ac_label_fraction.eq(0)]
    on = d[d.primary_ac_label_fraction.eq(1)]
    check(len(off) > 0 and len(on) > 0, 'Missing AC-off or AC-on pairs')

    indoor = d.room_temperature_c.to_numpy(float)
    envelope = {
        'status': 'RESIDE_THERMAL_ENVELOPE_BUILT',
        'role': 'PLAUSIBILITY_BOUNDS_ONLY_NOT_PARAMETER_SOURCE',
        'houses': int(d.candidate_house_id.nunique()),
        'total_pairs': int(len(d)),
        'observation_year_supported': 2019,
        'geography': 'Hyderabad India',
        'step_minutes': 15,
        'indoor_temperature_c': {
            'min': float(indoor.min()),
            'p01': float(np.quantile(indoor, 0.01)),
            'median': float(np.median(indoor)),
            'p99': float(np.quantile(indoor, 0.99)),
            'max': float(indoor.max()),
        },
        'ac_off_change': describe(off.temperature_change_c),
        'ac_on_change': describe(on.temperature_change_c),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'interpretation_warnings': [
            'AC labels were assigned using current AND temperature change, so the '
            'AC-on distribution is not an independent measurement of cooling.',
            'An AC-on bin includes thermostat cycling, so a near-zero change means '
            'the room was being held at setpoint, not that cooling was weak.',
            'Eleven houses in one city over nineteen summer days; not a population.',
            'These are bounds for rejecting implausible parameters, not evidence '
            'that any parameter set inside them is correct.',
        ],
        'causal_cooling_effect_established': False,
        'full_simulator_ready': False,
        'master_release_ready': False,
    }

    out = root / 'reports/reside_thermal_envelope_v1.json'
    out.write_text(json.dumps(envelope, indent=2), encoding='utf-8')
    print('RESIDE THERMAL ENVELOPE BUILT')
    print('Houses:', envelope['houses'], '| pairs:', envelope['total_pairs'])
    for name in ['ac_off_change', 'ac_on_change']:
        e = envelope[name]
        print(f"{name:14s} n={e['pairs']:6d} mean={e['mean_c_per_15min']:+.4f} "
              f"p01={e['p01_c_per_15min']:+.3f} p99={e['p99_c_per_15min']:+.3f}")
    i = envelope['indoor_temperature_c']
    print(f"indoor C: min={i['min']:.1f} median={i['median']:.1f} max={i['max']:.1f}")
    print('Role:', envelope['role'])
    print('Output:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
