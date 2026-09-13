"""Verify a SHARP Master Dataset release using ONLY the released files.

Run this after downloading the Kaggle copy, from inside the release folder's
parent, to prove the release is complete, unmodified and internally consistent
without touching the original working tree.

    python verify_sharp_release_v1.py --release release/SHARP_MASTER_V1

Checks: every manifest file present, every SHA-256 matching, the transition
table loadable, feature widths consistent with the schema, splits disjoint by
household AND by date, rewards and states finite, temperatures plausible, and
the declared honesty flags still present.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

FAILURES = []


def check(condition, message):
    if condition:
        print(f'  PASS  {message}')
    else:
        print(f'  FAIL  {message}')
        FAILURES.append(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(release):
    manifest_path = release / 'MANIFEST.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'No MANIFEST.json in {release}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    print(f"SHARP release {manifest['release_version']} at {release}\n")

    print('Integrity')
    missing = modified = 0
    for item in manifest['manifest']:
        path = release / item['path']
        if not path.exists():
            missing += 1
        elif sha256(path) != item['sha256']:
            modified += 1
    check(missing == 0, f"all {len(manifest['manifest'])} manifest files present")
    check(modified == 0, 'no file differs from its recorded SHA-256')

    print('\nTransition table')
    transitions = pd.read_parquet(release / 'rl_transitions/rl_transitions.parquet')
    schema = json.loads(
        (release / 'rl_transitions/feature_schema.json').read_text(encoding='utf-8'))
    report = json.loads(
        (release / 'rl_transitions/transition_validation.json').read_text(encoding='utf-8'))

    check(len(transitions) == report['transitions'],
          f"row count matches the report ({len(transitions):,})")
    width = schema['feature_count']
    check(transitions.state.map(len).eq(width).all(),
          f'every state vector has {width} features')
    check(transitions.next_state.map(len).eq(width).all(),
          f'every next_state vector has {width} features')

    states = np.stack(transitions.state.to_numpy()).astype(float)
    next_states = np.stack(transitions.next_state.to_numpy()).astype(float)
    check(np.isfinite(states).all() and np.isfinite(next_states).all(),
          'no nonfinite values in any observation')
    check(np.isfinite(transitions.reward.to_numpy(float)).all(),
          'no nonfinite rewards')

    print('\nLeakage gates')
    by_household = transitions.groupby('household_id').split.nunique()
    check(int(by_household.max()) == 1,
          'no household appears in more than one split')
    by_date = transitions.groupby('date').split.nunique()
    check(int(by_date.max()) == 1,
          'no context date appears in more than one split')
    for split in ['train', 'validation', 'test']:
        check(split in set(transitions.split), f'{split} split is present')

    print('\nPhysical plausibility')
    low = float(transitions.indoor_temperature_c.min())
    high = float(transitions.next_indoor_temperature_c.max())
    check(5.0 < low and high < 55.0,
          f'indoor temperature stays plausible ({low:.1f} to {high:.1f} C)')
    duty = transitions.compressor_duty_fraction.to_numpy(float)
    check(bool(((duty >= 0) & (duty <= 1)).all()),
          'compressor duty stays within [0, 1]')
    check(bool((transitions.grid_import_kwh.to_numpy(float) >= 0).all()),
          'grid import is never negative')

    print('\nEpisode structure')
    steps = transitions.groupby('episode_id').size()
    check(bool(steps.eq(96).all()), f'all {len(steps):,} episodes have 96 steps')
    terminal = transitions.groupby('episode_id').done.sum()
    check(bool(terminal.eq(1).all()), 'each episode ends with exactly one done flag')

    print('\nDeclared honesty flags')
    check(report['billing_reconciliation'] == 'PASS', 'billing reconciliation recorded PASS')
    check(report['state_continuity'] == 'PASS', 'state continuity recorded PASS')
    thermal = json.loads(
        (release / 'configs/sharp_thermal_rc_v1.json').read_text(encoding='utf-8'))
    check(thermal['fitted_to_reside'] is False,
          'thermal config still declares it is not fitted to RESIDE')
    check(thermal['causal_cooling_effect_established'] is False,
          'thermal config still declares no causal cooling effect')
    check(len(report['scope_and_limits']) >= 8,
          'the release still carries its full limitations list')

    print()
    if FAILURES:
        print(f'RELEASE VERIFICATION FAILED: {len(FAILURES)} check(s)')
        for item in FAILURES:
            print('  -', item)
        raise SystemExit(1)
    print('RELEASE VERIFICATION PASSED')
    print(f"  transitions: {report['transitions']:,}")
    print(f"  households:  {report['households']}")
    print(f"  episodes:    {report['episodes']:,}")
    print(f"  splits:      {report['transitions_by_split']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'release/SHARP_MASTER_V2')
    a = p.parse_args()
    verify(a.release.resolve())
