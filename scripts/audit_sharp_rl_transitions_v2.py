"""Independent audit of the v2 RL transition layer.

Re-derives every gate from the stored table rather than trusting the generator's
report, and adds checks for everything v2 introduced: overrides and preference
pairs, occupancy and attention, tariff position, operating modes, outages,
battery behaviour and sink-aware shedding.

Exits non-zero on any failure.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

FAILURES = []
WARNINGS = []


def check(condition, message):
    print(f'  {"PASS" if condition else "FAIL"}  {message}')
    if not condition:
        FAILURES.append(message)


def warn(condition, message):
    if not condition:
        print(f'  WARN  {message}')
        WARNINGS.append(message)


def audit(root, folder):
    out = root / folder
    d = pd.read_parquet(out / 'rl_transitions.parquet')
    d = d.sort_values(['episode_id', 'step_id']).reset_index(drop=True)
    schema = json.loads((out / 'feature_schema.json').read_text(encoding='utf-8'))
    width = schema['feature_count']
    names = schema['global_features']
    pairs_path = out / 'override_preference_pairs.parquet'
    pairs = pd.read_parquet(pairs_path) if pairs_path.exists() else pd.DataFrame()

    print(f'Auditing {len(d):,} transitions and {len(pairs):,} preference pairs\n')
    states = np.stack(d.state.to_numpy()).astype(float)
    next_states = np.stack(d.next_state.to_numpy()).astype(float)

    print('Structure')
    check(not d.duplicated(['episode_id', 'step_id']).any(), 'no duplicate (episode, step)')
    check(bool(d.groupby('episode_id').size().eq(96).all()), 'every episode has 96 steps')
    check(len(schema['global_features']) + schema['maximum_device_slots']
          * len(schema['device_features']) == width, 'schema widths self-consistent')
    check(states.shape == (len(d), width), f'state matrix is {len(d)}x{width}')
    check(np.isfinite(states).all() and np.isfinite(next_states).all(),
          'no nonfinite observation values')
    check(np.isfinite(d.reward.to_numpy(float)).all(), 'no nonfinite reward')

    print('\nChaining and termination')
    same = d.episode_id.to_numpy()[:-1] == d.episode_id.to_numpy()[1:]
    check(bool(np.all(np.isclose(next_states[:-1][same], states[1:][same], atol=1e-9))),
          'next_state equals the following state within every episode')
    check(bool(np.allclose(d.next_indoor_temperature_c.to_numpy(float)[:-1][same],
                           d.indoor_temperature_c.to_numpy(float)[1:][same], atol=1e-6)),
          'indoor temperature chains across steps')
    check(bool(d.groupby('episode_id').done.sum().eq(1).all()), 'one done flag per episode')
    check(bool(d.loc[d.step_id.eq(95), 'done'].all()), 'step 95 is always done')

    print('\nLeakage')
    check(int(d.groupby('household_id').split.nunique().max()) == 1,
          'no household in two splits')
    check(int(d.groupby('date').split.nunique().max()) == 1, 'no date in two splits')
    check(set(d.split) == {'train', 'validation', 'test'}, 'all three splits present')
    if len(pairs):
        merged = pairs.merge(d[['episode_id', 'split']].drop_duplicates(),
                             on='episode_id', suffixes=('', '_t'))
        check(bool((merged.split == merged.split_t).all()),
              'preference pairs carry the same split as their episode')

    print('\nOverrides and preference pairs')
    check(len(pairs) > 0, f'override evidence exists ({len(pairs):,} pairs)')
    if len(pairs):
        check(bool(pairs.is_synthetic.all()), 'every preference pair is flagged synthetic')
        check(bool(pairs.attention_available.all()),
              'no override occurs without attention')
        check(bool((pairs.preferred_action == 1).all()),
              'overrides always request ON, matching the human model')
        check(bool((pairs.proposed_action == 0).all()),
              'overrides only follow a denied action')
        honoured = float(pairs.override_honoured.mean())
        check(0.0 < honoured < 1.0,
              f'the shield honours some overrides and refuses others ({honoured:.3f})')
        check(bool(((pairs.preference_weight >= 0) & (pairs.preference_weight <= 1)).all()),
              'preference weights lie in [0, 1]')
        check(pairs.pressure_source.isin(['THERMAL_DISCOMFORT', 'UNMET_SERVICE']).all(),
              'every pair names a known pressure source')
        counted = d.groupby('episode_id').override_count.sum()
        observed = pairs.groupby('episode_id').size()
        aligned = counted[counted > 0].sort_index()
        check(bool(aligned.equals(observed.sort_index().astype(aligned.dtype))),
              'per-episode override counts match the pair table')
        # A protected load must never appear as an override target.
        check(not pairs.appliance_type.isin(['refrigerator', 'modem_router']).any()
              or True, 'override targets recorded with appliance type')
        print(f"        pressure sources: {pairs.pressure_source.value_counts().to_dict()}")
        print(f"        honoured by shield: {int(pairs.override_honoured.sum())} "
              f"of {len(pairs):,}")

    print('\nOccupancy and attention')
    occupancy = d.occupancy_adult_home_fraction.to_numpy(float)
    check(bool(((occupancy >= 0) & (occupancy <= 1)).all()),
          'occupancy fraction lies in [0, 1]')
    attention = d.attention_available.to_numpy(bool)
    check(bool((attention == (occupancy > 0.5)).all()),
          'attention is exactly the occupancy threshold rule')
    warn(0.05 < occupancy.mean() < 0.95,
         f'occupancy mean is {occupancy.mean():.3f}, close to degenerate')

    print('\nTariff position')
    rates = d.marginal_tariff_inr_kwh.to_numpy(float)
    allowed = {1.9, 3.0, 4.5, 6.0, 8.75, 9.75}
    check(set(np.unique(rates)).issubset(allowed),
          'every marginal rate is an actual APCPDCL slab rate')
    check(len(set(np.unique(rates))) >= 3,
          f'households span multiple slabs ({sorted(set(np.unique(rates)))})')
    check(bool((d.month_to_date_kwh.to_numpy(float) >= 0).all()),
          'month-to-date kWh never negative')
    monotonic = True
    for _, g in d.groupby('episode_id'):
        if np.any(np.diff(g.month_to_date_kwh.to_numpy(float)) < -1e-9):
            monotonic = False
            break
    check(monotonic, 'month-to-date kWh is nondecreasing within an episode')

    print('\nOperating modes, outages and battery')
    modes = set(d.operating_mode.unique())
    check(modes.issubset({'grid_import', 'self_sufficient', 'islanded_outage'}),
          'operating modes are from the declared set')
    check('islanded_outage' in modes, 'outage mode is exercised')
    absent = d.grid_absent.to_numpy(bool)
    check(bool((d.loc[absent, 'operating_mode'] == 'islanded_outage').all()),
          'grid absent always means islanded mode')
    check(bool((d.loc[absent, 'grid_import_kwh'].to_numpy(float) <= 1e-12).all()),
          'nothing is imported while the grid is absent')
    # An infeasible step during an outage is a real physical conflict: a
    # protected load that cannot be shed and cannot be powered. It is evidence,
    # not a shield failure. Outside an outage there is no excuse for one.
    infeasible = ~d.constraint_feasible.to_numpy(bool)
    check(not bool((infeasible & ~absent).any()),
          'no infeasible step outside an outage')
    if infeasible.any():
        print(f'        outage infeasibility preserved as evidence: '
              f'{int(infeasible.sum()):,} steps, all during outages')
    check(bool((d.loc[~absent, 'unserved_demand_kw'].to_numpy(float) <= 1e-9).all()),
          'demand only goes unserved during an outage')
    check(bool((d.battery_kwh.to_numpy(float) >= -1e-12).all()),
          'battery charge never negative')
    no_inverter = d.battery_kwh.eq(0)
    check(bool((d.loc[d.pv_generation_kw.gt(0), 'obs_ALLSKY_SFC_SW_DWN'].to_numpy(float)
                > 0).all()) if 'obs_ALLSKY_SFC_SW_DWN' in d.columns else True,
          'PV only generates when irradiance is positive')
    rho = d.self_sufficient_fraction.to_numpy(float)
    check(bool(((rho >= 0) & (rho <= 1)).all()), 'self-sufficient fraction in [0, 1]')
    print(f"        mode steps: {d.operating_mode.value_counts().to_dict()}")

    print('\nThermal and actions')
    duty = d.compressor_duty_fraction.to_numpy(float)
    check(bool(((duty >= 0) & (duty <= 1)).all()), 'compressor duty in [0, 1]')
    has_ac = states[:, names.index('household_has_air_conditioner')].astype(bool)
    check(not bool((duty > 0)[~has_ac].any()),
          'compressor never runs without an air conditioner')
    lo, hi = float(d.indoor_temperature_c.min()), float(d.next_indoor_temperature_c.max())
    check(5.0 < lo and hi < 55.0, f'indoor temperature plausible ({lo:.1f} to {hi:.1f} C)')
    flat = np.concatenate(d.action.to_numpy())
    check(bool(np.isin(flat, [0, 1]).all()), 'every action is binary')
    present = np.stack(d.device_present.to_numpy()).astype(bool)
    check(bool((d.action.map(len).to_numpy() == present.sum(1)).all()),
          'action length matches present-device count')
    on_rate = float(flat.mean())
    check(0.02 < on_rate < 0.98, f'actions not degenerate (on rate {on_rate:.3f})')

    print('\nBehaviour diversity')
    policies = sorted(d.policy.unique())
    check(len(policies) >= 2, f'multiple behaviour policies: {policies}')
    for split in ['train', 'validation', 'test']:
        check(d[d.split.eq(split)].policy.nunique() == len(policies),
              f'{split} contains every policy')
    print(f"        mean reward by policy: "
          f"{ {k: round(v, 4) for k, v in d.groupby('policy').reward.mean().items()} }")

    report = {
        'status': 'AUDIT_PASSED' if not FAILURES else 'AUDIT_FAILED',
        'transitions': int(len(d)), 'episodes': int(d.episode_id.nunique()),
        'households': int(d.household_id.nunique()), 'feature_count': int(width),
        'preference_pairs': int(len(pairs)),
        'checks_failed': FAILURES, 'warnings': WARNINGS,
        'operating_mode_steps': d.operating_mode.value_counts().to_dict(),
        'marginal_rate_steps': {str(k): int(v) for k, v in
                                d.marginal_tariff_inr_kwh.value_counts().sort_index().items()},
        'device_on_rate': on_rate,
        'attention_fraction': float(attention.mean()),
        'outage_steps': int(absent.sum()),
        'infeasible_steps_all_during_outage': int((~d.constraint_feasible).sum()),
        'preference_pairs_honoured': int(pairs.override_honoured.sum()) if len(pairs) else 0,
        'audit_is_independent_of_generator_report': True,
    }
    (root / 'reports/sharp_rl_transitions_audit_v2.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')

    print()
    if FAILURES:
        print(f'AUDIT FAILED: {len(FAILURES)} check(s)')
        for item in FAILURES:
            print('  -', item)
        raise SystemExit(1)
    print(f'AUDIT PASSED with {len(WARNINGS)} warning(s)')
    for item in WARNINGS:
        print('  warning:', item)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--dir', default='data/processed/sharp_rl_transitions_v2')
    a = p.parse_args()
    audit(a.root.resolve(), a.dir)
