"""Independent audit of the generated RL transition layer.

Deliberately re-derives its checks from the stored table rather than trusting
the generator's own report. Anything the generator asserted is recomputed here
from the data, and several properties the generator never checked are tested:
episode chaining, action legality, duty/AC consistency, billing monotonicity,
key uniqueness, behaviour diversity and per-split feature drift.

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


def audit(root):
    out = root / 'data/processed/sharp_rl_transitions_v1'
    d = pd.read_parquet(out / 'rl_transitions.parquet')
    # Sort ONCE, before any array is derived. Deriving arrays and then sorting
    # the frame silently misaligns every positional comparison below.
    d = d.sort_values(['episode_id', 'step_id']).reset_index(drop=True)
    schema = json.loads((out / 'feature_schema.json').read_text(encoding='utf-8'))
    width = schema['feature_count']
    n_global = len(schema['global_features'])
    n_device = len(schema['device_features'])
    slots = schema['maximum_device_slots']

    print(f'Auditing {len(d):,} transitions\n')

    print('Keys and structure')
    check(not d.duplicated(['episode_id', 'step_id']).any(),
          'no duplicate (episode_id, step_id)')
    sizes = d.groupby('episode_id').size()
    check(bool(sizes.eq(96).all()), f'all {len(sizes):,} episodes have exactly 96 steps')
    steps = d.groupby('episode_id').step_id.agg(['min', 'max'])
    check(bool(steps['min'].eq(0).all() and steps['max'].eq(95).all()),
          'every episode runs step 0 to 95')
    check(n_global + slots * n_device == width,
          f'schema widths are self-consistent ({n_global} + {slots}x{n_device} = {width})')

    print('\nFinite values and shapes')
    states = np.stack(d.state.to_numpy()).astype(float)
    next_states = np.stack(d.next_state.to_numpy()).astype(float)
    check(states.shape == (len(d), width), f'state matrix is {len(d)}x{width}')
    check(np.isfinite(states).all(), 'no nonfinite value in any state')
    check(np.isfinite(next_states).all(), 'no nonfinite value in any next_state')
    check(np.isfinite(d.reward.to_numpy(float)).all(), 'no nonfinite reward')
    check(not d.reward.isna().any(), 'no missing reward')

    print('\nEpisode chaining (recomputed, not trusted)')
    d = d.sort_values(['episode_id', 'step_id']).reset_index(drop=True)
    same_episode = d.episode_id.to_numpy()[:-1] == d.episode_id.to_numpy()[1:]
    chain_ok = np.all(np.isclose(next_states[:-1][same_episode],
                                 states[1:][same_episode], atol=1e-9))
    check(bool(chain_ok), 'next_state equals the following state within every episode')
    temp_ok = np.allclose(d.next_indoor_temperature_c.to_numpy(float)[:-1][same_episode],
                          d.indoor_temperature_c.to_numpy(float)[1:][same_episode],
                          atol=1e-6)
    check(bool(temp_ok), 'indoor temperature chains across steps within every episode')

    print('\nTermination')
    check(bool(d.groupby('episode_id').done.sum().eq(1).all()),
          'exactly one done flag per episode')
    check(bool(d.loc[d.step_id.eq(95), 'done'].all()), 'step 95 is always done')
    check(not bool(d.loc[d.step_id.lt(95), 'done'].any()), 'no early done flag')

    print('\nActions')
    present = np.stack(d.device_present.to_numpy()).astype(bool)
    action_len = d.action.map(len).to_numpy()
    device_count = present.sum(1)
    check(bool((action_len == device_count).all()),
          'action length always equals the present-device count')
    flat = np.concatenate(d.action.to_numpy())
    check(bool(np.isin(flat, [0, 1]).all()), 'every action is binary')
    requested = np.concatenate(d.requested_action.to_numpy())
    check(bool(np.isin(requested, [0, 1]).all()), 'every requested action is binary')
    shield_changed = int((flat != requested).sum())
    print(f'        shield altered {shield_changed:,} of {len(flat):,} device decisions '
          f'({100 * shield_changed / len(flat):.2f}%)')

    print('\nThermal and AC consistency')
    duty = d.compressor_duty_fraction.to_numpy(float)
    check(bool(((duty >= 0) & (duty <= 1)).all()), 'compressor duty stays within [0, 1]')
    has_ac_feature = states[:, schema['global_features'].index(
        'household_has_air_conditioner')].astype(bool)
    check(not bool((duty > 0)[~has_ac_feature].any()),
          'compressor never runs in a household without an air conditioner')
    lo = float(d.indoor_temperature_c.min())
    hi = float(d.next_indoor_temperature_c.max())
    check(5.0 < lo and hi < 55.0, f'indoor temperature plausible ({lo:.2f} to {hi:.2f} C)')
    outdoor = d.outdoor_temperature_c.to_numpy(float)
    check(bool(((outdoor > 0) & (outdoor < 55)).all()),
          f'outdoor temperature plausible ({outdoor.min():.2f} to {outdoor.max():.2f} C)')
    warn(has_ac_feature.mean() > 0.05,
         f'only {100 * has_ac_feature.mean():.1f}% of transitions are AC households, '
         'so the comfort signal is sparse (this reflects real IRES ownership)')

    print('\nEnergy and billing')
    check(bool((d.grid_import_kwh.to_numpy(float) >= 0).all()), 'grid import never negative')
    kwh_feature = schema['global_features'].index('billing_period_kwh')
    monotonic = True
    for _, g in d.groupby('episode_id'):
        series = np.stack(g.state.to_numpy()).astype(float)[:, kwh_feature]
        if np.any(np.diff(series) < -1e-9):
            monotonic = False
            break
    check(monotonic, 'billing-period kWh is nondecreasing within every episode')

    print('\nSplits and leakage')
    check(int(d.groupby('household_id').split.nunique().max()) == 1,
          'no household appears in more than one split')
    check(int(d.groupby('date').split.nunique().max()) == 1,
          'no context date appears in more than one split')
    check(int(d.groupby('episode_id').split.nunique().max()) == 1,
          'no episode spans two splits')
    check(set(d.split) == {'train', 'validation', 'test'}, 'all three splits present')

    print('\nBehaviour diversity')
    policies = sorted(d.policy.unique())
    check(len(policies) >= 2, f'more than one behaviour policy present: {policies}')
    for split in ['train', 'validation', 'test']:
        sub = d[d.split.eq(split)]
        check(sub.policy.nunique() == len(policies),
              f'{split} contains every behaviour policy')
    on_rate = float(flat.mean())
    check(0.02 < on_rate < 0.98,
          f'actions are not degenerate (device-on rate {on_rate:.3f})')

    print('\nReward')
    reward = d.reward.to_numpy(float)
    check(bool((reward <= 0).all()),
          'reward is a nonpositive penalty sum, as the reward module defines it')
    by_policy = d.groupby('policy').reward.mean()
    print('        mean reward by policy:',
          {k: round(v, 4) for k, v in by_policy.items()})
    warn(by_policy.nunique() > 1, 'behaviour policies produce identical mean reward')

    print('\nPer-split feature drift (train versus test)')
    train_mean = states[d.split.eq('train').to_numpy()].mean(0)
    test_mean = states[d.split.eq('test').to_numpy()].mean(0)
    spread = states.std(0)
    safe = spread > 1e-8
    drift = np.abs(train_mean - test_mean)[safe] / spread[safe]
    worst = float(drift.max())
    print(f'        largest standardised mean shift: {worst:.3f} sd')
    warn(worst < 3.0, f'a feature shifts {worst:.2f} sd between train and test')

    report = {
        'status': 'AUDIT_PASSED' if not FAILURES else 'AUDIT_FAILED',
        'transitions': int(len(d)),
        'episodes': int(d.episode_id.nunique()),
        'households': int(d.household_id.nunique()),
        'feature_count': int(width),
        'checks_failed': FAILURES,
        'warnings': WARNINGS,
        'shield_altered_device_decisions': shield_changed,
        'shield_altered_fraction': shield_changed / len(flat),
        'device_on_rate': on_rate,
        'ac_household_transition_fraction': float(has_ac_feature.mean()),
        'mean_reward_by_policy': {k: float(v) for k, v in by_policy.items()},
        'largest_train_test_standardised_shift_sd': worst,
        'audit_is_independent_of_generator_report': True,
    }
    (root / 'reports/sharp_rl_transitions_audit_v1.json').write_text(
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
    a = p.parse_args()
    audit(a.root.resolve())
