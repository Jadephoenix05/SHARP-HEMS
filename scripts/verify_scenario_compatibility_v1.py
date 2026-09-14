"""Prove the PV scenario can be combined with the main scenario without leaking.

Why this matters. The main dataset has pv_generation_kw identically zero, because
IRES reports almost no rooftop solar in Andhra Pradesh. A policy trained on it
alone has never once seen solar generation vary, so it cannot be trusted in a
home that has panels - it has no evidence about that condition at all. The PV
overlay scenario supplies that evidence.

But stacking two datasets is exactly where a leak gets introduced silently. If a
household sat in train in one scenario and validation in the other, or if a date
crossed a split boundary, the held-out numbers would be quietly inflated and
nothing downstream would notice.

So this checks the three things that would break it, and refuses on any of them.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

SCENARIOS = {
    'main': 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet',
    'pv': 'data/processed/sharp_rl_scenario_pv_v1/rl_transitions.parquet',
}
SPLITS = ['train', 'validation', 'test']


def verify(root):
    frames = {}
    for name, relative in SCENARIOS.items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f'Missing scenario {name}: {path}')
        frames[name] = pd.read_parquet(
            path, columns=['split', 'household_id', 'date', 'episode_id',
                           'state', 'pv_generation_kw', 'battery_kwh',
                           'self_sufficient_fraction', 'operating_mode'])

    failures = []

    widths = {name: len(f.state.iloc[0]) for name, f in frames.items()}
    if len(set(widths.values())) != 1:
        failures.append(f'Feature widths differ: {widths}')

    # 1. A household must sit in the same split in every scenario.
    assignment = {name: f.groupby('household_id').split.first()
                  for name, f in frames.items()}
    base = assignment['main']
    for name, other in assignment.items():
        shared = base.index.intersection(other.index)
        mismatched = int((base[shared] != other[shared]).sum())
        if mismatched:
            failures.append(f'{name}: {mismatched} households change split')

    # 2. Each scenario's dates for a split must already belong to that split.
    for name, f in frames.items():
        for split in SPLITS:
            theirs = set(f.loc[f.split.eq(split), 'date'].unique())
            ours = set(frames['main'].loc[frames['main'].split.eq(split), 'date'].unique())
            if not theirs <= ours:
                failures.append(f'{name}/{split}: dates outside the split: '
                                f'{sorted(theirs - ours)[:3]}')

    # 3. No date may appear in two splits once the scenarios are stacked.
    stacked = pd.concat([f[['split', 'date']] for f in frames.values()])
    by_split = {s: set(stacked.loc[stacked.split.eq(s), 'date'].unique()) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            shared = by_split[a] & by_split[b]
            if shared:
                failures.append(f'{a}/{b} share dates: {sorted(shared)[:3]}')

    # Not a failure, but it must be recorded: the same household-day-policy is
    # simulated in both scenarios, so episode_id is not unique across the union.
    # Anything grouping by episode_id has to carry the scenario too.
    overlap = len(set(frames['main'].episode_id) & set(frames['pv'].episode_id))

    coverage = {}
    for name, f in frames.items():
        coverage[name] = {
            'rows': int(len(f)),
            'households': int(f.household_id.nunique()),
            'steps_with_solar': float((f.pv_generation_kw > 0).mean()),
            'steps_with_battery_charge': float((f.battery_kwh > 0).mean()),
            'mean_self_sufficient_fraction': float(f.self_sufficient_fraction.mean()),
            'operating_modes': {k: int(v) for k, v
                                in f.operating_mode.value_counts().items()},
        }

    report = {
        'status': 'SCENARIOS_COMPATIBLE' if not failures else 'SCENARIOS_INCOMPATIBLE',
        'scenarios': SCENARIOS,
        'failures': failures,
        'coverage': coverage,
        'shared_episode_ids': overlap,
        'shared_episode_id_note': (
            'The same household-day-policy is simulated under both scenarios, so '
            'episode_id alone is not a key across the union. Carry the scenario '
            'name alongside it.'),
        'why_combine': (
            'pv_generation_kw is identically zero in the main scenario. Training '
            'on it alone gives the policy no evidence about a home with solar, '
            'and a zero-variance feature carries no information at all.'),
        'honest_limit': (
            'The PV overlay is a SCENARIO, not measured rooftop generation. IRES '
            'reports almost no rooftop solar in Andhra Pradesh, so a policy '
            'trained with it is prepared for solar homes, not validated on them.'),
    }
    (root / 'reports/scenario_compatibility_v1.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')

    print('SCENARIO COMPATIBILITY')
    for name, c in coverage.items():
        print(f"  {name:5s} {c['rows']:7,d} rows  solar in "
              f"{100 * c['steps_with_solar']:5.1f}% of steps  "
              f"self-sufficiency {c['mean_self_sufficient_fraction']:.3f}")
    print(f'  shared episode ids: {overlap} (expected; carry the scenario name)')
    if failures:
        print('\nINCOMPATIBLE - do not combine:')
        for failure in failures:
            print('  -', failure)
        return False
    print('\nSAFE TO COMBINE')
    print('  no household changes split, no date crosses a split, widths match')
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    raise SystemExit(0 if verify(a.root.resolve()) else 1)
