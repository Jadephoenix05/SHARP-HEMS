"""Calibrate each household's unmodelled background load against its own bill.

The problem this fixes. SHARP models only the appliances in its inventory, at
survey-reported usage hours. Comparing the result against the SAME households'
IRES-reported electricity bills showed it accounting for a median of 19 percent
of their actual consumption: 1.31 kWh per day modelled against 7.35 kWh per day
reported. Independently, eMARC Basic households draw 3.33 kWh per day against
SHARP's 1.31.

Four things the inventory cannot see explain the gap: appliances a survey
respondent did not list, usage hours people under-report, standby and charging
draw nobody thinks of as an appliance, and power values that are assumptions for
3,439 of 4,124 devices.

What this does. For each household it takes the residual between its reported
consumption and what SHARP models, and carries it as a constant BACKGROUND load.
The residual comes from that household's own reported bill, so this is
calibration against real data rather than an invented offset.

What it deliberately does NOT do. The background is not controllable and is
never offered to the agent as an action. It is load the household draws whatever
the controller decides, which is exactly what unmodelled load is. Scaling up
appliance power instead would have corrupted the appliance ratings to fix a
coverage problem, so that was rejected.

Honest limits. The residual absorbs every source of mismatch at once and cannot
attribute the gap between them. A household whose reported bill is wrong gets a
wrong background. Households with no reported bill take their split's median.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

HOURS_PER_DAY = 24.0
MAXIMUM_BACKGROUND_KW = 2.0   # refuse an implausible residual rather than model it


def check(condition, message):
    if not condition:
        raise ValueError(message)


def build(root, reference_policy):
    billing = pd.read_parquet(
        root / 'data/processed/appliance_inputs_v1/household_billing_position_v1.parquet')
    transitions = pd.read_parquet(
        root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet',
        columns=['episode_id', 'household_id', 'policy', 'aggregate_power_kw',
                 'controllable_power_kw'] if 'controllable_power_kw' in
        pd.read_parquet(root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet').columns
        else ['episode_id', 'household_id', 'policy', 'aggregate_power_kw'])
    reference = transitions[transitions.policy.eq(reference_policy)]
    check(len(reference) > 0, f'No {reference_policy} rows to calibrate against')

    # Measure only the appliances SHARP models. If a previous run already added
    # a background term, using aggregate power here would calibrate against the
    # calibration and the residual would collapse toward zero.
    column = ('controllable_power_kw' if 'controllable_power_kw' in reference.columns
              else 'aggregate_power_kw')
    print(f'  measuring modelled load from: {column}')
    per_episode = (reference.groupby(['episode_id', 'household_id'])[column]
                   .sum().mul(0.25).reset_index(name='modelled_kwh'))
    modelled = per_episode.groupby('household_id').modelled_kwh.mean()

    frame = billing[['template_id', 'split', 'monthly_kwh', 'basis']].copy()
    frame['reported_daily_kwh'] = frame.monthly_kwh / 30.0
    frame['modelled_daily_kwh'] = frame.template_id.map(modelled)
    frame = frame.dropna(subset=['modelled_daily_kwh'])
    check(len(frame) > 100, 'Too few households with both a bill and a simulation')

    frame['residual_daily_kwh'] = (frame.reported_daily_kwh
                                   - frame.modelled_daily_kwh).clip(lower=0.0)
    frame['background_kw'] = frame.residual_daily_kwh / HOURS_PER_DAY
    over = frame.background_kw > MAXIMUM_BACKGROUND_KW
    frame.loc[over, 'background_kw'] = MAXIMUM_BACKGROUND_KW
    frame['background_was_capped'] = over
    frame['modelled_share_of_reported'] = (frame.modelled_daily_kwh
                                           / frame.reported_daily_kwh.replace(0, np.nan))
    frame['basis_note'] = np.where(
        frame.basis.eq('INVERTED_FROM_REPORTED_BILL'),
        'RESIDUAL_AGAINST_OWN_REPORTED_BILL',
        'RESIDUAL_AGAINST_SPLIT_MEDIAN_ASSUMPTION')
    frame['is_controllable'] = False
    frame['is_measured'] = False

    check(np.isfinite(frame.background_kw.to_numpy(float)).all(), 'Nonfinite background')
    check(bool((frame.background_kw.to_numpy(float) >= 0).all()), 'Negative background')

    out = root / 'data/processed/appliance_inputs_v1'
    frame.to_parquet(out / 'household_background_load_v1.parquet', index=False)

    report = {
        'status': 'UNMODELLED_BACKGROUND_LOAD_CALIBRATED',
        'reference_policy': reference_policy,
        'households': int(len(frame)),
        'from_reported_bill': int(frame.basis.eq('INVERTED_FROM_REPORTED_BILL').sum()),
        'from_split_median': int((~frame.basis.eq('INVERTED_FROM_REPORTED_BILL')).sum()),
        'modelled_daily_kwh_median': float(frame.modelled_daily_kwh.median()),
        'reported_daily_kwh_median': float(frame.reported_daily_kwh.median()),
        'modelled_share_median': float(frame.modelled_share_of_reported.median()),
        'background_kw': {
            'min': float(frame.background_kw.min()),
            'p25': float(frame.background_kw.quantile(0.25)),
            'median': float(frame.background_kw.median()),
            'p75': float(frame.background_kw.quantile(0.75)),
            'max': float(frame.background_kw.max())},
        'capped_households': int(frame.background_was_capped.sum()),
        'cap_kw': MAXIMUM_BACKGROUND_KW,
        'what_the_residual_absorbs': [
            'Appliances the survey respondent did not report.',
            'Usage hours that are under-reported.',
            'Standby, charging and other draw not thought of as an appliance.',
            'Error in the 3,439 of 4,124 device power values that are assumptions.',
        ],
        'is_controllable_by_the_agent': False,
        'is_measured': False,
        'rejected_alternative': (
            'Scaling appliance power to close the gap. That would corrupt device '
            'ratings to fix a coverage problem and is not a calibration.'),
        'source_sha256': hashlib.sha256(
            (out / 'household_billing_position_v1.parquet').read_bytes()).hexdigest(),
    }
    (root / 'reports/unmodelled_background_load_v1.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')

    print('UNMODELLED BACKGROUND LOAD CALIBRATED')
    print(f"  households: {report['households']}  "
          f"({report['from_reported_bill']} from their own bill)")
    print(f"  modelled daily kWh  median {report['modelled_daily_kwh_median']:.2f}")
    print(f"  reported daily kWh  median {report['reported_daily_kwh_median']:.2f}")
    print(f"  SHARP covers median {100 * report['modelled_share_median']:.1f}% "
          'of reported consumption')
    print(f"  background kW  median {report['background_kw']['median']:.4f} "
          f"(p25 {report['background_kw']['p25']:.4f} "
          f"p75 {report['background_kw']['p75']:.4f})")
    print(f"  capped at {MAXIMUM_BACKGROUND_KW} kW: {report['capped_households']} households")
    print('  background is NOT controllable and is never an agent action')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--reference-policy', default='serve_preferred')
    a = p.parse_args()
    build(a.root.resolve(), a.reference_policy)
