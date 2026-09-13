"""Recover each household's monthly consumption by inverting the APCPDCL tariff.

Why this exists. Episodes are one simulated day, and the generator previously
opened every billing period at zero kWh. With a telescopic tariff that is not a
small simplification: a household consuming 1.4 kWh in a day never leaves the
first slab, so the agent only ever sees the lowest marginal rate of 1.90 INR per
kWh and never the 8.75 or 9.75 rates where demand response actually pays. The
entire incentive structure of the tariff was invisible to the policy.

The fix uses evidence the survey already carries. IRES reports
average_monthly_bill_inr for 392 of 498 Andhra Pradesh households. The APCPDCL
telescopic tariff is monotonic in consumption, so the bill can be inverted by
bisection to recover the monthly kWh consistent with that reported bill and the
household's sanctioned load.

WHAT THIS IS AND IS NOT. The recovered kWh is an inference from a reported bill
under a FY2025-26 tariff, applied to households surveyed earlier. It is not a
metered reading. Households with no reported bill get their split's median as a
declared assumption, flagged so nothing downstream mistakes it for a report.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd

from sharp_apcpdcl_tariff import Tariff

MAXIMUM_MONTHLY_KWH = 5000.0
BISECTION_STEPS = 60


def check(condition, message):
    if not condition:
        raise ValueError(message)


def invert_bill(tariff, bill_inr, connected_kw):
    """Smallest monthly kWh whose tariff subtotal reaches the reported bill."""
    bill = float(bill_inr)
    if not math.isfinite(bill) or bill <= 0:
        return None
    floor_bill = float(tariff.components(0, connected_kw)['tariff_subtotal_inr'])
    if bill <= floor_bill:
        # The reported bill is at or below the unavoidable fixed component.
        return 0.0
    ceiling = float(tariff.components(MAXIMUM_MONTHLY_KWH, connected_kw)['tariff_subtotal_inr'])
    if bill >= ceiling:
        return MAXIMUM_MONTHLY_KWH
    low, high = 0.0, MAXIMUM_MONTHLY_KWH
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        value = float(tariff.components(middle, connected_kw)['tariff_subtotal_inr'])
        if value < bill:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def build(root):
    households = pd.read_parquet(
        root / 'data/processed/appliance_inputs_v1/ap_households_with_splits_v1.parquet')
    check(len(households) == 498, 'Expected 498 household templates')
    tariff = Tariff.load(root / 'configs/tariffs/apcpdcl_2025_26_verified_components.json')

    rows = []
    for r in households.itertuples():
        connected = pd.to_numeric(getattr(r, 'sanctioned_load_kw'), errors='coerce')
        if not (isinstance(connected, float) and math.isfinite(connected) and connected > 0):
            connected = None
        bill = getattr(r, 'average_monthly_bill_inr')
        recovered = None
        if connected is not None and pd.notna(bill):
            recovered = invert_bill(tariff, bill, float(connected))
        rows.append({
            'template_id': r.template_id, 'split': r.split,
            'sanctioned_load_kw': float(connected) if connected is not None else None,
            'reported_monthly_bill_inr': float(bill) if pd.notna(bill) else None,
            'monthly_kwh': recovered,
            'basis': ('INVERTED_FROM_REPORTED_BILL' if recovered is not None
                      else 'PENDING_COHORT_MEDIAN')})

    frame = pd.DataFrame(rows)
    reported = frame[frame.basis.eq('INVERTED_FROM_REPORTED_BILL')]
    check(len(reported) > 100, 'Too few households with a usable reported bill')

    # Missing households take their OWN split's median, so no split learns from
    # another. Fall back to the reported median only if a split has none.
    overall = float(reported.monthly_kwh.median())
    medians = reported.groupby('split').monthly_kwh.median().to_dict()
    filled = 0
    for index, row in frame.iterrows():
        if row.basis == 'PENDING_COHORT_MEDIAN':
            frame.at[index, 'monthly_kwh'] = float(medians.get(row.split, overall))
            frame.at[index, 'basis'] = 'SPLIT_MEDIAN_DECLARED_ASSUMPTION'
            filled += 1

    check(frame.monthly_kwh.notna().all(), 'Null monthly kWh after fill')
    check(np.isfinite(frame.monthly_kwh.to_numpy(float)).all(), 'Nonfinite monthly kWh')
    check(bool((frame.monthly_kwh.to_numpy(float) >= 0).all()), 'Negative monthly kWh')
    frame['daily_kwh'] = frame.monthly_kwh / 30.0
    frame['is_inferred_not_metered'] = True

    out = root / 'data/processed/appliance_inputs_v1'
    frame.to_parquet(out / 'household_billing_position_v1.parquet', index=False)

    # What marginal rate will these households actually face?
    rates = [float(tariff.next_unit_rate(k)) for k in frame.monthly_kwh]
    frame_rates = pd.Series(rates)
    report = {
        'status': 'HOUSEHOLD_BILLING_POSITION_DERIVED',
        'households': int(len(frame)),
        'inverted_from_reported_bill': int(len(reported)),
        'split_median_assumption': int(filled),
        'monthly_kwh': {
            'min': float(frame.monthly_kwh.min()),
            'p25': float(frame.monthly_kwh.quantile(0.25)),
            'median': float(frame.monthly_kwh.median()),
            'p75': float(frame.monthly_kwh.quantile(0.75)),
            'max': float(frame.monthly_kwh.max())},
        'marginal_rate_inr_kwh': {
            'min': float(frame_rates.min()),
            'median': float(frame_rates.median()),
            'max': float(frame_rates.max()),
            'distribution': {str(k): int(v) for k, v in
                             frame_rates.value_counts().sort_index().items()}},
        'split_medians_monthly_kwh': {k: float(v) for k, v in medians.items()},
        'limitations': [
            'Monthly kWh is inferred by inverting a FY2025-26 tariff from a reported bill.',
            'It is not a metered reading, and the survey predates that tariff year.',
            'Households without a reported bill take their own split median, flagged as an assumption.',
            'Split medians are computed within a split so no split informs another.',
        ],
        'is_metered': False,
    }
    (root / 'reports/household_billing_position_v1.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')

    print('HOUSEHOLD BILLING POSITION DERIVED')
    print(f"  inverted from reported bill: {report['inverted_from_reported_bill']} / 498")
    print(f"  split-median assumption:     {report['split_median_assumption']}")
    print(f"  monthly kWh  median {report['monthly_kwh']['median']:.1f} "
          f"(p25 {report['monthly_kwh']['p25']:.1f}, p75 {report['monthly_kwh']['p75']:.1f})")
    print('  marginal rate distribution (INR/kWh):')
    for rate, count in report['marginal_rate_inr_kwh']['distribution'].items():
        print(f'    {rate:>6s} : {count}')
    print('  output:', out / 'household_billing_position_v1.parquet')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
