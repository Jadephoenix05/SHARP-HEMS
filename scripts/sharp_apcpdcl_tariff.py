"""Verified FY2025-26 Cat I(A) LT tariff components; not a complete utility bill.

For an RL transition use incremental_components(month_kwh_before, step_import_kwh).
Keep cumulative billing-period consumption in the state across daily episodes.
All calculation uses Decimal; round only the final displayed subtotal.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path
import argparse
import json

D = Decimal
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/tariffs/apcpdcl_2025_26_verified_components.json'


def nonnegative(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be numeric, not boolean')
    try:
        result = D(str(value))
    except Exception as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return result


@dataclass(frozen=True)
class Tariff:
    bounds: tuple
    rates: tuple
    customer_charges: tuple
    fixed_per_kw: Decimal
    tariff_id: str

    @classmethod
    def load(cls, path=CONFIG_PATH):
        c = json.loads(Path(path).read_text(encoding='utf-8'))
        rows = c['slabs']
        bounds = tuple(None if x['upper_kwh'] is None else D(str(x['upper_kwh'])) for x in rows)
        rates = tuple(nonnegative(x['energy_inr_kwh'], 'rate') for x in rows)
        charges = tuple(nonnegative(x['customer_inr_month'], 'customer charge') for x in rows)
        if bounds[-1] is not None or any(x is None for x in bounds[:-1]):
            raise ValueError('Only the last slab must be unbounded')
        if not bounds[0] > 0 or any(b <= a for a, b in zip(bounds[:-2], bounds[1:-1])):
            raise ValueError('Slab bounds must increase')
        if c['method'] != 'telescopic' or c['time_of_day_rates']:
            raise ValueError('Unsupported tariff method')
        return cls(bounds, rates, charges,
                   nonnegative(c['fixed_inr_per_kw_or_part_month'], 'fixed charge'),
                   c['tariff_id'])

    def energy_charge(self, kwh):
        q = nonnegative(kwh, 'kwh')
        cost = D(0)
        lower = D(0)
        for upper, rate in zip(self.bounds, self.rates):
            amount = max(D(0), q - lower) if upper is None else max(D(0), min(q, upper) - lower)
            cost += amount * rate
            if upper is None or q <= upper:
                break
            lower = upper
        return cost

    def customer_charge(self, kwh):
        q = nonnegative(kwh, 'kwh')
        for upper, fee in zip(self.bounds, self.customer_charges):
            if upper is None or q <= upper:
                return fee
        raise AssertionError('No customer charge slab')

    def fixed_charge(self, connected_kw):
        kw = nonnegative(connected_kw, 'connected_kw')
        if kw == 0:
            raise ValueError('A positive connected load is required; do not invent missing survey loads')
        return self.fixed_per_kw * kw.to_integral_value(rounding=ROUND_CEILING)

    def components(self, kwh, connected_kw):
        energy = self.energy_charge(kwh)
        customer = self.customer_charge(kwh)
        fixed = self.fixed_charge(connected_kw)
        return {'energy_inr': energy, 'customer_inr': customer, 'fixed_inr': fixed,
                'tariff_subtotal_inr': energy + customer + fixed}

    def incremental_components(self, month_kwh_before, step_import_kwh):
        before = nonnegative(month_kwh_before, 'month_kwh_before')
        after = before + nonnegative(step_import_kwh, 'step_import_kwh')
        energy = self.energy_charge(after) - self.energy_charge(before)
        customer = self.customer_charge(after) - self.customer_charge(before)
        # Fixed connected-load charge is booked once per billing period, not per step.
        return {'energy_inr': energy, 'customer_inr': customer,
                'tariff_increment_inr': energy + customer,
                'month_kwh_after': after}

    def next_unit_rate(self, month_kwh):
        q = nonnegative(month_kwh, 'month_kwh')
        for upper, rate in zip(self.bounds, self.rates):
            if upper is None or q < upper:
                return rate
        raise AssertionError('No energy slab')


def validate_examples(t):
    # Independent hand calculations from the printed official rate table.
    cases = [(0, '0'), (30, '57'), (75, '192'), (125, '417'),
             (225, '1017'), (400, '2548.25'), (401, '2558'), (500, '3523.25')]
    for kwh, expected in cases:
        if t.energy_charge(kwh) != D(expected):
            raise AssertionError(f'Energy mismatch at {kwh} kWh')
    if t.incremental_components(29, 2)['energy_inr'] != D('4.9'):
        raise AssertionError('Incorrect cross-slab energy')
    if t.fixed_charge('2.1') != D('30'):
        raise AssertionError('Incorrect fixed-charge rounding')
    # Stepped rewards must telescope to the same monthly subtotal.
    q = D(0)
    accumulated = t.components(0, 2)['tariff_subtotal_inr']
    for kwh in [D('29'), D('2'), D('50'), D('180'), D('139'), D('0.5')]:
        inc = t.incremental_components(q, kwh)
        accumulated += inc['tariff_increment_inr']
        q = inc['month_kwh_after']
    if accumulated != t.components(q, 2)['tariff_subtotal_inr']:
        raise AssertionError('Step totals do not reconcile')
    return {'status': 'PASS', 'scope': 'official_order_energy_fixed_customer_components',
            'full_utility_bill_verified': False, 'domestic_time_of_day': False,
            'energy_boundary_examples': len(cases), 'billing_state': 'cumulative_per_billing_period'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--kwh')
    p.add_argument('--connected-kw')
    args = p.parse_args()
    t = Tariff.load()
    if args.validate:
        report = validate_examples(t)
        dest = ROOT / 'reports/apcpdcl_tariff_component_validation_v1.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))
        print('Saved:', dest)
    elif args.kwh is not None and args.connected_kw is not None:
        r = t.components(args.kwh, args.connected_kw)
        print(json.dumps({k: str(v.quantize(D('.01'), rounding=ROUND_HALF_UP)) for k, v in r.items()}, indent=2))
        print('Excludes duty, FPPCA, true-up/down, subsidies and other recoveries; not a complete bill.')
    else:
        p.error('Use --validate or both --kwh and --connected-kw')

if __name__ == '__main__':
    main()
