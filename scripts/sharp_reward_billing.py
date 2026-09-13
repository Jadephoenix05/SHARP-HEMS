"""SHARP reward components and resumable per-household billing ledger.
FY2025-26 tariff applied as a simulation scenario, not a historical bill.
No reward weights, comfort model or export compensation are inferred here.
"""
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import json
from sharp_apcpdcl_tariff import Tariff, nonnegative

@dataclass(frozen=True)
class RewardWeights:
    cost_per_inr: float
    grid_peak_per_kwh: float
    discomfort_per_unit: float
    switching_per_event: float
    unmet_service_per_unit: float

    def validate(self):
        for name, value in vars(self).items():
            nonnegative(value, name)
        if not any(vars(self).values()):
            raise ValueError('At least one reward weight must be positive')


class BillingLedger:
    """One household, one billing period. Reuse across daily episodes.
    Caller explicitly defines billing-period boundaries. Month-start resets
    must never be substituted for an unknown real meter-reading cycle.
    A new period requires a new ledger, not a daily reset of this object.
    """
    def __init__(self, tariff, household_id, billing_period_id, connected_kw, *,
                 opening_kwh, opening_charges_already_booked):
        if not household_id or not billing_period_id:
            raise ValueError('Household and billing-period identifiers required')
        if type(opening_charges_already_booked) is not bool:
            raise ValueError('Opening-charge accounting must be explicit')
        self.tariff = tariff
        self.household_id = str(household_id)
        self.billing_period_id = str(billing_period_id)
        self.connected_kw = nonnegative(connected_kw, 'connected_kw')
        self.tariff.fixed_charge(self.connected_kw)  # Reject absent/zero load.
        self.kwh = nonnegative(opening_kwh, 'opening_kwh')
        if self.kwh > 0 and not opening_charges_already_booked:
            raise ValueError('Nonzero warm-start consumption requires previously booked opening charges')
        self.opening_booked = opening_charges_already_booked
        self.last_step = -1
        self.booked_since_initialization = Decimal(0)

    def charge(self, *, step_index, import_kwh):
        # Consecutive indices prevent repeated/skipped billing of transitions.
        if type(step_index) is not int or step_index != self.last_step + 1:
            raise ValueError('Billing step must follow the previous step exactly once')
        q = nonnegative(import_kwh, 'import_kwh')
        inc = self.tariff.incremental_components(self.kwh, q)
        opening_customer = Decimal(0)
        fixed = Decimal(0)
        if not self.opening_booked:
            opening_customer = self.tariff.customer_charge(0)
            fixed = self.tariff.fixed_charge(self.connected_kw)
        cost = inc['tariff_increment_inr'] + opening_customer + fixed
        result = {
            'household_id': self.household_id, 'billing_period_id': self.billing_period_id,
            'tariff_id': self.tariff.tariff_id,
            'step_index': step_index, 'month_kwh_before': str(self.kwh),
            'month_kwh_after': str(inc['month_kwh_after']),
            'import_kwh': str(q), 'energy_inr': str(inc['energy_inr']),
            'customer_increment_inr': str(inc['customer_inr']),
            'opening_customer_inr': str(opening_customer), 'fixed_inr': str(fixed),
            'cost_inr': str(cost), 'full_utility_bill_verified': False,
        }
        self.kwh = inc['month_kwh_after']; self.opening_booked = True
        self.last_step = step_index; self.booked_since_initialization += cost
        return result

    def state(self):
        return {'version':1, 'tariff_id':self.tariff.tariff_id,
                'household_id':self.household_id, 'billing_period_id':self.billing_period_id,
                'connected_kw':str(self.connected_kw), 'cumulative_kwh':str(self.kwh),
                'opening_booked':self.opening_booked, 'last_step':self.last_step,
                'booked_since_initialization':str(self.booked_since_initialization)}

    @classmethod
    def restore(cls, tariff, state):
        if state['version'] != 1 or state['tariff_id'] != tariff.tariff_id:
            raise ValueError('Ledger version or tariff mismatch')
        obj = cls(tariff, state['household_id'], state['billing_period_id'],
                  state['connected_kw'], opening_kwh=state['cumulative_kwh'],
                  opening_charges_already_booked=state['opening_booked'])
        last = state['last_step']
        if type(last) is not int or last < -1:
            raise ValueError('Invalid checkpoint step')
        if last >= 0 and not state['opening_booked']:
            raise ValueError('Inconsistent opening-charge state')
        obj.last_step = last
        obj.booked_since_initialization = nonnegative(state['booked_since_initialization'], 'booked_total')
        return obj


def reward_components(*, billing, grid_peak_severity, discomfort_units,
                      switching_events, unmet_service_units, weights):
    """All penalties are explicit, nonnegative inputs, evaluated after action.
    Discomfort and unmet service come from separate appliance/occupant models.
    Do not substitute absence of a human override for zero discomfort.
    """
    weights.validate()
    severity = nonnegative(grid_peak_severity, 'grid_peak_severity')
    if severity > 1:
        raise ValueError('Grid severity must be in [0,1]')
    if type(switching_events) is not int or switching_events < 0:
        raise ValueError('Switching count must be a nonnegative integer')
    raw = {
        'cost_inr':nonnegative(billing['cost_inr'], 'cost_inr'),
        'grid_peak_kwh':nonnegative(billing['import_kwh'], 'import_kwh') * severity,
        'discomfort_units':nonnegative(discomfort_units, 'discomfort_units'),
        'switching_events':Decimal(switching_events),
        'unmet_service_units':nonnegative(unmet_service_units, 'unmet_service_units'),
    }
    ws = [weights.cost_per_inr, weights.grid_peak_per_kwh,
          weights.discomfort_per_unit, weights.switching_per_event,
          weights.unmet_service_per_unit]
    weighted = {name: value * nonnegative(w, 'weight') for (name,value),w in zip(raw.items(),ws)}
    # Peak cost is a synthetic reward penalty, never an official ToD surcharge.
    return {'reward':float(-sum(weighted.values())),
            'raw_components':{k:float(v) for k,v in raw.items()},
            'weighted_penalties':{k:float(v) for k,v in weighted.items()}}


def self_test():
    tariff = Tariff.load()
    ledger = BillingLedger(tariff, 'test_house', 'test_period', 2,
                           opening_kwh=0, opening_charges_already_booked=False)
    total = Decimal(0)
    quantities = ['29','2','50','180','139','0.5']
    for i, q in enumerate(quantities):
        bill = ledger.charge(step_index=i, import_kwh=q)
        total += Decimal(bill['cost_inr'])
        if i > 0:
            assert bill['fixed_inr'] == '0' and bill['opening_customer_inr'] == '0'
        # Restore every step, as a simulator resume or daily boundary would.
        ledger = BillingLedger.restore(tariff, json.loads(json.dumps(ledger.state())))
    assert total == tariff.components('400.5',2)['tariff_subtotal_inr']
    assert total == ledger.booked_since_initialization
    before = ledger.state()
    try:
        ledger.charge(step_index=5, import_kwh=1)
    except ValueError:
        pass
    else:
        raise AssertionError('Repeated step accepted')
    assert ledger.state() == before
    try:
        ledger.charge(step_index=6, import_kwh=-1)
    except ValueError:
        pass
    else:
        raise AssertionError('Negative import accepted')
    assert ledger.state() == before
    warm = BillingLedger(tariff,'test_house','warm_period',2,opening_kwh=29,
                         opening_charges_already_booked=True)
    crossing = warm.charge(step_index=0,import_kwh=2)
    assert Decimal(crossing['energy_inr']) == Decimal('4.9')
    assert Decimal(crossing['opening_customer_inr']) == 0
    assert Decimal(crossing['fixed_inr']) == 0
    # Test-only weights; these are not a selected project reward configuration.
    weights = RewardWeights(1,2,3,4,5)
    r = reward_components(billing={'cost_inr':'10','import_kwh':'2'},
        grid_peak_severity=.5,discomfort_units=2,switching_events=1,
        unmet_service_units=1,weights=weights)
    assert r['reward'] == -27
    new_period = BillingLedger(tariff,'test_house','next_period',2,opening_kwh=0,
                               opening_charges_already_booked=False)
    zero = new_period.charge(step_index=0,import_kwh=0)
    assert Decimal(zero['cost_inr']) == tariff.components(0,2)['tariff_subtotal_inr']
    report={'status':'PASS','module':'reward_billing_v1',
            'tested':['monthly slab reconciliation','opening fees booked once',
                      'checkpoint restore','duplicate-step rejection','negative-import rejection',
                      'warm-start slab crossing','explicit new billing period','reward arithmetic'],
            'full_utility_bill_verified':False,'reward_weights_selected':False,
            'full_simulator_ready':False,'master_release_ready':False}
    p=Path(__file__).resolve().parents[1]/'reports/sharp_reward_billing_validation_v1.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('Report:',p)

if __name__=='__main__':
    self_test()
