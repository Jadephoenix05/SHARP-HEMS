"""SHARP binary-device action shield, version 1.
Pure simulation logic; this is not a certified hardware safety controller.
Power estimates and timer limits must be supplied by the appliance models.
A branch represents one modelled device, not an entire appliance category.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math

@dataclass(frozen=True)
class Device:
    device_id: str
    current_on: bool
    available: bool
    estimated_on_w: float
    estimated_off_w: float = 0.0
    must_run: bool = False
    force_off: bool = False
    noninterruptible_cycle_active: bool = False
    elapsed_state_steps: int = 0
    min_on_steps: int = 0
    min_off_steps: int = 0
    shed_priority: int = 0  # Higher number is shed first.


def validate_device(d):
    if not d.device_id:
        raise ValueError('Missing device ID')
    for field in ('current_on', 'available', 'must_run', 'force_off',
                  'noninterruptible_cycle_active'):
        if not isinstance(getattr(d, field), bool):
            raise ValueError(f'{field} must be a boolean')
    for field in ('estimated_on_w', 'estimated_off_w'):
        value = getattr(d, field)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{field} must be finite and nonnegative')
    if d.estimated_on_w < d.estimated_off_w:
        raise ValueError('ON power must not be below OFF/standby power')
    for field in ('elapsed_state_steps', 'min_on_steps', 'min_off_steps', 'shed_priority'):
        value = getattr(d, field)
        if type(value) is not int or value < 0:
            raise ValueError(f'{field} must be a nonnegative integer')
    if d.noninterruptible_cycle_active and not d.current_on:
        raise ValueError('Active noninterruptible cycle requires current_on')


def local_mask(d):
    """Return [OFF allowed, ON allowed] and reasons. Fault/OFF takes precedence.
    Minimum-off lockout takes precedence over a demand to run; this conflict
    is exposed instead of silently violating the lockout.
    """
    validate_device(d)
    if not d.available or d.force_off:
        reasons = ['unavailable' if not d.available else 'forced_off']
        if d.must_run or d.noninterruptible_cycle_active:
            reasons.append('required_service_interrupted')
        return [True, False], reasons
    if d.current_on:
        if d.must_run or d.noninterruptible_cycle_active or d.elapsed_state_steps < d.min_on_steps:
            return [False, True], ['protected_on']
    else:
        if d.elapsed_state_steps < d.min_off_steps:
            return [True, False], ['minimum_off_lockout'] + (['required_service_delayed'] if d.must_run else [])
        if d.must_run:
            return [False, True], ['required_on']
    return [True, True], []


def apply_shield(devices, requested, *, max_import_w, base_load_w=0.0,
                 available_solar_w=0.0, human_actions=None):
    """Project a requested binary vector onto local constraints and import cap.
    human_actions maps device IDs to explicit requested 0/1 overrides.
    Coupled capacity is checked after local masks; independent masks alone
    cannot encode a household-wide power constraint.
    """
    for value in (max_import_w, base_load_w, available_solar_w):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Power bounds must be finite and nonnegative')
    ids = [d.device_id for d in devices]
    if len(set(ids)) != len(ids) or len(requested) != len(devices):
        raise ValueError('Duplicate device IDs or wrong action-vector length')
    human_actions = {} if human_actions is None else dict(human_actions)
    if set(human_actions) - set(ids):
        raise ValueError('Human override references an unknown device')
    for value in list(requested) + list(human_actions.values()):
        if type(value) not in (int, bool) or value not in (0, 1):
            raise ValueError('Actions must be binary integers or booleans')
    masks, reasons, effective, executed = [], [], [], []
    for d, request in zip(devices, requested):
        mask, why = local_mask(d)
        wanted = int(human_actions.get(d.device_id, request))
        actual = wanted if mask[wanted] else int(mask[1])
        masks.append(mask); reasons.append(why)
        effective.append(wanted); executed.append(actual)
    def net_import():
        demand = base_load_w + sum(d.estimated_on_w if a else d.estimated_off_w
                                   for d, a in zip(devices, executed))
        return max(0.0, demand - available_solar_w)
    order = sorted(range(len(devices)), key=lambda i: (-devices[i].shed_priority, devices[i].device_id))
    for i in order:
        if net_import() <= max_import_w + 1e-9:
            break
        if executed[i] and masks[i][0] and devices[i].estimated_on_w > devices[i].estimated_off_w:
            executed[i] = 0
            reasons[i].append('import_capacity_shedding')
    excess = max(0.0, net_import() - max_import_w)
    return {
        'device_ids': ids, 'requested_actions': list(map(int, requested)),
        'human_requested_actions': human_actions, 'effective_requested_actions': effective,
        'executed_actions': executed, 'local_legal_action_masks': masks,
        'shield_reasons': reasons, 'estimated_import_w': net_import(),
        'capacity_excess_w': excess, 'capacity_feasible': excess <= 1e-9,
        'action_modified': [a != b for a, b in zip(effective, executed)],
        'human_override_honored': {key: executed[ids.index(key)] == int(value)
                                   for key, value in human_actions.items()},
    }


def advance_timers(devices, executed):
    """Timer state for the next step; appliance model advances cycle state."""
    if len(devices) != len(executed):
        raise ValueError('Wrong action-vector length')
    return [{'device_id': d.device_id, 'current_on': bool(a),
             'elapsed_state_steps': d.elapsed_state_steps + 1 if bool(a) == d.current_on else 1}
            for d, a in zip(devices, executed)]


def self_test():
    from itertools import product
    protected = Device('fridge', True, True, 100, must_run=True)
    flexible = Device('flexible', True, True, 1000, shed_priority=10)
    r = apply_shield([protected, flexible], [0, 1], max_import_w=500)
    assert r['executed_actions'] == [1, 0] and r['capacity_feasible']
    r = apply_shield([protected], [0], max_import_w=50)
    assert not r['capacity_feasible'] and r['capacity_excess_w'] == 50
    lock = Device('ac', False, True, 1800, elapsed_state_steps=1, min_off_steps=2)
    assert apply_shield([lock], [1], max_import_w=3000)['executed_actions'] == [0]
    running = Device('washer', True, True, 400, noninterruptible_cycle_active=True)
    assert apply_shield([running], [0], max_import_w=500)['executed_actions'] == [1]
    fault = Device('fault', True, True, 100, force_off=True, must_run=True)
    assert apply_shield([fault], [1], max_import_w=500)['executed_actions'] == [0]
    r = apply_shield([flexible], [0], max_import_w=500, human_actions={'flexible': 1})
    assert not r['human_override_honored']['flexible']
    r = apply_shield([flexible], [1], max_import_w=500, available_solar_w=600)
    assert r['executed_actions'] == [1] and r['estimated_import_w'] == 400
    standby = Device('standby', False, True, 100, estimated_off_w=10)
    assert apply_shield([standby], [0], max_import_w=5)['capacity_excess_w'] == 5
    assert advance_timers([lock], [1])[0]['elapsed_state_steps'] == 1
    assert advance_timers([lock], [0])[0]['elapsed_state_steps'] == 2
    checked = 0
    for actions in product((0, 1), repeat=3):
        ds = [protected, flexible, lock]
        r = apply_shield(ds, list(actions), max_import_w=600)
        assert all(mask[a] for mask, a in zip(r['local_legal_action_masks'], r['executed_actions']))
        assert r == apply_shield(ds, list(actions), max_import_w=600)
        checked += 1
    try:
        apply_shield([protected], [2], max_import_w=500)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid action accepted')
    report = {'status':'PASS', 'module':'binary_device_action_shield_v1',
              'exhaustive_three_device_action_vectors':checked,
              'tested':['protected loads','minimum off time','cycle protection','fault precedence',
                        'infeasible capacity','human override constrained by cap','solar offset',
                        'standby demand','state timers','determinism','invalid actions'],
              'scope':'simulation decision logic only', 'hardware_safety_certified':False,
              'full_simulator_ready':False, 'master_release_ready':False}
    root = Path(__file__).resolve().parents[1]
    p = root/'reports/sharp_action_shield_validation_v1.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2)); print('Report:', p)

if __name__ == '__main__':
    self_test()
