"""SHARP three-level action shield, version 2.

Action levels, matching the project spec's {allow, dim, shed} for lighting and
{allow, shed} elsewhere:

    0 = OFF       shed the load
    1 = ON        full power
    2 = REDUCED   dimmed or eco power, only where the appliance supports it

Level 2 sits between 0 and 1 in power but is a distinct branch value, so a
policy can choose to dim rather than face an all-or-nothing shed. Capacity
shedding steps a device DOWN one level at a time - full to reduced to off -
rather than dropping straight to off, so the least disruptive option is taken
first.
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
    supports_reduced: bool = False   # can this appliance run dimmed or in eco?
    estimated_reduced_w: float = 0.0


def validate_device(d):
    if not d.device_id:
        raise ValueError('Missing device ID')
    for field in ('current_on', 'available', 'must_run', 'force_off',
                  'noninterruptible_cycle_active'):
        if not isinstance(getattr(d, field), bool):
            raise ValueError(f'{field} must be a boolean')
    if not isinstance(d.supports_reduced, bool):
        raise ValueError('supports_reduced must be a boolean')
    for field in ('estimated_on_w', 'estimated_off_w', 'estimated_reduced_w'):
        value = getattr(d, field)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{field} must be finite and nonnegative')
    if d.estimated_on_w < d.estimated_off_w:
        raise ValueError('ON power must not be below OFF/standby power')
    if d.supports_reduced and not (d.estimated_off_w <= d.estimated_reduced_w
                                   <= d.estimated_on_w):
        raise ValueError('Reduced power must sit between OFF and ON power')
    for field in ('elapsed_state_steps', 'min_on_steps', 'min_off_steps', 'shed_priority'):
        value = getattr(d, field)
        if type(value) is not int or value < 0:
            raise ValueError(f'{field} must be a nonnegative integer')
    if d.noninterruptible_cycle_active and not d.current_on:
        raise ValueError('Active noninterruptible cycle requires current_on')


def local_mask(d):
    """Return [OFF allowed, ON allowed, REDUCED allowed] and reasons.

    Fault and OFF take precedence. A minimum-off lockout takes precedence over a
    demand to run; that conflict is exposed rather than silently violated.

    A device that must run may still be REDUCED where it supports it: dimming a
    light keeps the service while lowering demand, which is the point of having
    a third level at all.
    """
    validate_device(d)
    reduced = d.supports_reduced
    if not d.available or d.force_off:
        reasons = ['unavailable' if not d.available else 'forced_off']
        if d.must_run or d.noninterruptible_cycle_active:
            reasons.append('required_service_interrupted')
        return [True, False, False], reasons
    if d.current_on:
        if d.must_run or d.noninterruptible_cycle_active or d.elapsed_state_steps < d.min_on_steps:
            return [False, True, reduced], ['protected_on']
    else:
        if d.elapsed_state_steps < d.min_off_steps:
            return [True, False, False], ['minimum_off_lockout'] + (
                ['required_service_delayed'] if d.must_run else [])
        if d.must_run:
            return [False, True, reduced], ['required_on']
    return [True, True, reduced], []


# Power ordering of the levels, highest first. Capacity shedding walks this.
LEVELS_BY_POWER = [1, 2, 0]


def level_power(d, level):
    """Estimated draw at an action level."""
    if level == 1:
        return float(d.estimated_on_w)
    if level == 2:
        return float(d.estimated_reduced_w)
    return float(d.estimated_off_w)


def apply_shield(devices, requested, *, max_import_w, base_load_w=0.0,
                 available_solar_w=0.0, human_actions=None,
                 peak_locked_out=None):
    """Project a requested action vector onto local constraints and the import cap.

    WHO IS BEING CONSTRAINED MATTERS. The shield exists to stop the CONTROLLER
    from degrading essential service. It is not there to stop a resident using
    their own switch, and it could not be: a fan has a physical switch on the
    wall.

      - The POLICY may never shed a critical load that is in use.
      - A RESIDENT may switch anything off, including their own fan. It is their
        house. `human_actions` asking for level 0 always wins.
      - A RESIDENT may NOT energise a load in `peak_locked_out`. During a peak
        the luxury circuit is not carrying power, so switching the air
        conditioner on simply does nothing - the request is recorded and
        refused, not silently ignored.

    `human_actions` maps device IDs to explicit 0/1/2 requests from the
    occupant. `peak_locked_out` is the set of device IDs that cannot be
    energised this interval whatever anyone asks.

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
    peak_locked_out = set() if peak_locked_out is None else set(peak_locked_out)
    if peak_locked_out - set(ids):
        raise ValueError('Peak lockout references an unknown device')
    locked_critical = [d.device_id for d in devices
                       if d.device_id in peak_locked_out and d.must_run]
    if locked_critical:
        raise ValueError(
            f'Critical loads cannot be locked out at peak: {locked_critical}. '
            'A peak restricts luxury; it never cuts essential service.')
    for value in list(requested) + list(human_actions.values()):
        if type(value) not in (int, bool) or int(value) not in (0, 1, 2):
            raise ValueError('Actions must be 0 (off), 1 (on) or 2 (reduced)')
    masks, reasons, effective, executed = [], [], [], []
    refusals = {}
    for d, request in zip(devices, requested):
        mask, why = local_mask(d)
        by_human = d.device_id in human_actions
        wanted = int(human_actions.get(d.device_id, request))

        if d.device_id in peak_locked_out and wanted != 0:
            # No power on this circuit right now. Refuse and say so, rather than
            # accept the request and quietly do nothing.
            actual = 0
            why = why + ['peak_lockout']
            refusals[d.device_id] = 'peak_lockout'
        elif by_human and wanted == 0:
            # The resident switching their own appliance off. Always honoured -
            # the shield protects essential service from the CONTROLLER, not
            # from the person who lives there.
            actual = 0
            why = why + ['occupant_switched_off']
        else:
            # Fall back to the highest-power level still permitted, which
            # preserves required service; OFF only when nothing else is allowed.
            actual = wanted if mask[wanted] else next(
                (level for level in LEVELS_BY_POWER if mask[level]), 0)
            if actual != wanted and not by_human:
                refusals[d.device_id] = 'local_mask'
        masks.append(mask); reasons.append(why)
        effective.append(wanted); executed.append(actual)
    def net_import():
        demand = base_load_w + sum(level_power(d, a)
                                   for d, a in zip(devices, executed))
        return max(0.0, demand - available_solar_w)
    order = sorted(range(len(devices)), key=lambda i: (-devices[i].shed_priority, devices[i].device_id))
    # Step each device DOWN one level at a time rather than straight to off, so
    # a dimmable load is dimmed before it is shed. Repeat until the import fits
    # or no further reduction is permitted anywhere.
    progress = True
    while net_import() > max_import_w + 1e-9 and progress:
        progress = False
        for i in order:
            if net_import() <= max_import_w + 1e-9:
                break
            current = executed[i]
            lower = [level for level in LEVELS_BY_POWER
                     if masks[i][level]
                     and level_power(devices[i], level) < level_power(devices[i], current)]
            if not lower:
                continue
            executed[i] = lower[0]
            reasons[i].append('import_capacity_shedding')
            progress = True
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
        'peak_locked_out': sorted(peak_locked_out),
        'refusals': refusals,
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
    # A dimmable light: reduced power sits between off and full.
    lamp = Device('lamp', True, True, 100, estimated_off_w=0,
                  supports_reduced=True, estimated_reduced_w=40, shed_priority=5)
    assert local_mask(lamp)[0] == [True, True, True]
    assert apply_shield([lamp], [2], max_import_w=500)['executed_actions'] == [2]
    stepped = apply_shield([lamp], [1], max_import_w=50)
    assert stepped['executed_actions'] == [2], stepped['executed_actions']
    assert apply_shield([lamp], [1], max_import_w=10)['executed_actions'] == [0]
    # A must-run dimmable load may dim but never switch off.
    required = Device('required', True, True, 100, must_run=True,
                      supports_reduced=True, estimated_reduced_w=40)
    assert local_mask(required)[0] == [False, True, True]
    assert apply_shield([required], [1], max_import_w=50)['executed_actions'] == [2]
    assert apply_shield([required], [0], max_import_w=500)['executed_actions'] == [1]
    # A device that does not support dimming never receives level 2.
    assert local_mask(flexible)[0] == [True, True, False]
    assert apply_shield([flexible], [2], max_import_w=5000)['executed_actions'] == [1]
    try:
        validate_device(Device('bad', True, True, 100, supports_reduced=True,
                               estimated_reduced_w=200))
    except ValueError:
        pass
    else:
        raise AssertionError('Reduced power above ON power accepted')

    checked = 0
    for actions in product((0, 1, 2), repeat=3):
        ds = [protected, flexible, lock]
        r = apply_shield(ds, list(actions), max_import_w=600)
        assert all(mask[a] for mask, a in zip(r['local_legal_action_masks'], r['executed_actions']))
        assert r == apply_shield(ds, list(actions), max_import_w=600)
        checked += 1
    try:
        apply_shield([protected], [3], max_import_w=500)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid action level accepted')
    report = {'status':'PASS', 'module':'three_level_device_action_shield_v2',
              'action_levels':{'0':'off','1':'on','2':'reduced'},
              'exhaustive_three_device_action_vectors':checked,
              'tested':['protected loads','minimum off time','cycle protection','fault precedence',
                        'infeasible capacity','human override constrained by cap','solar offset',
                        'standby demand','state timers','determinism','invalid actions',
                        'reduced level offered only where supported',
                        'capacity pressure dims before it sheds',
                        'a must-run load may dim but never switch off',
                        'reduced power bounded by off and on power'],
              'scope':'simulation decision logic only', 'hardware_safety_certified':False,
              'full_simulator_ready':False, 'master_release_ready':False}
    root = Path(__file__).resolve().parents[1]
    p = root/'reports/sharp_action_shield_validation_v1.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2)); print('Report:', p)

if __name__ == '__main__':
    self_test()
