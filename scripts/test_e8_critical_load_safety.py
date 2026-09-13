"""E8 from the project spec: 10,000 attempts to shed a critical load.

The spec states the pass condition without softening it: "Must be exactly 0
violations." A safety shield that fails once in ten thousand is not a safety
shield, so this test has no tolerance and no statistical threshold.

Four independent attack routes are tried against the shield, because a shield
that only resists the obvious one is not evidence of anything:

  1. The POLICY asks to switch a critical load off.
  2. A HUMAN OVERRIDE asks to switch a critical load off. A user must not be
     able to override a safety constraint; that is the whole point of a shield.
  3. CAPACITY PRESSURE. The connection limit is made so tight that something
     must be shed, and the shield must shed a non-critical load instead.
  4. A MID-CYCLE INTERRUPT of a noninterruptible appliance.

Randomised device populations are drawn deterministically so a failure can be
reproduced exactly from the reported seed.
"""
from pathlib import Path
import argparse
import json
import numpy as np

from sharp_action_shield import Device, apply_shield, local_mask

ATTEMPTS = 10000


class Violation(AssertionError):
    pass


def make_population(rng):
    """A household with at least one critical load and some sheddable ones."""
    count = int(rng.integers(2, 12))
    devices = []
    critical_index = int(rng.integers(0, count))
    for index in range(count):
        critical = index == critical_index or bool(rng.random() < 0.2)
        on = bool(rng.random() < 0.7)
        cycle_active = bool(on and not critical and rng.random() < 0.2)
        devices.append(Device(
            device_id=f'd{index}', current_on=on, available=True,
            estimated_on_w=float(rng.integers(10, 2000)),
            must_run=critical and on,
            noninterruptible_cycle_active=cycle_active,
            elapsed_state_steps=int(rng.integers(0, 8)),
            min_on_steps=4 if cycle_active else 0, min_off_steps=0,
            shed_priority=0 if critical else 10))
    return devices, critical_index


def run(seed):
    rng = np.random.default_rng(seed)
    results = {'policy_shed_attempts': 0, 'override_shed_attempts': 0,
               'capacity_pressure_attempts': 0, 'cycle_interrupt_attempts': 0,
               'critical_loads_shed': 0, 'cycles_interrupted': 0,
               'infeasible_flagged': 0}
    failures = []

    for attempt in range(ATTEMPTS):
        devices, critical_index = make_population(rng)
        critical = devices[critical_index]
        route = attempt % 4

        if route == 0:
            requested = [0] * len(devices)
            human = None
            limit = 1e9
            results['policy_shed_attempts'] += 1
        elif route == 1:
            requested = [1] * len(devices)
            human = {critical.device_id: 0}
            limit = 1e9
            results['override_shed_attempts'] += 1
        elif route == 2:
            requested = [1] * len(devices)
            human = None
            # Tight enough that something must give.
            limit = max(1.0, float(critical.estimated_on_w) * 0.9)
            results['capacity_pressure_attempts'] += 1
        else:
            requested = [0] * len(devices)
            human = None
            limit = 1e9
            results['cycle_interrupt_attempts'] += 1

        shield = apply_shield(devices, requested, max_import_w=limit,
                              base_load_w=0.0, available_solar_w=0.0,
                              human_actions=human)
        executed = shield['executed_actions']

        for index, device in enumerate(devices):
            mask, _ = local_mask(device)
            if device.must_run and device.current_on and executed[index] == 0:
                results['critical_loads_shed'] += 1
                failures.append({'attempt': attempt, 'route': route,
                                 'device': device.device_id,
                                 'reason': 'critical load switched off'})
            if device.noninterruptible_cycle_active and executed[index] == 0:
                results['cycles_interrupted'] += 1
                failures.append({'attempt': attempt, 'route': route,
                                 'device': device.device_id,
                                 'reason': 'noninterruptible cycle interrupted'})
            if executed[index] == 1 and not mask[1]:
                failures.append({'attempt': attempt, 'route': route,
                                 'device': device.device_id,
                                 'reason': 'action taken outside the legal mask'})
        if not shield['capacity_feasible']:
            results['infeasible_flagged'] += 1

    return results, failures


def main(root, seed):
    results, failures = run(seed)
    passed = not failures
    report = {
        'status': 'E8_PASS' if passed else 'E8_FAIL',
        'test': 'critical load shed attempts',
        'attempts': ATTEMPTS,
        'seed': seed,
        'pass_condition': 'exactly zero violations',
        'attempt_routes': {
            'policy_requests_off': results['policy_shed_attempts'],
            'human_override_requests_off': results['override_shed_attempts'],
            'capacity_pressure': results['capacity_pressure_attempts'],
            'mid_cycle_interrupt': results['cycle_interrupt_attempts']},
        'critical_loads_shed': results['critical_loads_shed'],
        'noninterruptible_cycles_interrupted': results['cycles_interrupted'],
        'illegal_actions': len([f for f in failures
                                if 'legal mask' in f['reason']]),
        'infeasible_steps_flagged_not_hidden': results['infeasible_flagged'],
        'violations': len(failures),
        'first_failures': failures[:10],
        'note': ('An infeasible step means the shield could not satisfy the '
                 'connection limit without shedding something protected. It is '
                 'flagged and preserved as evidence, never resolved by shedding '
                 'a critical load.'),
    }
    path = root / 'reports/e8_critical_load_safety_v1.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print(f"E8 CRITICAL LOAD SAFETY: {report['status']}")
    print(f"  attempts: {ATTEMPTS:,} across 4 attack routes")
    for name, value in report['attempt_routes'].items():
        print(f'    {name}: {value:,}')
    print(f"  critical loads shed:            {report['critical_loads_shed']}")
    print(f"  noninterruptible cycles broken: {report['noninterruptible_cycles_interrupted']}")
    print(f"  illegal actions:                {report['illegal_actions']}")
    print(f"  infeasible steps flagged:       {report['infeasible_steps_flagged_not_hidden']:,}")
    print(f'  report: {path}')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--seed', type=int, default=20260914)
    a = p.parse_args()
    main(a.root.resolve(), a.seed)
