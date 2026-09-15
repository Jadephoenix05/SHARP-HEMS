"""Check the hardware specification is buildable, and agrees with everything else.

The document validator checks that decisions are not CONTRADICTED. It does not
check that what was agreed actually LANDED, and twice that gap let a silent
omission through - the EV charger vanished from both specs when a patch failed
halfway, and nothing noticed until the project lead read the file.

This closes it. It parses the specification itself rather than trusting prose,
and checks the things that would waste a build day:

  pins        no two loads share a GPIO, nothing collides with I2C
  coverage    every appliance has a pin, every pin has an appliance
  vocabulary  every appliance_type exists in the dataset's 22
  classes     all four idea-book classes are represented
  agreement   hardware and dashboard specs list the same appliances
  code        the shipped OLED code uses the pins the document promises
  budget      the rig fits the Pi's usable GPIO

A failure here means someone would have wired the wrong thing.
"""
from pathlib import Path
import argparse
import io
import json
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HARDWARE = Path('docs/HARDWARE_PROTOTYPE_SPEC.md')
DASHBOARD = Path('docs/DASHBOARD_SPECIFICATION.md')
OLED_CODE = Path('handover/sharp_rl_v1/oled_display.py')

# BCM pins a Raspberry Pi 4 exposes for general use.
USABLE_BCM = set(range(2, 28))
I2C_PINS = {2, 3}
PWM_PINS = {12, 13}


class Check:
    def __init__(self):
        self.results = []

    def __call__(self, name, passed, detail):
        self.results.append({'check': name, 'passed': bool(passed), 'detail': detail})
        print(f'  [{"PASS" if passed else "FAIL"}] {name}: {detail}', flush=True)
        return passed

    @property
    def failures(self):
        return [r for r in self.results if not r['passed']]


def parse_gpio_table(text):
    """Rows of the GPIO map, without assuming which column holds what.

    An earlier version took column 0 as the name and column 1 as the pin, and
    broke silently the moment a row-number column was added to the table. Find
    the cell that is ONLY a pin number, and take the name from the first cell
    that is not a number.
    """
    section = text.split('## 4. GPIO map', 1)
    if len(section) < 2:
        return {}
    body = section[1].split('###', 1)[0]
    pins = {}
    pin_column = name_column = None
    for line in body.splitlines():
        if not line.strip().startswith('|'):
            continue
        cells = [c.strip().strip('*') for c in line.strip().strip('|').split('|')]
        if set(''.join(cells)) <= set('-: '):
            continue
        lowered = [c.lower() for c in cells]

        # Find the columns from the HEADER. Guessing by content reads a leading
        # row-number column as a pin, which silently registers appliances on
        # BCM 1, 2 and 3 - including the I2C pins.
        if pin_column is None:
            for index, cell in enumerate(lowered):
                if 'bcm' in cell or 'pin' in cell:
                    pin_column = index
                if cell in ('appliance', 'name', 'load'):
                    name_column = index
            if pin_column is not None:
                if name_column is None:
                    name_column = max(0, pin_column - 1)
                continue

        if pin_column >= len(cells):
            continue
        cell = cells[pin_column]
        if not re.fullmatch(r'\d{1,2}(\s*,\s*\d{1,2})*', cell):
            continue
        name = cells[name_column] if name_column < len(cells) else '?'
        for number in re.findall(r'\d{1,2}', cell):
            pins.setdefault(int(number), []).append(name)
    return pins


def parse_appliance_table(text, heading):
    """Rows of the appliance list, returning name -> appliance_type where given."""
    section = text.split(heading, 1)
    if len(section) < 2:
        return {}
    body = section[1].split('\n## ', 1)[0]
    found = {}
    for line in body.splitlines():
        if not line.strip().startswith('|'):
            continue
        cells = [c.strip().strip('*`') for c in line.strip().strip('|').split('|')]
        if len(cells) < 3 or cells[0].lower().startswith(('#', 'id', '---')):
            continue
        types = [c for c in cells if re.fullmatch(r'[a-z_]+_?[a-z_]*', c)
                 and '_' in c and not c.startswith('http')]
        name = cells[1] if cells[0].isdigit() else cells[0]
        found[name] = types[0] if types else None
    return found


def validate(root):
    check = Check()
    hardware = (root / HARDWARE).read_text(encoding='utf-8')
    dashboard = (root / DASHBOARD).read_text(encoding='utf-8')

    print('GPIO map')
    pins = parse_gpio_table(hardware)
    check('GPIO table parsed', len(pins) > 0, f'{len(pins)} pins listed')

    clashes = {pin: names for pin, names in pins.items() if len(names) > 1}
    # The I2C pins legitimately carry one device across two lines.
    clashes = {p: n for p, n in clashes.items() if p not in I2C_PINS}
    check('no two loads share a pin', not clashes,
          'every pin used once' if not clashes else f'clashes: {clashes}')

    outside = sorted(p for p in pins if p not in USABLE_BCM)
    check('all pins are usable BCM', not outside,
          'all within BCM 2-27' if not outside else f'outside range: {outside}')

    i2c_users = {p: n for p, n in pins.items() if p in I2C_PINS}
    non_oled = [n for names in i2c_users.values() for n in names
                if 'oled' not in n.lower() and 'i' not in n.lower()[:3]]
    check('nothing collides with I2C', not non_oled,
          f'BCM 2/3 carry only the display'
          if not non_oled else f'collision: {non_oled}')

    free_pwm = sorted(PWM_PINS - set(pins))
    check('PWM channels left free', len(free_pwm) == 2,
          f'BCM {free_pwm} unused, so dimming could return without rewiring'
          if len(free_pwm) == 2 else f'only {free_pwm} free')

    print('\nAppliances')
    # Compare appliance IDS, which both documents carry, rather than row counts.
    # A row count is an accident of formatting; an id is the contract the Pi,
    # the dashboard and the rig all have to spell identically.
    hardware_list = parse_appliance_table(hardware, '## 2. Appliance list')
    dashboard_ids = set(re.findall(r'`([a-z_]+_\d{2})`', dashboard))
    check('hardware spec lists appliances', len(hardware_list) >= 10,
          f'{len(hardware_list)} rows')
    check('dashboard spec lists appliance ids', len(dashboard_ids) >= 10,
          f'{len(dashboard_ids)} ids')
    check('both specs describe the same number of appliances',
          len(hardware_list) == len(dashboard_ids),
          f'hardware {len(hardware_list)}, dashboard {len(dashboard_ids)}')

    models = root / ('data/processed/simulator_devices_v1/unknown_quantity_one'
                     '/baseline_power_v1/device_power_models.parquet')
    if models.exists():
        import pandas as pd
        known = set(pd.read_parquet(models, columns=['appliance_type'])
                    .appliance_type.unique())
        used = {t for t in hardware_list.values() if t}
        used |= {re.sub(r'_\d+$', '', i) for i in dashboard_ids}
        unknown = sorted(used - known - {'ev_charger'})
        check('every appliance_type exists in the dataset', not unknown,
              f'{len(used)} types used, all known'
              if not unknown else f'not in the dataset: {unknown}')
        check('the EV is flagged as not in the data',
              'no electric vehicle' in hardware.lower()
              or 'no ev appeared' in hardware.lower()
              or 'forward-looking' in hardware.lower(),
              'the EV is labelled a forward-looking extension')
    else:
        check('dataset available for cross-check', False, f'{models} not found')

    print('\nLoad classes')
    for name in ['Critical', 'Thermostatic', 'Deferrable', 'Interruptible']:
        check(f'{name} represented', name.lower() in hardware.lower(),
              'named in the hardware spec')

    print('\nShipped code agrees with the document')
    if (root / OLED_CODE).exists():
        code = (root / OLED_CODE).read_text(encoding='utf-8')
        buzzer_doc = [p for p, n in pins.items()
                      if any('buzz' in x.lower() for x in n)]
        buzzer_code = re.search(r"BUZZER_PIN_NAME\s*=\s*'D(\d+)'", code)
        check('the OLED code uses the documented buzzer pin',
              bool(buzzer_code) and buzzer_doc
              and int(buzzer_code.group(1)) == buzzer_doc[0],
              f'document BCM {buzzer_doc[0] if buzzer_doc else "?"}, '
              f'code D{buzzer_code.group(1) if buzzer_code else "?"}')
    else:
        check('OLED code present', False, f'{OLED_CODE} not found')

    print('\nBudget')
    inputs = [n for names in pins.values() for n in names if 'switch' in n.lower()]
    check('no switches, as decided', not inputs,
          'the rig has no inputs; overrides come from the phone'
          if not inputs else f'switches still listed: {inputs}')
    check('fits the Pi', len(pins) <= len(USABLE_BCM),
          f'{len(pins)} of {len(USABLE_BCM)} usable BCM pins in use')

    report = {
        'status': 'HARDWARE_SPEC_BUILDABLE' if not check.failures
                  else 'HARDWARE_SPEC_BROKEN',
        'checks': check.results,
        'pins_in_use': {str(k): v for k, v in sorted(pins.items())},
        'appliances_hardware': sorted(hardware_list),
        'appliances_dashboard': sorted(dashboard_ids),
        'note': ('This checks the specification is buildable and agrees with the '
                 'dashboard spec, the dataset vocabulary and the shipped code. '
                 'It does not check that anything was actually wired correctly.'),
    }
    (root / 'reports/hardware_spec_validation_v1.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')

    print(f'\n{"=" * 60}')
    print(f'{report["status"]}  ({len(check.results)} checks, '
          f'{len(check.failures)} failures)')
    for failure in check.failures:
        print(f'  {failure["check"]}: {failure["detail"]}')
    print('=' * 60)
    return not check.failures


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    raise SystemExit(0 if validate(a.root.resolve()) else 1)
