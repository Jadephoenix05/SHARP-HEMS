"""Account for every input to the dataset as source-derived, calibrated or assumed.

The point is to be able to answer "how much of this is real data?" with a number
rather than a feeling. Each component is classified once, honestly:

  SOURCE_REPORTED   a value a real dataset actually records for that entity
  SOURCE_DERIVED    computed from recorded values without adding information
  TRANSFER          measured, but on a different population (REFIT UK, iAWE Delhi)
  INFERRED          recovered from a reported value through a documented model
  ASSUMPTION        chosen by us, bounded where possible
  SYNTHETIC         generated from a stated rule, with no recorded counterpart

Nothing is upgraded by being useful. A REFIT median is TRANSFER however well it
behaves, and override evidence is SYNTHETIC however plausible it looks.
"""
from pathlib import Path
import argparse
import json
import pandas as pd

COMPONENTS = [
    ('household population', 'SOURCE_REPORTED', 'IRES 2020',
     '498 Andhra Pradesh households with their reported characteristics.'),
    ('appliance ownership and counts', 'SOURCE_REPORTED', 'IRES 2020',
     'Which appliances each household owns, and how many.'),
    ('sanctioned connection load', 'SOURCE_REPORTED', 'IRES 2020',
     'The kW limit each household is connected at.'),
    ('grid supply hours and outage duration', 'SOURCE_REPORTED', 'IRES 2020',
     '358 of 498 households report under 24 h supply; minimum 12 h.'),
    ('inverter battery ownership', 'SOURCE_REPORTED', 'IRES 2020',
     '35 of 498 households own an inverter battery.'),
    ('solar home system ownership', 'SOURCE_REPORTED', 'IRES 2020',
     '3 of 498, at 20-25 W. Essentially no rooftop PV in this population.'),
    ('reported monthly bill', 'SOURCE_REPORTED', 'IRES 2020',
     'Used to place each household on the tariff curve.'),
    ('AC and pump daily usage hours', 'SOURCE_REPORTED', 'IRES 2020',
     'Seasonal reported hours, where the household answered.'),
    ('tariff slabs and charges', 'SOURCE_REPORTED', 'APERC/APCPDCL FY2025-26 order',
     'Verified against the official order with page numbers and PDF SHA-256.'),
    ('outdoor weather and irradiance', 'SOURCE_REPORTED', 'NASA POWER',
     'Guntur 15-minute weather; LST to IST offset empirically verified.'),
    ('regional grid demand', 'SOURCE_REPORTED', 'Grid-India historical',
     'Southern-region demand; peak thresholds fitted on training years only.'),
    ('indoor temperature envelope bounds', 'SOURCE_REPORTED', 'RESIDE-AC',
     '20,041 observed 15-minute temperature pairs, used only to reject parameters.'),
    ('occupancy diaries', 'SOURCE_REPORTED', 'MoSPI TUS 2024',
     'Adult location diaries behind the at-home proxy.'),

    ('grid peak severity', 'SOURCE_DERIVED', 'Grid-India',
     'Percentile of reported demand within training years.'),
    ('billing and cost', 'SOURCE_DERIVED', 'APCPDCL order',
     'Telescopic slabs applied to simulated consumption; reconciled every episode.'),
    ('occupancy at-home fraction', 'SOURCE_DERIVED', 'TUS 2024',
     'One adult per household, not whole-household presence.'),

    ('appliance power, 512 devices', 'TRANSFER', 'REFIT CLEANED (UK)',
     'Median ON power from same-split single-appliance channels. UK measurement.'),
    ('appliance power, 173 devices', 'TRANSFER', 'iAWE (one Delhi home)',
     'Median above-threshold power. One household, not a population.'),

    ('monthly consumption position', 'INFERRED', 'IRES bill + APCPDCL tariff',
     'Tariff inverted from the reported bill. Not a metered reading.'),

    ('appliance power, 3,439 devices', 'ASSUMPTION', 'engineering values',
     'No REFIT or iAWE equivalent exists for fans, lighting, geysers, pumps.'),
    ('thermal time constant and cooling', 'ASSUMPTION', 'declared, envelope-bounded',
     'A fitted RESIDE coefficient was attempted and rejected: it lost to persistence.'),
    ('battery capacity and efficiency', 'ASSUMPTION', 'declared',
     'Typical inverter unit; IRES reports ownership but not capacity.'),
    ('outage placement within the day', 'ASSUMPTION', 'declared',
     'Duration is reported; the pattern code 1-5 has no published meaning.'),
    ('comfort band and setpoint', 'ASSUMPTION', 'policy choice',
     'RESIDE house metadata holds homeowner setpoints and is the future source.'),
    ('rooftop PV overlay', 'ASSUMPTION', 'scenario, default off',
     'Off by default precisely because IRES shows it does not exist here.'),

    ('override events', 'SYNTHETIC', 'stated behavioural rule',
     'No SHARP source records a real demand-response override; none exists yet.'),
    ('attention', 'SYNTHETIC', 'occupancy threshold',
     'Derived from the occupancy proxy by a declared rule.'),
    ('service schedules', 'SYNTHETIC', 'survey durations, synthetic clock',
     'Reported daily hours do not say WHEN an appliance runs.'),
]


def build(root):
    tiers = {}
    for name, tier, source, note in COMPONENTS:
        tiers.setdefault(tier, []).append(
            {'component': name, 'source': source, 'note': note})

    devices = pd.read_parquet(
        root / 'data/processed/simulator_devices_v1/unknown_quantity_one'
        / 'baseline_power_v1/device_power_models.parquet')
    basis = devices.power_parameter_basis.value_counts().to_dict()
    grounded = int(devices.power_parameter_basis.str.contains('REFIT|IAWE').sum())

    transitions = root / 'data/processed/sharp_rl_transitions_v2'
    report = json.loads((transitions / 'transition_validation.json').read_text(encoding='utf-8'))

    summary = {
        'status': 'DATASET_PROVENANCE_ACCOUNTED',
        'tier_counts': {k: len(v) for k, v in sorted(tiers.items())},
        'tiers': tiers,
        'appliance_power_devices': {
            'total': int(len(devices)),
            'measurement_grounded': grounded,
            'measurement_grounded_fraction': round(grounded / len(devices), 4),
            'by_basis': basis},
        'headline': (
            'Household population, appliance ownership, connection load, outage '
            'hours, inverter and solar ownership, tariff, weather, grid demand '
            'and occupancy diaries are all REAL reported data. Appliance power '
            'is measured but TRANSFERRED from UK and Delhi homes for 685 of '
            '4,124 devices and ASSUMED for the rest. Override behaviour and '
            'service clock times are SYNTHETIC because no source records them.'),
        'what_would_reduce_synthetic_content': [
            'A real deployment logging user overrides would replace the synthetic '
            'override rule with observed preference pairs. That is the whole point '
            'of the Raspberry Pi stage.',
            'Indian appliance-level metering at scale would replace REFIT transfer '
            'power. None is publicly available; iAWE is one home.',
            'RESIDE house metadata records homeowner AC setpoints and could replace '
            'the assumed comfort setpoint.',
            'A decoded IRES power-cut pattern codebook would replace assumed outage '
            'placement with reported placement.',
        ],
        'transitions': report['transitions'],
        'override_events_synthetic': report['override_events'],
    }
    path = root / 'reports/dataset_provenance_v1.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('DATASET PROVENANCE')
    for tier in ['SOURCE_REPORTED', 'SOURCE_DERIVED', 'TRANSFER', 'INFERRED',
                 'ASSUMPTION', 'SYNTHETIC']:
        items = tiers.get(tier, [])
        print(f'\n{tier}  ({len(items)} components)')
        for item in items:
            print(f"  - {item['component']}  [{item['source']}]")
    print('\nAppliance power grounding:')
    for k, v in basis.items():
        print(f'  {v:5d}  {k}')
    print(f"\n  measurement-grounded: {grounded} / {len(devices)} devices "
          f"({100 * grounded / len(devices):.1f}%)")
    print('\nReport:', path)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
