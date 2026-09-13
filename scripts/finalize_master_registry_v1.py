"""Record the V1 release state in the master source registry.

Marks each source with what it actually contributed and whether its DERIVED
products are cleared for the private release. A source is release_ready only if
its derived product is included in the release and its open semantic checks do
not affect that product. Raw microdata redistribution is a separate question and
stays false everywhere.

This never marks a source "validated" that still has open checks; it records the
open checks alongside the release decision.
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import shutil
import yaml

RELEASE_VERSION = 'SHARP_MASTER_V1'

UPDATES = {
    'refit_cleaned': {
        'release_contribution': 'Canonical 15-minute appliance traces and duty-cycle behaviour.',
        'derived_product_in_release': False,
        'raw_redistributed': False,
        'geography_warning': 'United Kingdom. Never present REFIT as Indian household data.',
    },
    'nasa_power_guntur': {
        'release_contribution': 'Guntur 15-minute weather forcing and observation streams.',
        'derived_product_in_release': True,
        'raw_redistributed': True,
        'notes_v1': ['Public domain reanalysis.',
                     'LST equals UTC+5 at this longitude, verified against a UTC pull.'],
    },
    'aperc_apcpdcl_tariff': {
        'release_contribution': 'Domestic LT energy charge configuration for the reward.',
        'derived_product_in_release': True,
        'raw_redistributed': False,
        'notes_v1': ['FY2025-26 slabs applied as a scenario to other-year context.',
                     'Experimental time-of-use multipliers are NOT official APCPDCL ToD tariffs.'],
    },
    'reside_ac': {
        'release_contribution': 'Plausibility bounds for indoor temperature dynamics only.',
        'derived_product_in_release': True,
        'raw_redistributed': True,
        'thermal_parameter_source': False,
        'notes_v1': ['CC0, so derived thermal tables are redistributable.',
                     'Clock year resolved to May 2019 by pre-registered weather correlation.',
                     'A fitted thermal coefficient was attempted and REJECTED: it lost to persistence in 7 of 11 houses.',
                     'Hyderabad, not Andhra Pradesh. Eleven houses is not a population.'],
    },
    'ires_2020': {
        'release_contribution': 'Andhra Pradesh household templates and appliance ownership priors.',
        'derived_product_in_release': True,
        'raw_redistributed': False,
    },
    'bee_clasp_2024': {
        'release_contribution': 'Modern Indian appliance ownership priors from published tables.',
        'derived_product_in_release': False,
        'raw_redistributed': False,
    },
    'iawe': {
        'release_contribution': 'Appliance power proxies for a subset of device types.',
        'derived_product_in_release': True,
        'raw_redistributed': False,
        'notes_v1': ['One New Delhi home. NOT an Indian population distribution.',
                     'Proxy wattages are not measured appliance ratings.'],
    },
    'tus_2024': {
        'release_contribution': 'Occupancy and activity priors behind service schedules.',
        'derived_product_in_release': False,
        'raw_redistributed': False,
        'notes_v1': ['MoSPI microdata is licence restricted and is not redistributed.'],
    },
    'emarc': {
        'release_contribution': 'Indian aggregate-load calibration and held-out validation.',
        'derived_product_in_release': True,
        'raw_redistributed': False,
    },
    'grid_india_historical': {
        'release_contribution': 'Southern-region demand, percentile and peak severity context.',
        'derived_product_in_release': True,
        'raw_redistributed': False,
        'notes_v1': ['Peak thresholds are fitted on training years only.'],
    },
    'iced_hourly': {
        'release_contribution': 'Not acquired for V1; grid context came from Grid-India instead.',
        'derived_product_in_release': False,
        'raw_redistributed': False,
    },
}


def build(root):
    registry_path = root / 'data_registry/sources.yaml'
    transitions_path = (root / 'data/processed/sharp_rl_transitions_v1'
                        / 'transition_validation.json')
    if not transitions_path.exists():
        raise FileNotFoundError('Generate the RL transitions before finalizing the registry.')
    transitions = json.loads(transitions_path.read_text(encoding='utf-8'))

    sources = yaml.safe_load(registry_path.read_text(encoding='utf-8'))
    if not isinstance(sources, dict):
        raise ValueError('Source registry must be a YAML mapping.')
    for key in UPDATES:
        if key not in sources:
            raise KeyError(f'Missing registry entry: {key}')

    backup = registry_path.with_name(
        'sources_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.yaml')
    shutil.copy2(registry_path, backup)

    for key, update in UPDATES.items():
        entry = sources[key]
        entry.update(update)
        open_checks = entry.get('remaining_checks') or []
        # A derived product ships only when it is actually in the release.
        entry['release_ready'] = bool(update['derived_product_in_release'])
        entry['release_version'] = RELEASE_VERSION if entry['release_ready'] else None
        entry['open_semantic_checks'] = len(open_checks)
        entry['directly_trains_bdq'] = False

    sources['sharp_rl_transitions'].update({
        'status': 'GENERATED_AND_VALIDATED',
        'role': 'rl_training_layer',
        'geography': 'Guntur and Vijayawada Andhra Pradesh',
        'processed_path': 'data/processed/sharp_rl_transitions_v1/rl_transitions.parquet',
        'validation_report': 'data/processed/sharp_rl_transitions_v1/transition_validation.json',
        'transitions': transitions['transitions'],
        'episodes': transitions['episodes'],
        'households': transitions['households'],
        'feature_count': transitions['feature_count'],
        'directly_trains_bdq': True,
        'release_ready': True,
        'release_version': RELEASE_VERSION,
        'gates': {
            'billing_reconciliation': transitions['billing_reconciliation'],
            'state_continuity': transitions['state_continuity'],
            'household_split_leakage': transitions['household_split_leakage'],
            'context_date_split_leakage': transitions['context_date_split_leakage'],
            'capacity_violation_steps': transitions['capacity_violation_steps'],
        },
        'remaining_checks': [
            'Appliance power values remain proxies, not measured Indian ratings.',
            'Thermal parameters are declared assumptions, not fitted physics.',
            'No solar, battery, outage or human-attention model.',
        ],
    })

    temporary = registry_path.with_suffix('.yaml.partial')
    temporary.write_text(yaml.safe_dump(sources, sort_keys=False, allow_unicode=True),
                         encoding='utf-8')
    temporary.replace(registry_path)

    ready = sorted(k for k, v in sources.items()
                   if isinstance(v, dict) and v.get('release_ready'))
    print('MASTER REGISTRY FINALIZED')
    print('Backup:', backup.name)
    print('Release version:', RELEASE_VERSION)
    print(f'Release-ready entries: {len(ready)} / {len(sources)}')
    for key in ready:
        print('  ', key)
    print('RL transitions:', f"{transitions['transitions']:,}",
          '| households:', transitions['households'])
    return sources


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
