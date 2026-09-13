"""Assemble the SHARP Master Dataset release for private Kaggle upload.

Copies only approved, redistributable artifacts into release/, writes a manifest
with a SHA-256 for every file, a data dictionary, and a README that states what
each layer is and — just as importantly — what it is not.

Redistribution policy applied here:
  * REFIT raw traces are NOT redistributed. REFIT CLEANED is CC BY 4.0 but the
    8.4 GB source stays local; the release carries derived canonical tables only.
  * iAWE, TUS and eMARC raw microdata are NOT redistributed. Only derived,
    aggregated calibration products are included.
  * RESIDE-AC is CC0, so its derived thermal tables are safe to include.
  * NASA POWER is public domain.
The release is marked PRIVATE; publish only after checking each upstream licence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

RELEASE_VERSION = 'SHARP_MASTER_V1'

# (destination, source, layer, description, redistributable)
ITEMS = [
    ('rl_transitions/rl_transitions.parquet',
     'data/processed/sharp_rl_transitions_v1/rl_transitions.parquet',
     'rl_transitions', 'The RL training layer: (state, action, reward, next_state, done).', True),
    ('rl_transitions/feature_schema.json',
     'data/processed/sharp_rl_transitions_v1/feature_schema.json',
     'rl_transitions', 'Feature order, device padding and action semantics.', True),
    ('rl_transitions/episode_summary.csv',
     'data/processed/sharp_rl_transitions_v1/episode_summary.csv',
     'rl_transitions', 'Per-episode energy, cost, comfort and violation summary.', True),
    ('rl_transitions/transition_validation.json',
     'data/processed/sharp_rl_transitions_v1/transition_validation.json',
     'rl_transitions', 'Generation report, leakage gates and declared limits.', True),

    ('simulator_inputs/ap_households_with_splits_v1.parquet',
     'data/processed/appliance_inputs_v1/ap_households_with_splits_v1.parquet',
     'simulator_inputs', 'Andhra Pradesh household templates with disjoint splits.', True),
    ('simulator_inputs/device_power_models.parquet',
     'data/processed/simulator_devices_v1/unknown_quantity_one/baseline_power_v1/device_power_models.parquet',
     'simulator_inputs', 'Per-device power proxies and their provenance flags.', True),
    ('simulator_inputs/weekly_service_requests.parquet',
     'data/processed/simulator_devices_v1/unknown_quantity_one/service_plans_v1/weekly_service_requests.parquet',
     'simulator_inputs', 'Seasonal weekday service-hour requests per device.', True),
    ('simulator_inputs/preferred_service_slots.parquet',
     'data/processed/simulator_devices_v1/unknown_quantity_one/service_plans_v1/preferred_service_slots.parquet',
     'simulator_inputs', 'Preferred 15-minute service slots per device.', True),
    ('simulator_inputs/regional_grid_guntur_weather_15min_v1.parquet',
     'data/processed/simulator_context_v1/regional_grid_guntur_weather_15min_v1.parquet',
     'simulator_inputs', 'Guntur weather and southern-region grid context at 15 minutes.', True),
    ('simulator_inputs/context_manifest_v1.json',
     'data/processed/simulator_context_v1/context_manifest_v1.json',
     'simulator_inputs', 'Context build provenance and anti-leakage record.', True),

    ('calibration/reside_thermal_envelope_v1.json',
     'reports/reside_thermal_envelope_v1.json',
     'calibration', 'Observed RESIDE bounds used only to reject implausible parameters.', True),
    ('calibration/reside_year_hypothesis_v1.json',
     'reports/reside_year_hypothesis_v1.json',
     'calibration', 'Evidence that the RESIDE clock is May 2019, not May 2021.', True),
    ('calibration/rc_model_comparison.csv',
     'data/processed/reside_thermal_rc_v1/rc_model_comparison.csv',
     'calibration', 'Why a fitted thermal coefficient was rejected: it loses to persistence.', True),
    ('calibration/observed_temperature_response_pairs.parquet',
     'data/processed/reside_thermal_inputs_v1/observed_temperature_response_pairs.parquet',
     'calibration', 'RESIDE adjacent 15-minute temperature pairs (CC0 source).', True),

    ('configs/sharp_thermal_rc_v1.json',
     'configs/thermal/sharp_thermal_rc_v1.json',
     'configs', 'Declared thermal parameters, provenance tags and sensitivity scenarios.', True),
    ('configs/apcpdcl_2025_26_verified_components.json',
     'configs/tariffs/apcpdcl_2025_26_verified_components.json',
     'configs', 'APCPDCL FY2025-26 domestic LT tariff components.', True),
    ('configs/sources.yaml',
     'data_registry/sources.yaml',
     'configs', 'Master source registry with per-source status and open checks.', True),

    ('validation/sharp_thermal_rc_validation_v1.json',
     'reports/sharp_thermal_rc_validation_v1.json',
     'validation', 'Thermal component unit tests and envelope conformance.', True),
    ('validation/sharp_transition_core_validation_v1.json',
     'reports/sharp_transition_core_validation_v1.json',
     'validation', 'Transition-core integration self-test.', True),
    ('validation/sharp_action_shield_validation_v1.json',
     'reports/sharp_action_shield_validation_v1.json',
     'validation', 'Safety shield validation.', True),
    ('validation/sharp_reward_billing_validation_v1.json',
     'reports/sharp_reward_billing_validation_v1.json',
     'validation', 'Reward and billing validation.', True),
    ('validation/reside_ac_time_validation_v1.csv',
     'reports/reside_ac_time_validation_v1.csv',
     'validation', 'RESIDE time alignment across 11 houses.', True),
    ('validation/emarc_daily_validation_v1.csv',
     'reports/emarc_daily_validation_v1.csv',
     'validation', 'eMARC daily reconciliation summary (aggregate only).', True),
]

NOT_REDISTRIBUTED = {
    'data/raw/refit': 'REFIT CLEANED is CC BY 4.0 but the raw archive stays local; derived tables only.',
    'data/raw/iawe': 'iAWE raw microdata is not redistributed; only derived calibration statistics.',
    'data/raw/tus_india': 'MoSPI TUS microdata is licence-restricted and is not redistributed.',
    'data/raw/emarc': 'eMARC raw household load is not redistributed; aggregate validation only.',
    'data/raw/ires': 'IRES microdata is not redistributed; only derived household priors.',
    'data/raw/bee_clasp': 'BEE/CLASP report content is not redistributed; only encoded published statistics.',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build(root):
    release = root / 'release' / RELEASE_VERSION
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)

    manifest, missing, dictionary = [], [], []
    for destination, source, layer, description, redistributable in ITEMS:
        origin = root / source
        if not origin.exists():
            missing.append(source)
            continue
        if not redistributable:
            continue
        target = release / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        manifest.append({
            'path': destination, 'layer': layer, 'source_path': source,
            'bytes': target.stat().st_size, 'sha256': sha256(target)})
        dictionary.append({'path': destination, 'layer': layer,
                           'description': description})

    # Write explicit per-split files so nobody can train on test by accident.
    # rl_transitions.parquet stays the canonical table; these are exact subsets.
    import pandas as pd
    everything = pd.read_parquet(release / 'rl_transitions/rl_transitions.parquet')
    split_dir = release / 'rl_transitions/splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'validation', 'test']:
        subset = everything[everything.split.eq(split)].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f'{split} split is empty')
        target = split_dir / f'{split}.parquet'
        subset.to_parquet(target, index=False, compression='zstd')
        destination = f'rl_transitions/splits/{split}.parquet'
        manifest.append({'path': destination, 'layer': 'rl_transitions',
                         'source_path': 'derived from rl_transitions.parquet',
                         'bytes': target.stat().st_size, 'sha256': sha256(target)})
        dictionary.append({'path': destination, 'layer': 'rl_transitions',
                           'description': f'Exact {split} subset of rl_transitions.parquet '
                                          f'({len(subset):,} rows).'})
    recovered = sum(len(pd.read_parquet(split_dir / f'{s}.parquet'))
                    for s in ['train', 'validation', 'test'])
    if recovered != len(everything):
        raise ValueError('Split files do not reconstruct the canonical table')

    if missing:
        raise FileNotFoundError(
            'Release inputs are missing; nothing has been published:\n  '
            + '\n  '.join(missing))

    transitions = json.loads(
        (release / 'rl_transitions/transition_validation.json').read_text(encoding='utf-8'))

    manifest_document = {
        'release_version': RELEASE_VERSION,
        'visibility': 'PRIVATE',
        'geography': 'Guntur and Vijayawada, Andhra Pradesh, India',
        'control_interval_minutes': 15,
        'files': len(manifest),
        'total_bytes': sum(item['bytes'] for item in manifest),
        'layers': sorted({item['layer'] for item in manifest}),
        'rl_transitions': {
            'transitions': transitions['transitions'],
            'episodes': transitions['episodes'],
            'households': transitions['households'],
            'feature_count': transitions['feature_count'],
            'transitions_by_split': transitions['transitions_by_split'],
            'households_by_split': transitions['households_by_split'],
        },
        'gates': {
            'billing_reconciliation': transitions['billing_reconciliation'],
            'state_continuity': transitions['state_continuity'],
            'household_split_leakage': transitions['household_split_leakage'],
            'context_date_split_leakage': transitions['context_date_split_leakage'],
            'capacity_violation_steps': transitions['capacity_violation_steps'],
        },
        'not_redistributed': NOT_REDISTRIBUTED,
        'manifest': manifest,
    }
    (release / 'MANIFEST.json').write_text(
        json.dumps(manifest_document, indent=2), encoding='utf-8')
    (release / 'DATA_DICTIONARY.json').write_text(
        json.dumps(dictionary, indent=2), encoding='utf-8')

    readme = f"""# SHARP Master Dataset {RELEASE_VERSION}

Shielded Human-override Adaptive Reward Personalisation.
Smart-home demand response for Guntur / Vijayawada, Andhra Pradesh, under APCPDCL.
Control interval: 15 minutes. Visibility: PRIVATE.

## What to train on

`rl_transitions/rl_transitions.parquet` is the only layer BDQ should train on.
It holds {transitions['transitions']:,} transitions from {transitions['episodes']:,} episodes across
{transitions['households']} Andhra Pradesh household templates, with
{transitions['feature_count']} state features and up to 28 device branches.

Splits are household-disjoint AND date-disjoint:
{json.dumps(transitions['households_by_split'], indent=2)}

Public source datasets do NOT train the policy. They define, calibrate and
validate the simulator that generated these transitions.

## Verified gates

  billing reconciliation .......... {transitions['billing_reconciliation']}
  state / next-state continuity ... {transitions['state_continuity']}
  household split leakage ......... {transitions['household_split_leakage']}
  context date split leakage ...... {transitions['context_date_split_leakage']}
  capacity violation steps ........ {transitions['capacity_violation_steps']}

## What this dataset is NOT

  * Appliance power values are proxies and declared assumptions. They are NOT
    measured Indian appliance ratings.
  * The thermal model uses DECLARED parameters bounded by the observed RESIDE
    envelope. No causal AC cooling effect is established anywhere in this
    pipeline. A fitted coefficient was attempted and rejected because it lost
    to plain persistence in 7 of 11 houses.
  * REFIT is UK evidence used for appliance behaviour. It is NOT Indian data.
  * RESIDE-AC is 11 Hyderabad houses over 19 May-2019 days. It is NOT an Indian
    population distribution, and it is NOT Andhra Pradesh.
  * iAWE is a single New Delhi home. It is NOT a population.
  * The FY2025-26 APCPDCL tariff is applied as a scenario to other-year context.
    Experimental time-of-use multipliers are NOT official APCPDCL ToD tariffs.
  * There is no solar, battery or outage model, and no export compensation.
  * Human attention is not modelled, so all non-override feedback stays censored.
  * A successful training run on this data is not evidence of real-world
    performance, and nothing here is approved for hardware control decisions.

## Not redistributed

Raw microdata stays with its original provider. See `not_redistributed` in
MANIFEST.json. Derived, aggregated products are included instead.

## Integrity

Every file has a SHA-256 in MANIFEST.json. Verify after download.
"""
    (release / 'README.md').write_text(readme, encoding='utf-8')

    print('SHARP MASTER RELEASE BUILT')
    print('Version:', RELEASE_VERSION)
    print('Files:', len(manifest))
    print('Total MB:', round(manifest_document['total_bytes'] / 1024 / 1024, 2))
    print('Layers:', ', '.join(manifest_document['layers']))
    print('Transitions:', f"{transitions['transitions']:,}")
    print('Gates:', json.dumps(manifest_document['gates']))
    print('Output:', release)
    return manifest_document


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
