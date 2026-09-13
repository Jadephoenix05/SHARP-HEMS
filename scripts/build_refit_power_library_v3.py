"""Split-aware REFIT appliance power, built on the project's existing rules.

This supersedes build_refit_appliance_power_library_v2.py, which had two faults
that this version fixes:

  1. LEAKAGE. v2 pooled all 20 REFIT houses into one number per appliance type
     and applied it to every split. REFIT houses carry their own split
     assignment (14 train, 3 validation, 3 test) in household_splits_v1.csv, so
     that let test-split REFIT houses set the power used by train households.
     v3 computes statistics per split from that split's REFIT houses only,
     exactly as route_ap_appliance_sources.py already required.

  2. MIXED SITES. v2 matched names by regular expression and so swept in
     "Television Site" and "Computer Site", which are multi-appliance plug
     sites, not single appliances. build_refit_trace_library.py deliberately
     excluded those with an exact-label whitelist. v3 reuses that whitelist
     unchanged.

Result: fewer grounded devices than v2 claimed, but every grounded value comes
from a single-appliance channel in the SAME split. Coverage is worth less than
correctness here.

Power is the MEDIAN of 15-minute mean power over intervals above the 5 W
threshold. For 15-minute energy accounting that is the right quantity: it
already embeds within-interval cycling. It is not a nameplate rating.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ON_THRESHOLD_W = 5.0

# Exact source labels only, reused verbatim from build_refit_trace_library.py.
# Mixed appliance sites such as "Television Site" stay excluded on purpose.
LABELS = {
    'Fridge': 'refrigerator',
    'Fridge(garage)': 'refrigerator',
    'Washing Machine': 'washing_machine',
    'Washing Machine (1)': 'washing_machine',
    'Washing Machine (2)': 'washing_machine',
    'Kettle': 'electric_kettle',
    'Television': 'television',
    'Router': 'modem_router',
    'Desktop Computer': 'desktop',
}

EXCLUDED_ON_PURPOSE = {
    'Television Site': 'Multi-appliance plug site, not a single television.',
    'Computer Site': 'Multi-appliance plug site, not a single computer.',
    'Fridge-Freezer': 'Combined unit; the project whitelist keeps Fridge exact.',
    'Freezer': 'A freezer is not a household refrigerator category here.',
    'Magimix (Blender)': 'Not an Indian mixer grinder; only two weak channels.',
    'Food Mixer': 'Same reason.',
    'Pond Pump': 'A pond pump is not a domestic water pump.',
    'Lamp (80Watts)': 'One channel cannot ground an Indian lighting mix.',
}

MINIMUM_ON_INTERVALS = 96  # at least one day of measurable operation


def check(condition, message):
    if not condition:
        raise ValueError(message)


def build(root):
    splits = pd.read_csv(root / 'data/processed/appliance_inputs_v1/household_splits_v1.csv',
                         dtype=str)
    refit_splits = splits[splits.source.eq('REFIT')]
    check(not refit_splits.household_id.duplicated().any(),
          'Duplicate REFIT split assignment')
    split_of_house = {int(r.household_id): r.split for r in refit_splits.itertuples()}
    print('REFIT house splits:',
          pd.Series(list(split_of_house.values())).value_counts().to_dict(), '\n')

    files = sorted((root / 'data/processed/refit_canonical_v1').glob('*.parquet'))
    check(len(files) == 20, f'Expected 20 REFIT house files, found {len(files)}')

    samples = {}
    channel_rows = []
    for path in files:
        frame = pd.read_parquet(path, columns=[
            'source_house_id', 'channel_id', 'appliance_name_source',
            'power_w', 'power_missing', 'power_nonfinite'])
        house = int(frame.source_house_id.iloc[0])
        split = split_of_house.get(house)
        check(split is not None, f'REFIT house {house} has no split assignment')
        frame = frame[frame.appliance_name_source.isin(LABELS)]
        if frame.empty:
            continue
        for channel, group in frame.groupby('channel_id'):
            appliance = LABELS[group.appliance_name_source.iloc[0]]
            good = group[(~group.power_missing) & (~group.power_nonfinite)]
            power = good.power_w.to_numpy(float)
            power = power[np.isfinite(power) & (power >= 0)]
            if len(power) < MINIMUM_ON_INTERVALS:
                continue
            on = power > ON_THRESHOLD_W
            if on.sum() < MINIMUM_ON_INTERVALS:
                continue
            samples.setdefault((appliance, split), []).append(power[on])
            channel_rows.append({
                'appliance_type': appliance, 'split': split, 'refit_house': house,
                'refit_channel': int(channel),
                'refit_name': group.appliance_name_source.iloc[0],
                'intervals': int(len(power)), 'on_intervals': int(on.sum()),
                'duty_fraction': float(on.mean()),
                'median_on_power_w': float(np.median(power[on]))})

    check(channel_rows, 'No REFIT channels matched the exact-label whitelist')

    library = {}
    for (appliance, split), pooled in sorted(samples.items()):
        values = np.concatenate(pooled)
        channels = [r for r in channel_rows
                    if r['appliance_type'] == appliance and r['split'] == split]
        library.setdefault(appliance, {})[split] = {
            'refit_channels': len(channels),
            'refit_houses': sorted({r['refit_house'] for r in channels}),
            'on_intervals': int(len(values)),
            'median_on_power_w': float(np.median(values)),
            'p25_on_power_w': float(np.quantile(values, 0.25)),
            'p75_on_power_w': float(np.quantile(values, 0.75)),
            'duty_fraction_mean': float(np.mean([r['duty_fraction'] for r in channels])),
        }

    print(f'{len(channel_rows)} single-appliance channels matched\n')
    print(f"{'appliance':18s} {'split':11s} {'chans':>5s} {'median W':>9s} {'duty':>6s}")
    for appliance in sorted(library):
        for split in ['train', 'validation', 'test']:
            entry = library[appliance].get(split)
            if entry:
                print(f"{appliance:18s} {split:11s} {entry['refit_channels']:5d} "
                      f"{entry['median_on_power_w']:9.1f} "
                      f"{entry['duty_fraction_mean']:6.3f}")
            else:
                print(f'{appliance:18s} {split:11s}     -         -      - '
                      f' (no same-split channel; keeps declared assumption)')

    out = root / 'data/processed/refit_power_library_v3'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(channel_rows).to_csv(out / 'refit_channel_statistics.csv', index=False)

    covered = {a: sorted(library[a]) for a in sorted(library)}
    report = {
        'status': 'REFIT_SPLIT_AWARE_POWER_LIBRARY_BUILT',
        'supersedes': 'refit_power_library_v2 (pooled across splits, matched mixed sites)',
        'on_threshold_w': ON_THRESHOLD_W,
        'exact_label_whitelist': LABELS,
        'excluded_on_purpose': EXCLUDED_ON_PURPOSE,
        'refit_house_splits': {str(k): v for k, v in sorted(split_of_house.items())},
        'channels_matched': len(channel_rows),
        'appliance_split_coverage': covered,
        'library': library,
        'leakage_control': (
            'Statistics for a split use only REFIT houses assigned to that split, '
            'matching the same-split rule in route_ap_appliance_sources.py.'),
        'limitations': [
            'REFIT is 20 UK homes. These are not Indian appliance measurements.',
            'A REFIT zero is not a confirmed physical OFF, so ON means measurable power.',
            'Median ON power is a 15-minute mean, not a nameplate rating.',
            'Air conditioning, fans, coolers, geysers, pumps, lighting, irons, rice '
            'cookers and purifiers have no REFIT equivalent and keep their assumptions.',
            'Some appliance and split combinations have no channel at all and '
            'therefore keep their declared assumption in that split.',
        ],
        'mapping_sha256': hashlib.sha256(
            (root / 'data/registry/refit_appliance_mapping.csv').read_bytes()).hexdigest(),
    }
    (out / 'refit_power_library.json').write_text(json.dumps(report, indent=2),
                                                  encoding='utf-8')
    print(f'\nREFIT SPLIT-AWARE POWER LIBRARY BUILT')
    print(f'  appliance types grounded: {len(library)}')
    print(f'  single-appliance channels: {len(channel_rows)}')
    print('  output:', out)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
