"""Compare SHARP's simulated household load against real Indian metered load.

This is the only check in the project that tests EXTERNAL realism. Everything
else verifies internal consistency: that the simulator obeys its own rules. This
asks a different question - does the load it produces look like load that real
Indian households actually draw?

Reference: eMARC mainline (whole-household) meters, 6.77 million 15-minute
readings from 140 households across Pune, Aurangabad, Kanpur rural and Gonda,
2018-2020.

WHAT THIS IS NOT. eMARC is Maharashtra and Uttar Pradesh, not Andhra Pradesh,
and 2018-2020 rather than 2021-2024. It is a CALIBRATION REFERENCE for whether
the shape and scale of simulated load are plausible for Indian homes, not a
held-out test set and not a ground truth for Andhra Pradesh.

Block-to-clock mapping, previously an open question in the registry, is settled
empirically here: the mainline profile peaks at 07:45 and 21:00 with a trough at
15:45, the textbook Indian residential double peak, which only makes sense if
block 0 is local midnight.

EXPECTED DISAGREEMENT. SHARP models only the appliances in its inventory and
carries no hidden background load, so its aggregate MUST come in below a real
mainline meter. Quantifying that gap is the point; a perfect match would mean
something was wrong.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

STEP_HOURS = 0.25


def check(condition, message):
    if not condition:
        raise ValueError(message)


def emarc_profiles(root):
    blocks = pd.read_parquet(
        root / 'data/interim/emarc/emarc_load_blocks_native_v1.parquet',
        columns=['block', 'load_kw', 'meter_role', 'household_id',
                 'household_type_source', 'source_date'])
    mains = blocks[blocks.meter_role.eq('mainline')].copy()
    check(len(mains) > 1_000_000, 'Too few eMARC mainline readings')
    check(int(mains.block.min()) == 0 and int(mains.block.max()) == 95,
          'eMARC blocks are not a 0-95 fifteen-minute grid')

    # Stratify before comparing. eMARC's pooled morning peak is driven by its
    # water-heater households; the Andhra Pradesh sample has 13 geysers in 498,
    # so it is a "Basic" population and must be compared with Basic homes.
    # Comparing against the pooled profile made SHARP look wrong for a reason
    # that was really a population difference.
    basic = mains[mains.household_type_source.eq('Basic')]
    check(len(basic) > 100_000, 'Too few eMARC Basic-household readings')
    profile = basic.groupby('block').load_kw.mean()
    peak_block, trough_block = int(profile.idxmax()), int(profile.idxmin())
    pooled = mains.groupby('block').load_kw.mean()

    daily = (basic.groupby(['household_id', 'source_date'])
             .agg(kwh=('load_kw', lambda s: float(s.sum()) * STEP_HOURS),
                  peak_kw=('load_kw', 'max'),
                  mean_kw=('load_kw', 'mean'),
                  readings=('load_kw', 'size')).reset_index())
    daily = daily[daily.readings.eq(96)]
    check(len(daily) > 1000, 'Too few complete eMARC household-days')
    daily['par'] = daily.peak_kw / daily.mean_kw.replace(0, np.nan)

    by_type = (mains.groupby(['household_type_source', 'block']).load_kw.mean()
               .unstack(0))
    return profile, daily, by_type, peak_block, trough_block, int(pooled.idxmax())


def sharp_profiles(root):
    d = pd.read_parquet(
        root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet',
        columns=['episode_id', 'step_id', 'household_id', 'policy',
                 'aggregate_power_kw', 'grid_absent'])
    # Compare like with like: eMARC meters record what the household actually
    # drew under its own behaviour, so use the schedule-following policy rather
    # than the exploratory ones.
    d = d[d.policy.eq('serve_preferred')]
    check(len(d) > 0, 'No serve_preferred rows to compare')
    profile = d.groupby('step_id').aggregate_power_kw.mean()
    daily = (d.groupby(['episode_id', 'household_id'])
             .agg(kwh=('aggregate_power_kw', lambda s: float(s.sum()) * STEP_HOURS),
                  peak_kw=('aggregate_power_kw', 'max'),
                  mean_kw=('aggregate_power_kw', 'mean'),
                  steps=('aggregate_power_kw', 'size'),
                  outage_steps=('grid_absent', 'sum')).reset_index())
    daily = daily[daily.steps.eq(96)]
    daily['par'] = daily.peak_kw / daily.mean_kw.replace(0, np.nan)
    return profile, daily


def build(root):
    print('Reading eMARC mainline meters...', flush=True)
    (e_profile, e_daily, e_by_type, peak_block, trough_block,
     pooled_peak) = emarc_profiles(root)
    print('Reading SHARP simulated load...', flush=True)
    s_profile, s_daily = sharp_profiles(root)

    def clock(block):
        return f'{block // 4:02d}:{15 * (block % 4):02d}'

    # Shape agreement, on normalised profiles so scale does not dominate.
    e_shape = (e_profile / e_profile.mean()).to_numpy()
    s_shape = (s_profile / s_profile.mean()).to_numpy()
    correlation = float(np.corrcoef(e_shape, s_shape)[0, 1])
    s_peak, s_trough = int(s_profile.idxmax()), int(s_profile.idxmin())

    # Evening peak window, the period demand response actually targets.
    evening = slice(72, 88)  # 18:00-22:00
    e_evening = float(e_profile.iloc[evening].mean() / e_profile.mean())
    s_evening = float(s_profile.iloc[evening].mean() / s_profile.mean())

    report = {
        'status': 'AGGREGATE_LOAD_COMPARED_TO_EMARC',
        'reference': {
            'source': 'eMARC mainline meters, Prayas Energy Group',
            'stratum': 'Basic households only (no air conditioner, no water heater)',
            'why_this_stratum': (
                'The Andhra Pradesh sample has 13 geysers in 498 households and '
                '51 air conditioners, so it is a Basic population. eMARC pooled '
                'across all types peaks at 07:45 because its water-heater homes '
                'do; Basic homes peak at 21:15, which is the fair comparison.'),
            'pooled_peak_clock_for_contrast': f'{pooled_peak // 4:02d}:'
                                              f'{15 * (pooled_peak % 4):02d}',
            'readings': int(len(e_daily)) and int(e_daily.readings.sum()),
            'household_days': int(len(e_daily)),
            'regions': 'Pune, Aurangabad, Kanpur rural, Gonda',
            'period': '2018-2020',
            'geography_warning': 'Maharashtra and Uttar Pradesh, NOT Andhra Pradesh.',
        },
        'block_to_clock_resolved': {
            'mapping': 'block index is the 15-minute slot from local midnight',
            'evidence': (f'mainline peak at block {peak_block} ({clock(peak_block)}) '
                         f'and trough at block {trough_block} ({clock(trough_block)}), '
                         'the Indian residential morning/evening double peak'),
            'previously': 'open check in data_registry/sources.yaml',
        },
        'daily_energy_kwh': {
            'emarc_median': float(e_daily.kwh.median()),
            'emarc_p25': float(e_daily.kwh.quantile(0.25)),
            'emarc_p75': float(e_daily.kwh.quantile(0.75)),
            'sharp_median': float(s_daily.kwh.median()),
            'sharp_p25': float(s_daily.kwh.quantile(0.25)),
            'sharp_p75': float(s_daily.kwh.quantile(0.75)),
            'ratio_sharp_over_emarc': float(s_daily.kwh.median() / e_daily.kwh.median()),
        },
        'peak_to_average_ratio': {
            'emarc_median': float(e_daily.par.median()),
            'sharp_median': float(s_daily.par.median()),
        },
        'diurnal_shape': {
            'correlation': correlation,
            'emarc_peak_block': peak_block, 'emarc_peak_clock': clock(peak_block),
            'sharp_peak_block': s_peak, 'sharp_peak_clock': clock(s_peak),
            'emarc_trough_clock': clock(trough_block),
            'sharp_trough_clock': clock(s_trough),
            'emarc_evening_ratio': e_evening,
            'sharp_evening_ratio': s_evening,
        },
        'emarc_by_household_type_mean_kw': {
            str(k): float(v.mean()) for k, v in e_by_type.items()},
        'interpretation': [
            'SHARP models only inventoried appliances and carries NO hidden '
            'background load, so its aggregate is expected to sit below a real '
            'mainline meter. The ratio quantifies that gap.',
            'eMARC is Maharashtra and Uttar Pradesh in 2018-2020; SHARP is Andhra '
            'Pradesh in 2021-2024. Differences in level are not necessarily errors.',
            'Shape correlation is the meaningful number here, because it is scale '
            'free and tests whether the simulator reproduces Indian daily rhythm.',
        ],
        'is_held_out_test': False,
        'is_ground_truth_for_andhra_pradesh': False,
    }
    (root / 'reports/aggregate_load_vs_emarc_v1.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')

    comparison = pd.DataFrame({
        'block': range(96),
        'clock': [clock(b) for b in range(96)],
        'emarc_mean_kw': e_profile.to_numpy(),
        'sharp_mean_kw': s_profile.to_numpy(),
        'emarc_normalised': e_shape,
        'sharp_normalised': s_shape})
    comparison.to_csv(root / 'reports/aggregate_load_profile_vs_emarc_v1.csv', index=False)

    print('\nAGGREGATE LOAD vs eMARC')
    print(f"  eMARC household-days : {len(e_daily):,}")
    print(f"  SHARP household-days : {len(s_daily):,}")
    print()
    print(f"  daily kWh   eMARC median {report['daily_energy_kwh']['emarc_median']:.2f}"
          f"  (p25 {report['daily_energy_kwh']['emarc_p25']:.2f}"
          f" p75 {report['daily_energy_kwh']['emarc_p75']:.2f})")
    print(f"              SHARP median {report['daily_energy_kwh']['sharp_median']:.2f}"
          f"  (p25 {report['daily_energy_kwh']['sharp_p25']:.2f}"
          f" p75 {report['daily_energy_kwh']['sharp_p75']:.2f})")
    print(f"              ratio {report['daily_energy_kwh']['ratio_sharp_over_emarc']:.2f}x")
    print()
    print(f"  peak/average  eMARC {report['peak_to_average_ratio']['emarc_median']:.2f}"
          f"   SHARP {report['peak_to_average_ratio']['sharp_median']:.2f}")
    print()
    print(f"  shape correlation: {correlation:+.3f}")
    print(f"  peak    eMARC {clock(peak_block)}   SHARP {clock(s_peak)}")
    print(f"  trough  eMARC {clock(trough_block)}   SHARP {clock(s_trough)}")
    print(f"  evening share  eMARC {e_evening:.2f}x mean   SHARP {s_evening:.2f}x mean")
    print('\n  report: reports/aggregate_load_vs_emarc_v1.json')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
