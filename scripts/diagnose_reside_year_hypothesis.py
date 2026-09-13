"""Test which candidate year the RESIDE-AC clock actually belongs to.

RESIDE CSV timestamps read May 2019; the dataset description states May 2021.
The calendar day-of-month and clock time are identical under both hypotheses,
so only the actual day-to-day weather differs. If the indoor record really
comes from one of those years, daily-mean indoor temperature should track that
year's daily-mean outdoor temperature and not the other year's.

This diagnoses the CLOCK only. It does not fit a thermal model, does not
establish causal cooling, and does not resolve the sensor drift note.

The decision rule below is fixed before any result is computed.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import requests

TZ = 'Asia/Kolkata'
CITY = 'hyderabad_telangana'
LATITUDE, LONGITUDE = 17.3850, 78.4867
YEARS = (2019, 2021)

# PRE-REGISTERED DECISION RULE. A year is accepted only if BOTH hold.
MIN_HOUSES_FAVOURING = 9       # out of 11
MIN_MEAN_CORRELATION_GAIN = 0.15
MIN_DAYS = 15                  # usable common days out of the 19-day window


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_power_csv(root, year):
    path = root / 'data/raw/weather' / f'nasa_power_{CITY}_hourly_{year}_may.csv'
    check(path.exists(), f'Missing {path.name}; run download_reside_site_weather.py first')
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    start = lines.index('-END HEADER-') + 1
    raw = pd.read_csv(path, skiprows=start, encoding='utf-8-sig')
    lst = pd.to_datetime(raw[['YEAR', 'MO', 'DY', 'HR']].rename(
        columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}))
    check(lst.notna().all() and not lst.duplicated().any(), f'{year}: invalid weather clock')
    w = raw[['T2M', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS10M']].apply(
        pd.to_numeric, errors='raise').replace(-999, np.nan)
    check(not np.isinf(w.to_numpy(float)).any(), f'{year}: infinite weather values')
    return lst, w


def verify_clock_offset(root):
    """Confirm NASA LST equals UTC+5 at this longitude, as established for Guntur."""
    path = root / 'data/raw/weather' / f'{CITY}_utc_clock_check.json'
    if not path.exists():
        print('Downloading Hyderabad UTC clock check...', flush=True)
        response = requests.get(
            'https://power.larc.nasa.gov/api/temporal/hourly/point',
            params={'parameters': 'T2M,RH2M', 'community': 'RE',
                    'longitude': LONGITUDE, 'latitude': LATITUDE,
                    'start': '20210503', 'end': '20210510',
                    'format': 'JSON', 'time-standard': 'UTC'},
            timeout=300)
        response.raise_for_status()
        path.write_bytes(response.content)
    j = json.loads(path.read_text(encoding='utf-8'))
    check(j['header']['time_standard'] == 'UTC', 'Clock check is not UTC')
    u = pd.DataFrame(j['properties']['parameter'])[['T2M', 'RH2M']].replace(-999, np.nan)
    u.index = pd.to_datetime(u.index, format='%Y%m%d%H')
    lst, w = read_power_csv(root, 2021)
    best = None
    for offset in range(3, 8):
        joined = w[['T2M', 'RH2M']].set_axis(lst - pd.Timedelta(hours=offset)).join(
            u, rsuffix='_utc', how='inner').dropna()
        if len(joined) < 24:
            continue
        mae = float(np.abs(joined.T2M - joined.T2M_utc).mean())
        if best is None or mae < best[1]:
            best = (offset, mae)
    check(best is not None, 'No overlapping hours for the clock check')
    check(best[0] == 5 and best[1] < 1e-6,
          f'Hyderabad LST offset is {best[0]} h (MAE {best[1]:.6f}); '
          'the established UTC+5 convention does not hold here')
    print(f'Clock check: LST = UTC+{best[0]} h, temperature MAE {best[1]:.6f} C', flush=True)
    return best[0]


def outdoor_daily(root, year, offset_hours):
    lst, w = read_power_csv(root, year)
    stamp = (lst - pd.Timedelta(hours=offset_hours)).dt.tz_localize('UTC').dt.tz_convert(TZ)
    check(stamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all(),
          f'{year}: missing or nonhourly weather records')
    frame = w.assign(month_day=stamp.dt.strftime('%m-%d')).dropna(subset=['T2M'])
    counts = frame.groupby('month_day').T2M.size()
    complete = counts[counts.eq(24)].index
    return frame[frame.month_day.isin(complete)].groupby('month_day').T2M.mean()


def indoor_daily(pairs, ac_off_only):
    d = pairs.copy()
    d['timestamp_ist'] = pd.to_datetime(d.timestamp_utc, utc=True).dt.tz_convert(TZ)
    if ac_off_only:
        d = d[d.primary_ac_label_fraction.eq(0)]
    d['month_day'] = d.timestamp_ist.dt.strftime('%m-%d')
    grouped = d.groupby(['candidate_house_id', 'month_day']).room_temperature_c
    # Require enough intervals for a daily mean to be meaningful.
    minimum = 8 if ac_off_only else 48
    return grouped.mean()[grouped.size().ge(minimum)]


def correlate(indoor, outdoor_by_year):
    rows = []
    for house, series in indoor.groupby(level=0):
        series = series.droplevel(0)
        common = series.index
        for year in YEARS:
            common = common.intersection(outdoor_by_year[year].index)
        if len(common) < MIN_DAYS:
            rows.append({'house': int(house), 'days': len(common), 'usable': False,
                         **{f'r_{y}': None for y in YEARS}})
            continue
        inside = series.loc[common].to_numpy(float)
        row = {'house': int(house), 'days': len(common), 'usable': True}
        for year in YEARS:
            outside = outdoor_by_year[year].loc[common].to_numpy(float)
            if inside.std() == 0 or outside.std() == 0:
                row[f'r_{year}'] = None
                row['usable'] = False
            else:
                row[f'r_{year}'] = float(np.corrcoef(inside, outside)[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)


def decide(table):
    usable = table[table.usable].copy()
    if len(usable) < MIN_HOUSES_FAVOURING:
        return 'INCONCLUSIVE', {'reason': f'Only {len(usable)} usable houses'}
    means = {y: float(usable[f'r_{y}'].mean()) for y in YEARS}
    winner = max(means, key=means.get)
    other = [y for y in YEARS if y != winner][0]
    favouring = int((usable[f'r_{winner}'] > usable[f'r_{other}']).sum())
    gain = means[winner] - means[other]
    evidence = {'candidate_year': winner, 'other_year': other,
                'mean_correlation': means, 'houses_favouring_candidate': favouring,
                'usable_houses': len(usable), 'mean_correlation_gain': gain,
                'required_houses': MIN_HOUSES_FAVOURING,
                'required_gain': MIN_MEAN_CORRELATION_GAIN}
    passed = favouring >= MIN_HOUSES_FAVOURING and gain >= MIN_MEAN_CORRELATION_GAIN
    return (f'YEAR_{winner}_SUPPORTED' if passed else 'INCONCLUSIVE'), evidence


def build(root):
    offset = verify_clock_offset(root)
    outdoor = {y: outdoor_daily(root, y, offset) for y in YEARS}
    for y in YEARS:
        print(f'Outdoor {y}: {len(outdoor[y])} complete days, '
              f'mean {outdoor[y].mean():.2f} C, sd {outdoor[y].std():.2f} C', flush=True)

    source = root / 'data/processed/reside_thermal_inputs_v1'
    frames = []
    for path in sorted(source.glob('house_*_documented_units.parquet')):
        frames.append(pd.read_parquet(path)[
            ['candidate_house_id', 'timestamp_utc', 'room_temperature_c',
             'primary_ac_label_fraction']])
    check(len(frames) == 11, 'Expected eleven RESIDE houses')
    pairs = pd.concat(frames, ignore_index=True).dropna(subset=['room_temperature_c'])

    results = {}
    for name, ac_off_only in [('all_intervals', False), ('ac_off_intervals', True)]:
        table = correlate(indoor_daily(pairs, ac_off_only), outdoor)
        verdict, evidence = decide(table)
        results[name] = {'verdict': verdict, 'evidence': evidence,
                         'per_house': table.to_dict('records')}
        table.to_csv(root / f'reports/reside_year_hypothesis_{name}_v1.csv', index=False)
        print(f'\n[{name}] verdict: {verdict}')
        print(table.to_string(index=False))

    primary = results['ac_off_intervals']['verdict']
    secondary = results['all_intervals']['verdict']
    agreed = primary == secondary and primary != 'INCONCLUSIVE'
    report = {
        'status': 'RESIDE_YEAR_HYPOTHESIS_DIAGNOSED',
        'primary_test': 'ac_off_intervals',
        'primary_verdict': primary,
        'secondary_verdict': secondary,
        'tests_agree': agreed,
        'year_discrepancy_resolved': bool(agreed),
        'decision_rule': {'min_houses_favouring': MIN_HOUSES_FAVOURING,
                          'min_mean_correlation_gain': MIN_MEAN_CORRELATION_GAIN,
                          'min_days': MIN_DAYS, 'preregistered': True},
        'results': results,
        'scope': 'CLOCK_YEAR_DIAGNOSIS_ONLY',
        'thermal_model_fitted': False,
        'causal_cooling_effect_established': False,
        'full_simulator_ready': False,
        'master_release_ready': False,
    }
    out = root / 'reports/reside_year_hypothesis_v1.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nRESIDE YEAR HYPOTHESIS DIAGNOSED')
    print('Primary (AC-off):', primary)
    print('Secondary (all intervals):', secondary)
    print('Year discrepancy resolved:', agreed)
    print('Report:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
