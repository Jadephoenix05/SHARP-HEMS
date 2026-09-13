"""SHARP regional grid + Guntur weather context. Run from the project folder.

Grid clock: explicit scenario assumption, NOT source-confirmed IST.
Absolute grid values retain unspecified source units; RL grid features are unitless.
Hourly means are held for simulator forcing. Policy observations use a separate
one-hour-delayed stream; forcing columns must not enter the policy observation.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import numpy as np
import pandas as pd

TZ = 'Asia/Kolkata'
WEATHER = ['T2M', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS10M']


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def split_for(t):
    return np.where(t.dt.year <= 2022, 'train',
                    np.where(t.dt.year == 2023, 'validation', 'test'))


def held(clock, source, names, prefix, delay_hours):
    clock = clock.copy()
    clock['timestamp_ist'] = clock.timestamp_ist.astype(pd.DatetimeTZDtype(unit='ns', tz=TZ))
    right = source[['timestamp_ist'] + names].copy()
    right['available_at'] = right.pop('timestamp_ist') + pd.Timedelta(hours=delay_hours)
    right['available_at'] = right.available_at.astype(pd.DatetimeTZDtype(unit='ns', tz=TZ))
    right = right.rename(columns={c: prefix + c for c in names})
    # At exactly one hour old, the previous hourly record is expired.
    result = pd.merge_asof(clock, right.sort_values('available_at'),
                           left_on='timestamp_ist', right_on='available_at',
                           direction='backward', tolerance=pd.Timedelta(minutes=59, seconds=59))
    return result.drop(columns='available_at')


def fit_reference(grid):
    train = grid.loc[grid.timestamp_ist.dt.year.between(2021, 2022)]
    refs = {}
    for month in range(1, 13):
        values = np.sort(train.loc[train.timestamp_ist.dt.month.eq(month), 'demand'].to_numpy())
        check(len(values) >= 24, f'Insufficient training grid hours in month {month}')
        p90, p99 = np.quantile(values, [.9, .99])
        check(p99 > p90, f'Degenerate grid thresholds in month {month}')
        refs[month] = (values, float(p90), float(p99))
    return refs


def grid_features(frame, column, prefix, refs):
    percentile = np.full(len(frame), np.nan)
    severity = np.full(len(frame), np.nan)
    for month, (reference, p90, p99) in refs.items():
        mask = frame.timestamp_ist.dt.month.eq(month).to_numpy()
        values = frame.loc[mask, column].to_numpy()
        percentile[mask] = np.searchsorted(reference, values, side='right') / len(reference)
        severity[mask] = np.clip((values - p90) / (p99 - p90), 0, 1)
        invalid = mask & ~np.isfinite(frame[column].to_numpy())
        percentile[invalid] = np.nan
        severity[invalid] = np.nan
    frame[prefix + 'grid_percentile'] = percentile
    frame[prefix + 'grid_peak_severity'] = severity


def self_test():
    times = pd.date_range('2021-01-01 00:30', periods=3, freq='h', tz=TZ)
    source = pd.DataFrame({'timestamp_ist': times, 'x': [10., 20., 30.]})
    clock = pd.DataFrame({'timestamp_ist': pd.date_range(times[0], periods=12, freq='15min')})
    obs = held(clock, source, ['x'], 'obs_', 1)
    check(obs.obs_x.iloc[:4].isna().all(), 'Hourly mean exposed before availability')
    check(obs.obs_x.iloc[4:8].eq(10).all(), 'Incorrect delayed hold')
    gap = held(clock, source.iloc[[0, 2]], ['x'], 'forcing_', 0)
    check(gap.forcing_x.iloc[4:8].isna().all(), 'Missing hour was forward-filled')
    alltimes = pd.date_range('2021-01-01', '2024-04-30 23:00', freq='h', tz=TZ)
    grid = pd.DataFrame({'timestamp_ist': alltimes, 'demand': 100 + np.arange(len(alltimes)) % 1000})
    a = fit_reference(grid)
    grid.loc[grid.timestamp_ist.dt.year.ge(2023), 'demand'] = 999999
    b = fit_reference(grid)
    check(all(np.array_equal(a[m][0], b[m][0]) for m in a), 'Validation/test influenced thresholds')
    print('SELF-TEST PASS: delayed observations, gap handling, training-only thresholds')


def build(root, grid_timezone):
    grid_path = root / 'data/raw/grid_india/regional_hourly/hourlyLoadDataIndia.xlsx'
    weather_path = root / 'data/raw/weather/nasa_power_guntur_ap_hourly_2021_2025.csv'
    utc_path = root / 'data/raw/weather/guntur_utc_clock_check.json'
    for p in [grid_path, weather_path, utc_path]:
        check(p.is_file(), f'Missing input: {p}')
    print('Reading and validating grid and weather...', flush=True)
    raw = pd.read_excel(grid_path)
    g = pd.DataFrame({'timestamp_ist': pd.to_datetime(raw['datetime'], errors='raise'),
                      'demand': pd.to_numeric(raw['Southern Region Hourly Demand'], errors='raise')})
    check(g.timestamp_ist.dt.tz is None, 'Grid clock changed: inspect timezone before rebuilding')
    check(g.timestamp_ist.notna().all() and not g.timestamp_ist.duplicated().any(), 'Invalid/duplicate grid clock')
    g['timestamp_ist'] = g.timestamp_ist.dt.tz_localize(grid_timezone).dt.tz_convert(TZ)
    g = g.sort_values('timestamp_ist').reset_index(drop=True)
    check(g.timestamp_ist.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all(), 'Grid has missing or nonhourly records')
    check(np.isfinite(g.demand).all() and g.demand.gt(0).all(), 'Invalid grid demand')

    lines = weather_path.read_text(encoding='utf-8-sig').splitlines()
    n = next(i for i, line in enumerate(lines) if line.startswith('YEAR,'))
    raww = pd.read_csv(io.StringIO('\n'.join(lines[n:])))
    lst = pd.to_datetime(raww[['YEAR','MO','DY','HR']].rename(
        columns={'YEAR':'year','MO':'month','DY':'day','HR':'hour'}))
    check(lst.notna().all() and not lst.duplicated().any(), 'Invalid/duplicate weather clock')
    w = raww[WEATHER].apply(pd.to_numeric, errors='raise').replace(-999, np.nan)
    check(not np.isinf(w.to_numpy()).any(), 'Infinite weather values')
    j = json.loads(utc_path.read_text())
    check(j['header']['time_standard'] == 'UTC', 'Clock check is not UTC')
    u = pd.DataFrame(j['properties']['parameter'])[['T2M','RH2M']].replace(-999, np.nan)
    u.index = pd.to_datetime(u.index, format='%Y%m%d%H') + pd.Timedelta(hours=5)
    comparison = w[['T2M','RH2M']].set_axis(lst).join(u, rsuffix='_utc', how='inner').dropna()
    check(len(comparison) >= 168, 'Insufficient valid UTC comparison hours')
    for c in ['T2M','RH2M']:
        check(np.allclose(comparison[c], comparison[c+'_utc'], atol=1e-6, rtol=0), 'NASA offset check no longer matches')
    w.insert(0, 'timestamp_ist', (lst - pd.Timedelta(hours=5)).dt.tz_localize('UTC').dt.tz_convert(TZ))
    w = w.sort_values('timestamp_ist').reset_index(drop=True)
    check(w.timestamp_ist.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all(), 'Missing/nonhourly weather records')
    check(w.RH2M.dropna().between(0, 100).all(), 'Humidity outside 0-100 percent')
    check(w.ALLSKY_SFC_SW_DWN.dropna().ge(0).all() and w.WS10M.dropna().ge(0).all(), 'Negative solar/wind values')

    start = pd.Timestamp('2021-01-01', tz=TZ)
    end = min(pd.Timestamp('2024-04-30 23:45', tz=TZ), g.timestamp_ist.max() + pd.Timedelta(minutes=45))
    clock = pd.DataFrame({'timestamp_ist': pd.date_range(start, end, freq='15min')})
    result = clock.copy()
    for source, cols in [(g, ['demand']), (w, WEATHER)]:
        for prefix, delay in [('forcing_', 0), ('obs_', 1)]:
            temp = held(clock, source, cols, prefix, delay)
            for c in temp.columns.drop('timestamp_ist'):
                result[c] = temp[c].to_numpy()
    numeric = result.columns.drop('timestamp_ist').tolist()
    result['day_id'] = result.timestamp_ist.dt.strftime('%Y-%m-%d')
    valid = result[numeric].notna().all(axis=1)
    day_counts = valid.groupby(result.day_id).sum()
    complete_days = day_counts[day_counts.eq(96)].index
    excluded = day_counts[~day_counts.eq(96)]
    result = result.loc[result.day_id.isin(complete_days)].reset_index(drop=True)
    check(len(result) > 0, 'No complete context days')
    result['split'] = split_for(result.timestamp_ist)
    check(set(result.split) == {'train','validation','test'}, 'Missing split')
    refs = fit_reference(g)
    grid_features(result, 'obs_demand', 'obs_', refs)
    grid_features(result, 'forcing_demand', 'forcing_', refs)
    # Do not imply that unnamed source demand units were verified as MW.
    result = result.rename(columns={'obs_demand':'obs_grid_demand_source_value',
                                    'forcing_demand':'forcing_grid_demand_source_value'})
    result.insert(1, 'timestamp_utc', result.timestamp_ist.dt.tz_convert('UTC'))
    check(not result.timestamp_utc.duplicated().any(), 'Duplicate output time')
    check(result.groupby('day_id').size().eq(96).all(), 'Incomplete day survived filtering')
    allowed = ['obs_T2M','obs_RH2M','obs_ALLSKY_SFC_SW_DWN','obs_WS10M',
               'obs_grid_percentile','obs_grid_peak_severity']
    check(np.isfinite(result[allowed].to_numpy()).all(), 'Nonfinite policy features')
    output = root / 'data/processed/simulator_context_v1'
    output.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output/'regional_grid_guntur_weather_15min_v1.parquet', index=False)
    manifest = {
        'status':'CONTEXT_BUILT_WITH_DECLARED_ASSUMPTIONS', 'master_release_ready':False,
        'rows':len(result), 'complete_days':len(complete_days), 'split_rows':result.groupby('split').size().to_dict(),
        'first_timestamp_ist':str(result.timestamp_ist.min()), 'last_timestamp_ist':str(result.timestamp_ist.max()),
        'grid_geography':'Southern Region India, not AP or Guntur measurements',
        'grid_timezone_assumption':grid_timezone, 'grid_source_timezone_verified':False,
        'grid_absolute_unit':'unverified; excluded from policy feature allowlist',
        'grid_source':'https://www.kaggle.com/datasets/shubhamvashisht/hourly-load-india-electrical-load-forecasting',
        'weather_lst_minus_utc_hours':5, 'weather_clock_comparison_hours':len(comparison),
        'holding_convention':'source label treated as interval start; values held for one hour; not measured 15-minute data',
        'policy_availability':'one-hour delay after source label; conservative simulation convention, not actual publication latency',
        'forcing_columns':'environment/reference use only; forbidden in policy observations',
        'policy_context_allowlist':allowed,
        'solar_unit':'Wh/m2 over source hour; numerically equivalent to hourly-mean W/m2, not instantaneous irradiance',
        'threshold_fit_period':'2021-01-01 through 2022-12-31, training only',
        'splits':{'train':'2021-2022','validation':'2023','test':'2024 Jan-Apr; partial-year evaluation'},
        'excluded_incomplete_days':{str(k):int(v) for k,v in excluded.items()},
        'source_sha256':{str(p.relative_to(root)):sha(p) for p in [grid_path,weather_path,utc_path]},
        'builder_sha256':sha(Path(__file__)),
        'monthly_thresholds':{str(m):{'p90':r[1],'p99':r[2],'training_hours':len(r[0])} for m,r in refs.items()}}
    (output/'context_manifest_v1.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    reports = root/'reports'; reports.mkdir(exist_ok=True)
    (reports/'sharp_context_build_v1.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('\nSHARP CONTEXT BUILT')
    print('Rows:', len(result), '| Complete days:', len(complete_days))
    print(result.groupby('split').size().to_string())
    print('Excluded incomplete days:', len(excluded))
    print('Grid timezone: explicitly assumed', grid_timezone)
    print('Output:', output)
    print('This completes the context layer; the full RL master is not yet release-ready.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--grid-timezone-assumption', choices=['Asia/Kolkata','UTC'])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.grid_timezone_assumption:
            parser.error('Provide --grid-timezone-assumption explicitly; source timezone is unconfirmed')
        build(args.root.resolve(), args.grid_timezone_assumption)
