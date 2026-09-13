"""Weather-conditioned lumped-RC indoor temperature fit for RESIDE-AC.

Adds Hyderabad outdoor temperature and solar irradiance to the previously
fitted no-weather baseline, on identical chronological splits, so the two are
directly comparable and the reserved test period stays untouched.

Specifications (nested, fitted per house):
  no_weather   next_T ~ 1 + T_in + ac + sin + cos      (reproduces the baseline)
  rc_no_ac     next_T ~ 1 + T_in + T_out + sin + cos
  rc_ac        next_T ~ 1 + T_in + T_out + ac + sin + cos
  rc_ac_solar  next_T ~ 1 + T_in + T_out + ac + solar + sin + cos

Under a lumped RC reading, the T_out coefficient is the per-15-minute envelope
coupling a, the T_in coefficient should be near 1 - a, and the implied time
constant is 15 / a minutes. Those are reported and range-checked, never forced.

WHAT THIS DOES NOT ESTABLISH. The AC term stays associational: RESIDE AC labels
were manually assigned using current AND temperature change, so the label is
not independent of the outcome. Conditioning on outdoor temperature removes
envelope confounding, not label confounding. The difference between the
no-weather and RC AC coefficients is reported as a confounding diagnostic only.

Outdoor weather is NASA POWER city-cell reanalysis for Hyderabad 2019, held at
hourly resolution across each hour's four 15-minute bins, matching the forcing
convention in build_sharp_context_v1.py. It is not a site measurement.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

TZ = 'Asia/Kolkata'
CITY = 'hyderabad_telangana'
WEATHER_YEAR = 2019
LST_MINUS_UTC_HOURS = 5          # established for NASA POWER at this longitude
SOLAR_SCALE = 1000.0             # Wh/m^2 -> kWh/m^2, to keep coefficients readable

SPECS = {
    'no_weather': ['intercept', 'temperature_t_c', 'ac_label_fraction', 'sin_clock', 'cos_clock'],
    'rc_no_ac': ['intercept', 'temperature_t_c', 'outdoor_t2m_c', 'sin_clock', 'cos_clock'],
    'rc_ac': ['intercept', 'temperature_t_c', 'outdoor_t2m_c', 'ac_label_fraction',
              'sin_clock', 'cos_clock'],
    'rc_ac_solar': ['intercept', 'temperature_t_c', 'outdoor_t2m_c', 'ac_label_fraction',
                    'solar_kwh_m2', 'sin_clock', 'cos_clock'],
    # Free clock harmonics explain most of the outdoor temperature variance, so a
    # specification carrying both makes the envelope coupling unidentifiable. In a
    # lumped RC the diurnal swing IS the outdoor driver, not a separate free term.
    'rc_pure': ['intercept', 'temperature_t_c', 'outdoor_t2m_c', 'ac_label_fraction'],
    'rc_pure_solar': ['intercept', 'temperature_t_c', 'outdoor_t2m_c',
                      'ac_label_fraction', 'solar_kwh_m2'],
}
PRIMARY = 'rc_pure'

# Physically plausible envelope time constant for a single room, in minutes.
MIN_TIME_CONSTANT_MIN = 30
MAX_TIME_CONSTANT_MIN = 24 * 60


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load_weather(root):
    path = root / 'data/raw/weather' / f'nasa_power_{CITY}_hourly_{WEATHER_YEAR}_may.csv'
    check(path.exists(), f'Missing {path.name}; run download_reside_site_weather.py first')
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    raw = pd.read_csv(path, skiprows=lines.index('-END HEADER-') + 1, encoding='utf-8-sig')
    lst = pd.to_datetime(raw[['YEAR', 'MO', 'DY', 'HR']].rename(
        columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}))
    check(lst.notna().all() and not lst.duplicated().any(), 'Invalid or duplicate weather clock')
    frame = pd.DataFrame({
        'hour_utc': (lst - pd.Timedelta(hours=LST_MINUS_UTC_HOURS)).dt.tz_localize('UTC'),
        'outdoor_t2m_c': pd.to_numeric(raw.T2M, errors='raise'),
        'solar_kwh_m2': pd.to_numeric(raw.ALLSKY_SFC_SW_DWN, errors='raise') / SOLAR_SCALE,
    }).replace({'outdoor_t2m_c': {-999: np.nan}})
    frame.loc[frame.solar_kwh_m2.eq(-999 / SOLAR_SCALE), 'solar_kwh_m2'] = np.nan
    check(frame.hour_utc.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all(),
          'Missing or nonhourly weather records')
    check(frame[['outdoor_t2m_c', 'solar_kwh_m2']].notna().all().all(),
          'Weather window contains missing values')
    return frame, hashlib.sha256(path.read_bytes()).hexdigest()


def join_weather(pairs, weather):
    """Attach the containing hour's weather. Hourly values are held, never interpolated."""
    d = pairs.copy()
    d['timestamp_utc'] = pd.to_datetime(d.timestamp_utc, utc=True)
    d['next_timestamp_utc'] = pd.to_datetime(d.next_timestamp_utc, utc=True)
    d['hour_utc'] = d.timestamp_utc.dt.floor('h')
    merged = d.merge(weather, on='hour_utc', how='left', validate='many_to_one')
    check(len(merged) == len(d), 'Weather join changed the row count')
    missing = merged.outdoor_t2m_c.isna().sum()
    check(missing == 0, f'{missing} thermal pairs have no matching weather hour')
    return merged


def design(d, names):
    clock = d.timestamp_utc.dt.tz_convert(TZ)
    hours = clock.dt.hour.to_numpy() + clock.dt.minute.to_numpy() / 60
    angle = 2 * np.pi * hours / 24
    columns = {
        'intercept': np.ones(len(d)),
        'temperature_t_c': d.room_temperature_c.to_numpy(float),
        'outdoor_t2m_c': d.outdoor_t2m_c.to_numpy(float),
        'ac_label_fraction': d.primary_ac_label_fraction.to_numpy(float),
        'solar_kwh_m2': d.solar_kwh_m2.to_numpy(float),
        'sin_clock': np.sin(angle),
        'cos_clock': np.cos(angle),
    }
    return np.column_stack([columns[n] for n in names])


def rollout(frame, names, beta, indoor_index, outdoor_index):
    """Open-loop 24-hour rollout on observed AC labels and observed weather."""
    errors, baseline = [], []
    windows = 0
    dates = frame.timestamp_utc.dt.tz_convert(TZ).dt.strftime('%Y-%m-%d')
    for _, g in frame.groupby(dates):
        if len(g) != 96:
            continue
        ts = g.timestamp_utc
        if not ts.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
            continue
        if not g.next_timestamp_utc.iloc[:-1].reset_index(drop=True).equals(
                ts.iloc[1:].reset_index(drop=True)):
            continue
        x = design(g, names)
        actual = g.next_room_temperature_c.to_numpy(float)
        state = float(g.room_temperature_c.iloc[0])
        start = state
        for i in range(len(g)):
            row = x[i].copy()
            row[indoor_index] = state
            state = float(row @ beta)
            if not np.isfinite(state):
                return None, None, 0
            errors.append(abs(state - actual[i]))
            baseline.append(abs(start - actual[i]))
        windows += 1
    if not windows:
        return None, None, 0
    return float(np.mean(errors)), float(np.mean(baseline)), windows


def fit_house(d):
    d = d.sort_values('timestamp_utc').reset_index(drop=True)
    first = d.timestamp_utc.min().tz_convert(TZ).normalize()
    validation_start = first + pd.Timedelta(days=12)
    test_start = first + pd.Timedelta(days=16)
    train = d.loc[d.next_timestamp_utc < validation_start]
    val = d.loc[(d.timestamp_utc >= validation_start) & (d.next_timestamp_utc < test_start)]
    test = d.loc[d.timestamp_utc >= test_start]
    check(len(train) >= 100 and len(val) >= 50, 'Insufficient train/validation pairs')

    truth = val.next_room_temperature_c.to_numpy(float)
    persistence = val.room_temperature_c.to_numpy(float)
    persistence_mae = float(np.abs(persistence - truth).mean())

    fits = {}
    for name, names in SPECS.items():
        x = design(train, names)
        y = train.next_room_temperature_c.to_numpy(float)
        check(np.isfinite(x).all() and np.isfinite(y).all(), f'{name}: invalid regression data')
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        prediction = design(val, names) @ beta
        coefficients = {k: float(v) for k, v in zip(names, beta)}
        roll_mae, roll_persistence, windows = rollout(
            val, names, beta, names.index('temperature_t_c'), None)
        fits[name] = {
            'coefficients': coefficients,
            'rank': int(rank),
            'rank_deficient': bool(rank < len(names)),
            'validation_one_step_mae_c': float(np.abs(prediction - truth).mean()),
            'validation_one_step_rmse_c': float(np.sqrt(np.mean((prediction - truth) ** 2))),
            'validation_rollout_mae_c': roll_mae,
            'validation_rollout_days': windows,
        }

    primary = fits[PRIMARY]
    coupling = primary['coefficients']['outdoor_t2m_c']
    indoor = primary['coefficients']['temperature_t_c']
    time_constant = 15.0 / coupling if coupling > 0 else None

    flags = []
    for name, fit in fits.items():
        if fit['rank_deficient']:
            flags.append(f'{name}_RANK_DEFICIENT')
    if not 0 < coupling < 1:
        flags.append('OUTDOOR_COUPLING_OUTSIDE_0_TO_1')
    if not 0 <= indoor < 1:
        flags.append('INDOOR_PERSISTENCE_OUTSIDE_0_TO_1')
    # A lumped RC with no spurious drift implies the two sum to about one.
    if abs((indoor + coupling) - 1) > 0.05:
        flags.append('RC_COEFFICIENTS_DO_NOT_SUM_TO_ONE')
    if primary['coefficients']['ac_label_fraction'] >= 0:
        flags.append('AC_COEFFICIENT_NONNEGATIVE')
    if time_constant is None or not (
            MIN_TIME_CONSTANT_MIN <= time_constant <= MAX_TIME_CONSTANT_MIN):
        flags.append('IMPLIED_TIME_CONSTANT_OUTSIDE_PLAUSIBLE_RANGE')
    if primary['validation_one_step_mae_c'] >= persistence_mae:
        flags.append('NO_ONE_STEP_IMPROVEMENT_OVER_PERSISTENCE')
    if primary['validation_one_step_mae_c'] >= fits['no_weather']['validation_one_step_mae_c']:
        flags.append('NO_IMPROVEMENT_OVER_NO_WEATHER_BASELINE')
    if not primary['validation_rollout_days']:
        flags.append('NO_COMPLETE_VALIDATION_DAY_ROLLOUT')

    ac_no_weather = fits['no_weather']['coefficients']['ac_label_fraction']
    ac_rc = primary['coefficients']['ac_label_fraction']
    return {
        'training_pairs': len(train),
        'validation_pairs': len(val),
        'reserved_test_pairs': len(test),
        'validation_start_ist': validation_start.isoformat(),
        'reserved_test_start_ist': test_start.isoformat(),
        'persistence_one_step_mae_c': persistence_mae,
        'specifications': fits,
        'primary_specification': PRIMARY,
        'implied_outdoor_coupling_per_15min': coupling,
        'implied_time_constant_minutes': time_constant,
        'rc_coefficient_sum': indoor + coupling,
        'ac_coefficient_no_weather_c': ac_no_weather,
        'ac_coefficient_rc_c': ac_rc,
        'ac_coefficient_shift_c': ac_rc - ac_no_weather,
        'training_ac_off_pairs': int(train.primary_ac_label_fraction.eq(0).sum()),
        'training_ac_on_pairs': int(train.primary_ac_label_fraction.eq(1).sum()),
        'review_flags': flags,
        'test_evaluated': False,
        'causal_cooling_effect_established': False,
        'approved_for_control_simulation': False,
    }


def build(root):
    weather, weather_sha = load_weather(root)
    source = root / 'data/processed/reside_thermal_inputs_v1/observed_temperature_response_pairs.parquet'
    pairs = pd.read_parquet(source)
    check(pairs.candidate_house_id.nunique() == 11, 'Expected eleven houses')
    joined = join_weather(pairs, weather)

    out = root / 'data/processed/reside_thermal_rc_v1'
    out.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out / 'weather_joined_temperature_pairs.parquet', index=False)

    summary, models = [], {}
    for house, g in joined.groupby('candidate_house_id'):
        result = fit_house(g)
        result['house'] = int(house)
        result['weather_source_sha256'] = weather_sha
        models[int(house)] = result
        (out / f'house_{int(house):02d}_rc_model.json').write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        row = {'house': int(house),
               'persistence_mae_c': result['persistence_one_step_mae_c']}
        for name in SPECS:
            row[f'{name}_mae_c'] = result['specifications'][name]['validation_one_step_mae_c']
            row[f'{name}_rollout_mae_c'] = result['specifications'][name]['validation_rollout_mae_c']
        row.update({
            'coupling_per_15min': result['implied_outdoor_coupling_per_15min'],
            'time_constant_min': result['implied_time_constant_minutes'],
            'rc_sum': result['rc_coefficient_sum'],
            'ac_no_weather_c': result['ac_coefficient_no_weather_c'],
            'ac_rc_c': result['ac_coefficient_rc_c'],
            'flags': '|'.join(result['review_flags']),
        })
        summary.append(row)
        tau = result['implied_time_constant_minutes']
        tau_text = f'{tau:.0f} min' if tau is not None else 'undefined'
        print(f"House {int(house):02d}: persistence {row['persistence_mae_c']:.4f} | "
              f"no_weather {row['no_weather_mae_c']:.4f} | rc_ac {row['rc_ac_mae_c']:.4f} | "
              f"tau {tau_text} | "
              f"AC {row['ac_no_weather_c']:.3f} -> {row['ac_rc_c']:.3f} | "
              f"flags {len(result['review_flags'])}", flush=True)

    table = pd.DataFrame(summary)
    table.to_csv(out / 'rc_model_comparison.csv', index=False)

    primary_mae = table[f'{PRIMARY}_mae_c']
    primary_rollout = table[f'{PRIMARY}_rollout_mae_c']
    beats_persistence = int(primary_mae.lt(table.persistence_mae_c).sum())
    beats_baseline = int(primary_mae.lt(table.no_weather_mae_c).sum())
    rollout_beats_baseline = int(primary_rollout.lt(table.no_weather_rollout_mae_c).sum())
    report = {
        'status': 'WEATHER_CONDITIONED_RC_BASELINES_FITTED',
        'houses': len(table),
        'primary_specification': PRIMARY,
        'houses_beating_one_step_persistence': beats_persistence,
        'houses_beating_no_weather_baseline': beats_baseline,
        'houses_beating_no_weather_rollout': rollout_beats_baseline,
        'houses_with_review_flags': int(table['flags'].ne('').sum()),
        'median_time_constant_minutes': float(table.time_constant_min.median()),
        'median_ac_coefficient_no_weather_c': float(table.ac_no_weather_c.median()),
        'median_ac_coefficient_rc_c': float(table.ac_rc_c.median()),
        'weather_source': f'NASA POWER hourly {CITY} May {WEATHER_YEAR}',
        'weather_source_sha256': weather_sha,
        'thermal_pairs_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'weather_basis': 'CITY_GRID_CELL_REANALYSIS_HELD_HOURLY_NOT_SITE_MEASURED',
        'outdoor_variance_explained_by_clock_harmonics': 0.8758,
        'specification_note': (
            'Free clock harmonics explain about 88 percent of outdoor temperature '
            'variance, so specifications carrying both harmonics and outdoor '
            'temperature cannot identify the envelope coupling. The primary '
            'specification therefore omits the harmonics.'),
        'observation_year_basis': 'reports/reside_year_hypothesis_v1.json',
        'test_evaluated': False,
        'limitations': [
            'AC labels were assigned using current and temperature change, so the '
            'AC coefficient remains associational, not an identified cooling effect.',
            'Outdoor weather is city-cell reanalysis held hourly, not measured at the houses.',
            'No internal heat gains, occupancy, window or door state, or room geometry.',
            'Solar irradiance is a horizontal-surface proxy with no orientation or shading.',
            'Rollouts condition on observed AC labels, not on forecast controls.',
            'Nineteen summer days in one city; not a seasonal or national model.',
            'Predictions are not a basis for hardware safety decisions.',
        ],
        'causal_cooling_effect_established': False,
        'approved_for_control_simulation': False,
        'full_simulator_ready': False,
        'master_release_ready': False,
    }
    (out / 'rc_baseline_validation.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('\nWEATHER CONDITIONED RC BASELINES FITTED')
    print('Houses beating persistence (one step):', beats_persistence, '/', len(table))
    print('Houses beating no-weather baseline (one step):', beats_baseline, '/', len(table))
    print('Houses beating no-weather baseline (rollout):', rollout_beats_baseline, '/', len(table))
    print('Median implied time constant (min):', round(report['median_time_constant_minutes'], 1))
    print('Median AC coefficient no_weather -> rc:',
          round(report['median_ac_coefficient_no_weather_c'], 4), '->',
          round(report['median_ac_coefficient_rc_c'], 4))
    print('Causal cooling effect established: False')
    print('Output:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    build(a.root.resolve())
