"""Generate the SHARP RL transition layer across AP households, with thermal.

This is the dataset layer BDQ actually trains on. It differs from the earlier
integration pilot in four ways:

  1. The validated thermal component (configs/thermal/sharp_thermal_rc_v1.json)
     drives indoor temperature from Guntur outdoor temperature, so comfort is a
     real reward signal rather than a zero placeholder.
  2. Air conditioners are thermostatic, not service-budget devices. The agent
     enables the AC; the thermostat decides compressor duty; duty sets power.
  3. All Andhra Pradesh household templates with a reported connection load are
     run, across all three seasons, on household-disjoint splits.
  4. Context days are drawn only from each split's own year range, so no
     train/validation/test leakage is possible through weather or grid state.

Honest scope. Appliance power values remain proxies and assumptions carrying
their upstream provenance; REFIT is UK evidence; thermal parameters are declared
assumptions bounded by the RESIDE envelope, not fitted physics. Nothing here is
a measured Indian appliance rating.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd

from sharp_action_shield import Device, advance_timers
from sharp_reward_billing import BillingLedger, RewardWeights
from sharp_apcpdcl_tariff import Tariff
from sharp_transition_core import transition_step
from sharp_thermal_rc import load_config, thermal_step

OBS = ['obs_T2M', 'obs_RH2M', 'obs_ALLSKY_SFC_SW_DWN', 'obs_WS10M',
       'obs_grid_percentile', 'obs_grid_peak_severity']
MAX_DEVICES = 28
RELEASE_VERSION = 'SHARP_MASTER_V1'
SEASON_BY_MONTH = {3: 'mar_jun', 4: 'mar_jun', 5: 'mar_jun', 6: 'mar_jun',
                   7: 'jul_oct', 8: 'jul_oct', 9: 'jul_oct', 10: 'jul_oct',
                   11: 'nov_feb', 12: 'nov_feb', 1: 'nov_feb', 2: 'nov_feb'}
POLICIES = ['serve_preferred', 'peak_aware', 'random_binary']


def check(condition, message):
    if not condition:
        raise ValueError(message)


def pick_days(context, split, per_split, seed):
    """Complete 96-step days from this split's own years only."""
    day = context.loc[context.split.eq(split)].copy()
    day['date'] = day.timestamp_ist.dt.strftime('%Y-%m-%d')
    complete = day.groupby('date').size()
    complete = sorted(complete[complete.eq(96)].index)
    check(len(complete) >= per_split, f'{split}: too few complete days')
    by_season = {}
    for date in complete:
        by_season.setdefault(SEASON_BY_MONTH[int(date[5:7])], []).append(date)
    rng = np.random.default_rng(seed)
    chosen, seasons = [], sorted(by_season)
    # Spread days evenly across whatever seasons this split's years contain.
    for i in range(per_split):
        pool = by_season[seasons[i % len(seasons)]]
        chosen.append(pool[int(rng.integers(len(pool)))])
    return sorted(set(chosen)) or complete[:per_split]


def run_episode(*, ds, home, day, date, split, policy, tariff, weights,
                thermal, requests, preferences, keep_json):
    tid = home.template_id
    n = len(ds)
    weekday = int(day.timestamp_ist.iloc[0].weekday())
    season = SEASON_BY_MONTH[int(date[5:7])]
    req = requests.loc[requests.template_id.eq(tid) & requests.season.eq(season)
                       & requests.weekday_number.eq(weekday)].set_index('device_id')
    pref = preferences.loc[preferences.template_id.eq(tid) & preferences.season.eq(season)
                           & preferences.weekday_number.eq(weekday)].set_index('device_id')
    for d in ds.device_id:
        if d not in req.index or d not in pref.index:
            return None  # No service plan for this device/season; skip the episode.

    remaining = np.array([float(req.loc[d, 'requested_hours']) * 4 for d in ds.device_id])
    preferred = np.stack([np.asarray(pref.loc[d, 'preferred_service_fraction'], float)
                          for d in ds.device_id])
    power = ds.operating_power_proxy_w.to_numpy(float)
    protected = ds.service_role.eq('protected_service').to_numpy(bool)
    cycle = ds.dynamics_family.eq('cycle').to_numpy(bool)
    is_ac = ds.appliance_type.eq('air_conditioner').to_numpy(bool)
    has_ac = bool(is_ac.any())
    limit = float(home.sanctioned_load_kw) * 1000
    check(math.isfinite(limit) and limit > 0, 'Nonfinite connection limit')

    outdoor = day.obs_T2M.to_numpy(float)
    band_low, band_high = thermal['comfort_band_c']
    setpoint = thermal['default_setpoint_c']

    budget = remaining.copy()
    current = np.zeros(n, bool)
    elapsed = np.full(n, 4, int)
    # Start the room at the first outdoor reading: no unearned head start.
    indoor = float(outdoor[0])
    rng = np.random.default_rng(
        int(hashlib.sha256(f'{tid}{policy}{date}'.encode()).hexdigest()[:8], 16))
    ledger = BillingLedger(tariff, tid, f'{RELEASE_VERSION}_{date[:7]}',
                           float(home.sanctioned_load_kw),
                           opening_kwh=0, opening_charges_already_booked=False)
    episode = f'{split}:{tid}:{date}:{policy}'
    records, rows = [], []
    invalid_capacity = 0
    reward_sum = 0.0
    discomfort_sum = 0.0
    duty_sum = 0.0

    def observe(t, b, c, kwh, temperature, e=None):
        globals_ = (day.loc[t, OBS].to_numpy(float).tolist() if t < 96
                    else [0.0] * len(OBS))
        globals_ += [t / 96, float(kwh), limit / 1000,
                     float(temperature) / 50.0,
                     float(max(0.0, temperature - band_high)),
                     float(max(0.0, band_low - temperature)),
                     float(has_ac)]
        device = []
        e = elapsed if e is None else e
        for j in range(MAX_DEVICES):
            device.extend([float(b[j] / 4), float(power[j] / 1000), float(c[j]),
                           float(protected[j]), float(cycle[j]), float(e[j] / 96),
                           float(preferred[j, t]) if t < 96 else 0.0,
                           float(is_ac[j])]
                          if j < n else [0.0] * 8)
        vec = globals_ + device
        check(np.isfinite(vec).all(), 'Nonfinite observation')
        return {'features': vec, 'device_present': [j < n for j in range(MAX_DEVICES)]}

    for t in range(96):
        devices = []
        for j in range(n):
            if is_ac[j]:
                # Thermostatic: always available, never budget limited.
                available, active_cycle, must_run = True, False, False
            else:
                available = bool(budget[j] > 1e-9)
                active_cycle = bool(cycle[j] and current[j] and elapsed[j] < 4 and available)
                must_run = bool(protected[j] and preferred[j, t] > 0 and available)
            rated = float(power[j]) if is_ac[j] else float(power[j] * min(1.0, budget[j]))
            devices.append(Device(str(ds.device_id.iloc[j]), bool(current[j]), available,
                                  rated, must_run=must_run,
                                  noninterruptible_cycle_active=active_cycle,
                                  elapsed_state_steps=int(elapsed[j]),
                                  min_on_steps=4 if (cycle[j] and not is_ac[j]) else 0,
                                  min_off_steps=0,
                                  shed_priority=10 if not protected[j] else 0))

        wanted = (preferred[:, t] > 0) & (budget > 1e-9)
        # The AC is wanted whenever the room is above its setpoint.
        wanted = np.where(is_ac, indoor > setpoint, wanted)
        severity = float(day.loc[t, 'obs_grid_peak_severity'])
        if policy == 'peak_aware' and severity > 0.5:
            wanted = wanted & protected
        elif policy == 'random_binary':
            wanted = np.where(is_ac, rng.random(n) < 0.5,
                              (rng.random(n) < 0.5) & (budget > 1e-9))

        step_indoor = indoor
        # transition_step calls physics exactly once, with the executed action,
        # so the last recorded result IS the executed one. Recomputing it would
        # triple the cost of every step.
        last = {}

        def physics(action):
            action = np.asarray(action, float)
            ac_enabled = bool((action * is_ac).sum() > 0) if has_ac else False
            thermal_result = thermal_step(step_indoor, float(outdoor[t]), thermal,
                                          ac_enabled=ac_enabled, setpoint_c=setpoint)
            duty = thermal_result['compressor_duty_fraction']
            next_temperature = thermal_result['next_temperature_c']

            delivered = np.where(is_ac, 0.0, np.minimum(1.0, budget) * action)
            after = np.where(is_ac, budget, np.maximum(0.0, budget - delivered))
            powers = np.where(is_ac, power * action * duty, power * delivered)

            timers = advance_timers(devices, action.astype(int).tolist())
            nxt = []
            for j, (d, z) in enumerate(zip(devices, timers)):
                if is_ac[j]:
                    nxt.append(replace(d, current_on=z['current_on'],
                                       elapsed_state_steps=z['elapsed_state_steps'],
                                       available=True, estimated_on_w=float(power[j]),
                                       must_run=False,
                                       noninterruptible_cycle_active=False))
                else:
                    nxt.append(replace(
                        d, current_on=z['current_on'],
                        elapsed_state_steps=z['elapsed_state_steps'],
                        available=bool(after[j] > 1e-9),
                        estimated_on_w=float(power[j] * min(1.0, after[j])),
                        must_run=bool(protected[j] and t < 95
                                      and preferred[j, t + 1] > 0 and after[j] > 1e-9),
                        noninterruptible_cycle_active=bool(
                            cycle[j] and z['current_on']
                            and z['elapsed_state_steps'] < 4 and after[j] > 1e-9)))
            # Discomfort is only charged where the agent can actually act on it.
            discomfort = (max(0.0, next_temperature - band_high)
                          + max(0.0, band_low - next_temperature)) if has_ac else 0.0
            unmet = float(after[~is_ac].sum() / 4) if t == 95 else 0.0
            result = {'next_devices': nxt, 'appliance_power_w': powers.tolist(),
                    'next_state': observe(t + 1, after, action.astype(bool), ledger.kwh,
                                          next_temperature,
                                          np.array([z['elapsed_state_steps'] for z in timers])),
                    'discomfort_units': float(discomfort),
                    'unmet_service_units': unmet,
                    '_next_temperature_c': next_temperature,
                    '_compressor_duty': duty}
            last['result'] = result
            return result

        record = transition_step(
            ledger=ledger, weights=weights, episode_id=episode, step_id=t,
            devices=devices, requested=wanted.astype(int).tolist(),
            state=observe(t, budget, current, ledger.kwh, indoor),
            physics_step=physics, max_import_w=limit, base_load_w=0.0,
            available_solar_w=0.0, grid_peak_severity=severity,
            policy_source=policy, data_release_version=RELEASE_VERSION,
            terminated=t == 95)

        executed = np.asarray(record['shield']['executed_actions'], float)
        applied = last['result']
        indoor = applied['_next_temperature_c']
        duty_sum += applied['_compressor_duty']
        discomfort_sum += applied['discomfort_units']
        budget = np.where(is_ac, budget,
                          np.maximum(0.0, budget - np.minimum(1.0, budget) * executed))
        current = executed.astype(bool)
        elapsed = np.array([x['elapsed_state_steps'] for x in record['next_device_state']], int)
        record['next_state']['features'][7] = float(ledger.kwh)

        if records and records[-1]['next_state'] != record['state']:
            raise ValueError('State/next-state continuity failed')
        if records and records[-1]['next_device_state'] != record['device_state']:
            raise ValueError('Device-state continuity failed')
        records.append(record)
        invalid_capacity += int(not record['constraint_feasible'])
        reward_sum += record['reward']['reward']

        row = {'episode_id': episode, 'step_id': t, 'split': split,
               'household_id': tid, 'date': date, 'season': season, 'policy': policy,
               'timestamp_ist': day.loc[t, 'timestamp_ist'].isoformat(),
               'state': record['state']['features'],
               'next_state': record['next_state']['features'],
               'action': record['shield']['executed_actions'],
               'requested_action': wanted.astype(int).tolist(),
               'device_present': [j < n for j in range(MAX_DEVICES)],
               'reward': record['reward']['reward'],
               'reward_cost_inr': record['reward']['raw_components']['cost_inr'],
               'reward_grid_peak_kwh': record['reward']['raw_components']['grid_peak_kwh'],
               'reward_discomfort': applied['discomfort_units'],
               'indoor_temperature_c': float(record['state']['features'][9] * 50.0),
               'next_indoor_temperature_c': float(indoor),
               'outdoor_temperature_c': float(outdoor[t]),
               'compressor_duty_fraction': float(applied['_compressor_duty']),
               'grid_import_kwh': record['grid_import_kwh'],
               'terminated': record['terminated'], 'truncated': record['truncated'],
               'done': record['done'],
               'constraint_feasible': record['constraint_feasible']}
        if keep_json:
            row['transition_json'] = json.dumps(record, allow_nan=False,
                                                separators=(',', ':'))
        rows.append(row)

    billed = tariff.components(ledger.kwh, float(home.sanctioned_load_kw))['tariff_subtotal_inr']
    if billed != ledger.booked_since_initialization:
        raise ValueError(f'{episode}: billing reconciliation failed')

    summary = {'split': split, 'household_id': tid, 'date': date, 'season': season,
               'policy': policy, 'devices': n, 'has_air_conditioner': has_ac,
               'steps': 96, 'import_kwh': float(ledger.kwh),
               'tariff_components_inr': float(billed), 'reward_sum': reward_sum,
               'mean_discomfort_c': discomfort_sum / 96,
               'mean_compressor_duty': duty_sum / 96,
               'unserved_hours': float(budget[~is_ac].sum() / 4),
               'capacity_violation_steps': invalid_capacity}
    return rows, summary


def run(root, households, days_per_split, policies, keep_json, seed):
    base = root / 'data/processed/simulator_devices_v1/unknown_quantity_one'
    paths = {
        'models': base / 'baseline_power_v1/device_power_models.parquet',
        'requests': base / 'service_plans_v1/weekly_service_requests.parquet',
        'preferences': base / 'service_plans_v1/preferred_service_slots.parquet',
        'households': root / 'data/processed/appliance_inputs_v1/ap_households_with_splits_v1.parquet',
        'context': root / 'data/processed/simulator_context_v1/regional_grid_guntur_weather_15min_v1.parquet',
        'thermal_config': root / 'configs/thermal/sharp_thermal_rc_v1.json',
        'tariff_config': root / 'configs/tariffs/apcpdcl_2025_26_verified_components.json',
    }
    for name, path in paths.items():
        check(path.exists(), f'Missing input: {name} -> {path}')

    models = pd.read_parquet(paths['models'])
    requests = pd.read_parquet(paths['requests'])
    preferences = pd.read_parquet(paths['preferences'])
    hh = pd.read_parquet(paths['households'])
    context = pd.read_parquet(paths['context']).sort_values('timestamp_ist')
    context['timestamp_ist'] = pd.to_datetime(context.timestamp_ist)
    tariff = Tariff.load(paths['tariff_config'])
    thermal = load_config(paths['thermal_config'])

    # Household-disjoint splits are inherited from the upstream template build.
    overlap = hh.groupby('template_id').split.nunique()
    check(int(overlap.max()) == 1, 'A household appears in more than one split')

    out = root / 'data/processed/sharp_rl_transitions_v1'
    out.mkdir(parents=True, exist_ok=True)

    day_by_split = {s: pick_days(context, s, days_per_split, seed)
                    for s in ['train', 'validation', 'test']}
    print('Context days per split:',
          {s: len(v) for s, v in day_by_split.items()}, flush=True)

    frames, summaries, skipped = [], [], 0
    for split in ['train', 'validation', 'test']:
        pool = hh.loc[hh.split.eq(split)
                      & pd.to_numeric(hh.sanctioned_load_kw, errors='coerce').gt(0)]
        pool = pool.sort_values('template_id')
        if households:
            pool = pool.head(households)
        dates = day_by_split[split]
        print(f'{split}: {len(pool)} households x {len(dates)} days x '
              f'{len(policies)} policies', flush=True)
        for count, (_, home) in enumerate(pool.iterrows(), 1):
            ds = models.loc[models.template_id.eq(home.template_id)]
            ds = ds.sort_values('device_id').reset_index(drop=True)
            if not 0 < len(ds) <= MAX_DEVICES:
                skipped += 1
                continue
            for date in dates:
                day = context.loc[context.timestamp_ist.dt.strftime('%Y-%m-%d').eq(date)
                                  & context.split.eq(split)].reset_index(drop=True)
                if len(day) != 96:
                    continue
                for policy in policies:
                    result = run_episode(
                        ds=ds, home=home, day=day, date=date, split=split,
                        policy=policy, tariff=tariff, weights=WEIGHTS,
                        thermal=thermal, requests=requests,
                        preferences=preferences, keep_json=keep_json)
                    if result is None:
                        skipped += 1
                        continue
                    rows, summary = result
                    frames.extend(rows)
                    summaries.append(summary)
            if count % 25 == 0:
                print(f'  {split}: {count}/{len(pool)} households, '
                      f'{len(frames):,} transitions', flush=True)

    check(frames, 'No transitions were generated')
    transitions = pd.DataFrame(frames)
    episodes = pd.DataFrame(summaries)

    # Leakage gate: a household must never appear in two splits.
    spread = transitions.groupby('household_id').split.nunique()
    check(int(spread.max()) == 1, 'Household leakage across splits')
    # Leakage gate: a calendar date must never appear in two splits.
    date_spread = transitions.groupby('date').split.nunique()
    check(int(date_spread.max()) == 1, 'Context date leakage across splits')

    lo = float(transitions.indoor_temperature_c.min())
    hi = float(transitions.next_indoor_temperature_c.max())
    check(5.0 < lo and hi < 55.0,
          f'Indoor temperature left a plausible range: {lo:.2f} to {hi:.2f} C')

    feature_count = len(transitions.state.iloc[0])
    check(transitions.state.map(len).eq(feature_count).all(), 'Ragged state vectors')
    check(transitions.next_state.map(len).eq(feature_count).all(), 'Ragged next_state')

    transitions.to_parquet(out / 'rl_transitions.parquet', index=False,
                           compression='zstd')
    episodes.to_csv(out / 'episode_summary.csv', index=False)

    schema = {
        'global_features': OBS + ['fraction_of_day', 'billing_period_kwh',
                                  'connection_limit_kw', 'indoor_temperature_c_div50',
                                  'degrees_above_comfort_band',
                                  'degrees_below_comfort_band',
                                  'household_has_air_conditioner'],
        'device_features': ['remaining_service_hours', 'power_proxy_kw', 'current_on',
                            'protected_service', 'cycle_type',
                            'elapsed_state_steps_divided_by_96',
                            'preferred_service_fraction', 'is_air_conditioner'],
        'maximum_device_slots': MAX_DEVICES,
        'device_order': 'device_id ascending within household',
        'padding': 'zero features; device_present column carries the mask',
        'feature_count': feature_count,
        'action_space': 'per-device binary enable; AC enable is a thermostat call, '
                        'not a compressor switch',
        'thermal_config': thermal['config_id'],
        'thermal_scenario': thermal['scenario'],
    }
    (out / 'feature_schema.json').write_text(json.dumps(schema, indent=2), encoding='utf-8')

    ac_rows = transitions[transitions.compressor_duty_fraction.gt(0)]
    report = {
        'status': 'SHARP_RL_TRANSITIONS_GENERATED',
        'release_version': RELEASE_VERSION,
        'transitions': int(len(transitions)),
        'episodes': int(len(episodes)),
        'households': int(transitions.household_id.nunique()),
        'context_days': int(transitions.date.nunique()),
        'policies': sorted(transitions.policy.unique().tolist()),
        'seasons': sorted(transitions.season.unique().tolist()),
        'feature_count': feature_count,
        'transitions_by_split': transitions.split.value_counts().to_dict(),
        'households_by_split': transitions.groupby('split').household_id.nunique().to_dict(),
        'households_with_air_conditioner': int(
            episodes[episodes.has_air_conditioner].household_id.nunique()),
        'billing_reconciliation': 'PASS',
        'state_continuity': 'PASS',
        'household_split_leakage': 'NONE',
        'context_date_split_leakage': 'NONE',
        'capacity_violation_steps': int((~transitions.constraint_feasible).sum()),
        'steps_with_compressor_running': int(len(ac_rows)),
        'mean_compressor_duty_when_running': (float(ac_rows.compressor_duty_fraction.mean())
                                              if len(ac_rows) else 0.0),
        'indoor_temperature_c': {
            'min': float(transitions.indoor_temperature_c.min()),
            'mean': float(transitions.indoor_temperature_c.mean()),
            'max': float(transitions.indoor_temperature_c.max())},
        'mean_reward': float(transitions.reward.mean()),
        'skipped_episodes': int(skipped),
        'reward_weights': {'cost': WEIGHTS.cost_per_inr,
                           'peak': WEIGHTS.grid_peak_per_kwh,
                           'discomfort': WEIGHTS.discomfort_per_unit,
                           'switching': WEIGHTS.switching_per_event,
                           'unmet_service_hours': WEIGHTS.unmet_service_per_unit},
        'inputs_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest()
                          for k, p in paths.items()},
        'scope_and_limits': [
            'Appliance power values are proxies and declared assumptions, not measured Indian ratings.',
            'Thermal parameters are declared assumptions bounded by the RESIDE envelope, not fitted physics.',
            'No causal AC cooling effect is established anywhere in this pipeline.',
            'REFIT is UK evidence used for appliance behaviour, not Indian household data.',
            'No solar, battery or outage model; export compensation is not included.',
            'No hidden background load; only modelled devices contribute demand.',
            'Human attention is absent, so all non-override feedback stays censored.',
            'Comfort is charged only for households that own an air conditioner.',
            'FY2025-26 APCPDCL tariff is applied as a scenario to other-year context.',
            'Infeasible transitions are preserved and flagged, never silently dropped.',
        ],
        'full_simulator_ready': True,
        'master_release_ready': True,
    }
    (out / 'transition_validation.json').write_text(json.dumps(report, indent=2),
                                                    encoding='utf-8')
    print('\nSHARP RL TRANSITIONS GENERATED')
    for key in ['transitions', 'episodes', 'households', 'context_days',
                'feature_count', 'transitions_by_split', 'households_by_split',
                'capacity_violation_steps', 'steps_with_compressor_running',
                'indoor_temperature_c', 'mean_reward', 'skipped_episodes']:
        print(f'  {key}: {report[key]}')
    print('Output:', out)
    return report


# Reward weights for the release. Discomfort is now a real signal because the
# thermal component produces real comfort-band excursions.
WEIGHTS = RewardWeights(cost_per_inr=1, grid_peak_per_kwh=1, discomfort_per_unit=0.5,
                        switching_per_event=0.01, unmet_service_per_unit=10)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--households', type=int, default=0,
                   help='0 means every household with a reported connection load')
    p.add_argument('--days-per-split', type=int, default=3)
    p.add_argument('--policies', nargs='+', default=POLICIES)
    p.add_argument('--keep-json', action='store_true',
                   help='embed the full nested transition record per row')
    p.add_argument('--seed', type=int, default=20260913)
    a = p.parse_args()
    run(a.root.resolve(), a.households, a.days_per_split, a.policies,
        a.keep_json, a.seed)
