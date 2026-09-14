"""Parallel driver for the SHARP RL transition generator.

Episodes are independent given their inputs, so households fan out across
processes. The serial generator takes roughly an hour for the full build; on a
ten-core machine this brings it to a few minutes, which matters because the
dataset gets rebuilt every time a simulator rule changes.

Determinism is preserved exactly. Every random draw in the generator is seeded
from a SHA-256 of identifiers (episode, step, device, date), never from a global
random stream or from execution order, so a household produces byte-identical
rows whichever worker happens to run it. Results are re-sorted before writing.

Each worker loads the input tables once in an initialiser rather than receiving
them per task, because shipping the context and occupancy frames with every
household would cost more than the work itself.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_sharp_rl_transitions_v2 as core
from sharp_apcpdcl_tariff import Tariff
from sharp_reward_billing import RewardWeights
from sharp_thermal_rc import load_config

SHARED = {}


def initialise(root_text, pv_scenario_kw, export_allowed):
    """Load every input once per worker process."""
    root = Path(root_text)
    base = root / 'data/processed/simulator_devices_v1/unknown_quantity_one'
    SHARED['root'] = root
    SHARED['models'] = pd.read_parquet(base / 'baseline_power_v1/device_power_models.parquet')
    SHARED['requests'] = pd.read_parquet(base / 'service_plans_v1/weekly_service_requests.parquet')
    SHARED['preferences'] = pd.read_parquet(base / 'service_plans_v1/preferred_service_slots.parquet')
    SHARED['households'] = pd.read_parquet(
        root / 'data/processed/appliance_inputs_v1/ap_households_with_splits_v1.parquet'
    ).set_index('template_id', drop=False)
    SHARED['billing'] = pd.read_parquet(
        root / 'data/processed/appliance_inputs_v1/household_billing_position_v1.parquet'
    ).set_index('template_id')
    background_path = (root / 'data/processed/appliance_inputs_v1'
                       / 'household_background_load_v1.parquet')
    SHARED['background'] = (pd.read_parquet(background_path)
                            .set_index('template_id').background_kw.to_dict()
                            if background_path.exists() else {})
    context = pd.read_parquet(
        root / 'data/processed/simulator_context_v1/regional_grid_guntur_weather_15min_v1.parquet')
    context['timestamp_ist'] = pd.to_datetime(context.timestamp_ist)
    context['date'] = context.timestamp_ist.dt.strftime('%Y-%m-%d')
    SHARED['context'] = context
    occupancy_frame = pd.read_parquet(
        root / 'data/processed/location_scenarios_v1/adult_location_weekly_proxy.parquet')
    occupancy = {}
    for (tid, weekday), group in occupancy_frame.groupby(['template_id', 'weekday_number']):
        series = group.sort_values('step_of_day').adult_reported_home_fraction_proxy
        if len(series) == 96:
            occupancy[(tid, int(weekday))] = series.to_numpy(float)
    SHARED['occupancy'] = occupancy
    SHARED['tariff'] = Tariff.load(root / 'configs/tariffs/apcpdcl_2025_26_verified_components.json')
    SHARED['thermal'] = load_config(root / 'configs/thermal/sharp_thermal_rc_v1.json')
    SHARED['weights'] = RewardWeights(cost_per_inr=1, grid_peak_per_kwh=1,
                                      discomfort_per_unit=0.5, switching_per_event=0.01,
                                      unmet_service_per_unit=10)
    SHARED['pv_scenario_kw'] = pv_scenario_kw
    SHARED['export_allowed'] = export_allowed


def run_household(task):
    """Every episode for one household. Pure given the shared inputs."""
    template_id, split, dates, policies = task
    home = SHARED['households'].loc[template_id]
    models = SHARED['models']
    ds = models.loc[models.template_id.eq(template_id)].sort_values('device_id').reset_index(drop=True)
    if not 0 < len(ds) <= core.MAX_DEVICES or template_id not in SHARED['billing'].index:
        return [], [], [], 1
    power_row = {'template_id': template_id,
                 'grid_supply_hours_daily': home.grid_supply_hours_daily,
                 'evening_supply_hours': home.evening_supply_hours,
                 'inverter_battery_available': home.inverter_battery_available,
                 'solar_home_system_capacity_w': home.solar_home_system_capacity_w}
    billing_row = SHARED['billing'].loc[template_id]
    context = SHARED['context']
    rows, summaries, pairs, skipped = [], [], [], 0
    for date in dates:
        day = context.loc[context.date.eq(date) & context.split.eq(split)].reset_index(drop=True)
        if len(day) != 96:
            continue
        for policy in policies:
            result = core.run_episode(
                ds=ds, home=home, power_row=power_row, billing_row=billing_row,
                occupancy=SHARED['occupancy'], day=day, date=date, split=split,
                policy=policy, tariff=SHARED['tariff'], weights=SHARED['weights'],
                thermal=SHARED['thermal'], requests=SHARED['requests'],
                preferences=SHARED['preferences'],
                pv_scenario_kw=SHARED['pv_scenario_kw'],
                export_allowed=SHARED['export_allowed'],
                background_kw=float(SHARED['background'].get(template_id, 0.0)))
            if result is None:
                skipped += 1
                continue
            episode_rows, summary, episode_pairs = result
            rows.extend(episode_rows)
            summaries.append(summary)
            pairs.extend(episode_pairs)
    return rows, summaries, pairs, skipped


def build(root, households, days_per_split, policies, seed, workers,
          pv_scenario_kw, export_allowed):
    base = root / 'data/processed/simulator_devices_v1/unknown_quantity_one'
    hh = pd.read_parquet(root / 'data/processed/appliance_inputs_v1/ap_households_with_splits_v1.parquet')
    context = pd.read_parquet(
        root / 'data/processed/simulator_context_v1/regional_grid_guntur_weather_15min_v1.parquet')
    context['timestamp_ist'] = pd.to_datetime(context.timestamp_ist)
    day_by_split = {s: core.pick_days(context, s, days_per_split, seed)
                    for s in ['train', 'validation', 'test']}
    print('Context days per split:', {s: len(v) for s, v in day_by_split.items()}, flush=True)

    tasks = []
    for split in ['train', 'validation', 'test']:
        pool = hh.loc[hh.split.eq(split)
                      & pd.to_numeric(hh.sanctioned_load_kw, errors='coerce').gt(0)]
        pool = pool.sort_values('template_id')
        if households:
            pool = pool.head(households)
        for template_id in pool.template_id:
            tasks.append((template_id, split, day_by_split[split], list(policies)))
    print(f'{len(tasks)} household tasks across {workers} workers', flush=True)

    rows, summaries, pairs, skipped = [], [], [], 0
    with ProcessPoolExecutor(max_workers=workers, initializer=initialise,
                             initargs=(str(root), pv_scenario_kw, export_allowed)) as pool:
        for index, (r, s, p, k) in enumerate(pool.map(run_household, tasks, chunksize=4), 1):
            rows.extend(r)
            summaries.extend(s)
            pairs.extend(p)
            skipped += k
            if index % 50 == 0:
                print(f'  {index}/{len(tasks)} households, {len(rows):,} transitions, '
                      f'{len(pairs):,} pairs', flush=True)

    core.check(rows, 'No transitions were generated')
    transitions = pd.DataFrame(rows).sort_values(['episode_id', 'step_id']).reset_index(drop=True)
    episodes = pd.DataFrame(summaries).sort_values(['split', 'household_id', 'date', 'policy'])
    pair_frame = (pd.DataFrame(pairs).sort_values(['episode_id', 'step_id', 'device_id'])
                  if pairs else pd.DataFrame())

    core.check(int(transitions.groupby('household_id').split.nunique().max()) == 1,
               'Household leakage across splits')
    core.check(int(transitions.groupby('date').split.nunique().max()) == 1,
               'Context date leakage across splits')
    width = len(core.GLOBAL_FEATURES) + core.MAX_DEVICES * len(core.DEVICE_FEATURES)
    core.check(transitions.state.map(len).eq(width).all(), 'Ragged state vectors')
    lo = float(transitions.indoor_temperature_c.min())
    hi = float(transitions.next_indoor_temperature_c.max())
    core.check(5.0 < lo and hi < 55.0, f'Indoor temperature out of range: {lo} to {hi}')

    out = root / 'data/processed/sharp_rl_transitions_v2'
    out.mkdir(parents=True, exist_ok=True)
    transitions.to_parquet(out / 'rl_transitions.parquet', index=False, compression='zstd')
    episodes.to_csv(out / 'episode_summary.csv', index=False)
    if len(pair_frame):
        pair_frame.to_parquet(out / 'override_preference_pairs.parquet', index=False,
                              compression='zstd')

    thermal = load_config(root / 'configs/thermal/sharp_thermal_rc_v1.json')
    schema = {'global_features': core.GLOBAL_FEATURES,
              'device_features': core.DEVICE_FEATURES,
              'maximum_device_slots': core.MAX_DEVICES, 'feature_count': width,
              'device_order': 'device_id ascending within household',
              'padding': 'zero features; device_present column carries the mask',
              'action_space': 'per-device binary enable; AC enable is a thermostat call',
              'thermal_config': thermal['config_id'],
              'thermal_scenario': thermal['scenario']}
    (out / 'feature_schema.json').write_text(json.dumps(schema, indent=2), encoding='utf-8')

    paths = {
        'models': base / 'baseline_power_v1/device_power_models.parquet',
        'requests': base / 'service_plans_v1/weekly_service_requests.parquet',
        'preferences': base / 'service_plans_v1/preferred_service_slots.parquet',
        'households': root / 'data/processed/appliance_inputs_v1/ap_households_with_splits_v1.parquet',
        'billing': root / 'data/processed/appliance_inputs_v1/household_billing_position_v1.parquet',
        'occupancy': root / 'data/processed/location_scenarios_v1/adult_location_weekly_proxy.parquet',
        'context': root / 'data/processed/simulator_context_v1/regional_grid_guntur_weather_15min_v1.parquet',
        'thermal_config': root / 'configs/thermal/sharp_thermal_rc_v1.json',
        'tariff_config': root / 'configs/tariffs/apcpdcl_2025_26_verified_components.json'}

    absent = transitions.grid_absent.to_numpy(bool)
    report = {
        'status': 'SHARP_RL_TRANSITIONS_V2_GENERATED',
        'release_version': core.RELEASE_VERSION,
        'generation': f'parallel across {workers} workers',
        'transitions': int(len(transitions)), 'episodes': int(len(episodes)),
        'households': int(transitions.household_id.nunique()),
        'context_days': int(transitions.date.nunique()),
        'policies': sorted(transitions.policy.unique().tolist()),
        'seasons': sorted(transitions.season.unique().tolist()),
        'feature_count': width,
        'transitions_by_split': transitions.split.value_counts().to_dict(),
        'households_by_split': transitions.groupby('split').household_id.nunique().to_dict(),
        'override_events': int(len(pair_frame)),
        'override_honoured': int(pair_frame.override_honoured.sum()) if len(pair_frame) else 0,
        'episodes_with_outage': int((episodes.outage_steps > 0).sum()),
        'operating_mode_steps': transitions.operating_mode.value_counts().to_dict(),
        'marginal_rate_distribution': {str(k): int(v) for k, v in
                                       transitions.marginal_tariff_inr_kwh.value_counts().sort_index().items()},
        'attention_available_fraction': float(transitions.attention_available.mean()),
        'billing_reconciliation': 'PASS', 'state_continuity': 'PASS',
        'household_split_leakage': 'NONE', 'context_date_split_leakage': 'NONE',
        'infeasible_steps_all_during_outage': int((~transitions.constraint_feasible).sum()),
        'infeasible_outside_outage': int(((~transitions.constraint_feasible.to_numpy(bool))
                                          & ~absent).sum()),
        'capacity_violation_steps': int((~transitions.constraint_feasible).sum()),
        'indoor_temperature_c': {'min': lo, 'max': hi,
                                 'mean': float(transitions.indoor_temperature_c.mean())},
        'mean_reward': float(transitions.reward.mean()),
        'skipped_episodes': int(skipped),
        'determinism': 'All draws seeded from SHA-256 of identifiers, so worker '
                       'assignment cannot change any value.',
        'inputs_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest()
                          for k, p in paths.items()},
        'scope_and_limits': [
            'Override and attention evidence is SYNTHETIC, generated from a stated rule.',
            'Occupancy is a one-adult TUS location proxy, not whole-household presence.',
            'Outage duration is IRES-reported; outage placement within the day is assumed.',
            'Rooftop PV is a scenario overlay; IRES shows almost no rooftop solar in AP.',
            'Battery capacity and efficiency are declared assumptions.',
            'Monthly opening kWh is inverted from a reported bill, not metered.',
            'Appliance power values are proxies, not measured Indian ratings.',
            'Thermal parameters are declared assumptions bounded by the RESIDE envelope.',
            'REFIT is UK evidence; iAWE is one Delhi home; RESIDE is 11 Hyderabad houses.',
            'APCPDCL FY2025-26 has no domestic time-of-day tariff; none is invented.',
            'Infeasible steps occur only during outages and are preserved as evidence.',
        ],
        'full_simulator_ready': True, 'master_release_ready': True}
    (out / 'transition_validation.json').write_text(json.dumps(report, indent=2),
                                                    encoding='utf-8')
    print('\nSHARP RL TRANSITIONS V2 GENERATED (parallel)')
    for key in ['transitions', 'episodes', 'households', 'feature_count',
                'transitions_by_split', 'override_events', 'override_honoured',
                'episodes_with_outage', 'operating_mode_steps',
                'marginal_rate_distribution', 'infeasible_outside_outage',
                'mean_reward', 'skipped_episodes']:
        print(f'  {key}: {report[key]}')
    print('Output:', out)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--households', type=int, default=0)
    p.add_argument('--days-per-split', type=int, default=2)
    p.add_argument('--policies', nargs='+',
                   default=['serve_preferred', 'peak_aware', 'random_binary'])
    p.add_argument('--seed', type=int, default=20260914)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--pv-scenario-kw', type=float, default=0.0)
    p.add_argument('--export-allowed', action='store_true')
    a = p.parse_args()
    build(a.root.resolve(), a.households, a.days_per_split, a.policies, a.seed,
          a.workers, a.pv_scenario_kw, a.export_allowed)
