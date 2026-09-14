"""Generate the full SHARP RL transition layer.

Supersedes generate_sharp_rl_transitions_v1.py, which produced a valid demand
response scheduling dataset but supported none of the project's three novelty
claims. This version adds, in one pass:

  1. OVERRIDES AND ATTENTION. The shield's human_actions plumbing was built and
     unused. Overrides now flow through it, and every intervention is emitted as
     a weighted preference pair for a Bradley-Terry reward model. Synthetic, and
     labelled as such everywhere.

  2. SOURCE AND SINK AWARENESS. Outages, inverter batteries and PV give a
     self-sufficient fraction, a battery state and an operating mode. Where
     shedding a self-generated load has zero value (battery full, export
     blocked) the SHIELD removes the shed action rather than discouraging it.

  3. REALISTIC BILLING POSITION. Episodes previously opened every billing period
     at zero kWh, so every household stayed in the lowest tariff slab and never
     saw a marginal rate above 1.90 INR per kWh. Each household now opens its
     month where its reported bill says it sits.

  4. OCCUPANCY, TARIFF AND MEMORY IN THE STATE. Occupancy and attention from the
     TUS proxy, the marginal tariff rate, the operating mode, and per-device
     anti-fatigue memory (steps since shed, overrides so far).

Provenance, unchanged from the project's rules: REFIT is UK evidence, iAWE is
one Delhi home, RESIDE is 11 Hyderabad houses, thermal parameters are declared
assumptions, and appliance power is a proxy rather than a measured rating.
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
from sharp_human_model import decide_overrides, attention_available
from sharp_power_system import (build_household_power, dispatch, outage_schedule,
                                on_inverter_circuit, pv_generation_kw,
                                shed_value_per_kw, MODE_GRID_IMPORT,
                                MODE_SELF_SUFFICIENT, MODE_ISLANDED)

OBS = ['obs_T2M', 'obs_RH2M', 'obs_ALLSKY_SFC_SW_DWN', 'obs_WS10M',
       'obs_grid_percentile', 'obs_grid_peak_severity']
MAX_DEVICES = 28
RELEASE_VERSION = 'SHARP_MASTER_V2'
SEASON_BY_MONTH = {3: 'mar_jun', 4: 'mar_jun', 5: 'mar_jun', 6: 'mar_jun',
                   7: 'jul_oct', 8: 'jul_oct', 9: 'jul_oct', 10: 'jul_oct',
                   11: 'nov_feb', 12: 'nov_feb', 1: 'nov_feb', 2: 'nov_feb'}
POLICIES = ['serve_preferred', 'peak_aware', 'random_binary']

GLOBAL_FEATURES = OBS + [
    'fraction_of_day', 'month_to_date_kwh_div500', 'connection_limit_kw',
    'indoor_temperature_c_div50', 'degrees_above_comfort_band',
    'degrees_below_comfort_band', 'household_has_air_conditioner',
    'occupancy_adult_home_fraction', 'attention_available',
    'marginal_tariff_inr_kwh_div10',
    'mode_grid_import', 'mode_self_sufficient', 'mode_islanded',
    'self_sufficient_fraction', 'battery_state_of_charge',
    'pv_generation_kw', 'unserved_demand_kw', 'grid_absent',
    'recent_override_count_div10']
DEVICE_FEATURES = ['remaining_service_hours', 'power_proxy_kw', 'current_on',
                   'protected_service', 'cycle_type',
                   'elapsed_state_steps_divided_by_96', 'preferred_service_fraction',
                   'is_air_conditioner', 'steps_since_shed_div96',
                   'device_override_count_div10']


def check(condition, message):
    if not condition:
        raise ValueError(message)


def pick_days(context, split, per_split, seed):
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
    for i in range(per_split):
        pool = by_season[seasons[i % len(seasons)]]
        chosen.append(pool[int(rng.integers(len(pool)))])
    return sorted(set(chosen)) or complete[:per_split]


def run_episode(*, ds, home, power_row, billing_row, occupancy, day, date, split,
                policy, tariff, weights, thermal, requests, preferences,
                pv_scenario_kw, export_allowed, background_kw=0.0):
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
            return None

    occupancy_day = occupancy.get((tid, weekday))
    if occupancy_day is None:
        return None

    remaining = np.array([float(req.loc[d, 'requested_hours']) * 4 for d in ds.device_id])
    preferred = np.stack([np.asarray(pref.loc[d, 'preferred_service_fraction'], float)
                          for d in ds.device_id])
    power = ds.operating_power_proxy_w.to_numpy(float)
    # Necessity loads are masked out of the shed action entirely. SHARP restricts
    # luxury load; it never cuts a household's essential supply.
    protected = ds.is_necessity.to_numpy(bool)
    cycle = ds.dynamics_family.eq('cycle').to_numpy(bool)
    is_ac = ds.appliance_type.eq('air_conditioner').to_numpy(bool)
    supports_reduced = ds.supports_reduced.to_numpy(bool)
    reduced_fraction = ds.reduced_power_fraction.to_numpy(float)
    on_inverter = ds.appliance_type.map(on_inverter_circuit).to_numpy(bool)
    device_ids = ds.device_id.astype(str).tolist()
    has_ac = bool(is_ac.any())
    sanctioned_w = float(home.sanctioned_load_kw) * 1000
    check(math.isfinite(sanctioned_w) and sanctioned_w > 0, 'Nonfinite connection limit')
    # Sanctioned load is a CONTRACTUAL figure, not a breaker that trips. The
    # median Andhra Pradesh household here is sanctioned at 0.52 kW and 222 of
    # 464 are under 0.5 kW, so a household's own lights and fans can exceed it.
    # Treating it as a hard physical cap would make necessity supply infeasible,
    # which contradicts the whole design: SHARP restricts luxury load, it never
    # cuts a household's essential power.
    #
    # So the PHYSICAL capacity used for feasibility is at least enough to carry
    # the household's own necessity load, and exceeding the sanctioned figure is
    # priced through the grid-peak reward term instead of being forbidden.
    # Declared assumption; the 1.1 factor is headroom, not a measured rating.
    necessity_w = float(ds.loc[ds.is_necessity, 'operating_power_proxy_w'].sum())
    # Unmodelled household load, calibrated to this household's own reported
    # bill. It is not controllable and never appears as an agent action, but it
    # is real consumption and must be billed and counted against capacity.
    base_w = float(background_kw) * 1000.0
    check(math.isfinite(base_w) and base_w >= 0, 'Nonfinite background load')
    # A noninterruptible cycle cannot be stopped once started, so the connection
    # must also carry the largest one the household owns. Without this a rice
    # cooker mid-cycle plus the household's own lights and fans could exceed the
    # limit by a few watts, leaving the shield to choose between breaking a cycle
    # and shedding an essential. A supply that could not run a household's
    # essentials alongside one appliance cycle would not have that appliance.
    # A household with no cycle appliance yields an empty max, which pandas
    # returns as NaN. `nan or 0.0` is nan, because nan is truthy, which silently
    # destroyed the whole capacity headroom for those households.
    cycle_series = ds.loc[ds.dynamics_family.eq('cycle'), 'operating_power_proxy_w']
    cycle_w = float(cycle_series.max()) if len(cycle_series) else 0.0
    check(math.isfinite(cycle_w), 'Nonfinite cycle power')
    limit = max(sanctioned_w, (necessity_w + cycle_w) * 1.1 + base_w)

    household = build_household_power(power_row, pv_scenario_kw=pv_scenario_kw,
                                      export_allowed=export_allowed)
    outage = outage_schedule(household, date)

    outdoor = day.obs_T2M.to_numpy(float)
    irradiance = day.obs_ALLSKY_SFC_SW_DWN.to_numpy(float)
    band_low, band_high = thermal['comfort_band_c']
    setpoint = thermal['default_setpoint_c']

    # Open the billing month where this household's reported bill says it sits.
    day_of_month = int(date[8:10])
    opening_kwh = float(billing_row['monthly_kwh']) * max(0, day_of_month - 1) / 30.0

    budget = remaining.copy()
    current = np.zeros(n, bool)
    elapsed = np.full(n, 4, int)
    steps_since_shed = np.full(n, 96, int)
    device_override_count = np.zeros(n, int)
    indoor = float(outdoor[0])
    battery_kwh = household.battery_capacity_kwh * 0.5
    override_total = 0

    ledger = BillingLedger(tariff, tid, f'{RELEASE_VERSION}_{date[:7]}',
                           float(home.sanctioned_load_kw),
                           opening_kwh=opening_kwh,
                           opening_charges_already_booked=day_of_month > 1)
    episode = f'{split}:{tid}:{date}:{policy}'
    # Overrides decided at step t only reach the controller at t + latency.
    # pending[step] = {device_id: OverrideEvent}
    pending = {}
    records, rows, pairs = [], [], []
    invalid_capacity = 0
    reward_sum = 0.0

    def observe(t, b, c, kwh, temperature, flows, worthless,
                shed_memory, override_counts, e=None):
        occupancy_fraction = float(occupancy_day[t]) if t < 96 else 0.0
        globals_ = (day.loc[t, OBS].to_numpy(float).tolist() if t < 96
                    else [0.0] * len(OBS))
        mode = flows['operating_mode']
        globals_ += [
            t / 96, float(kwh) / 500.0, limit / 1000,
            float(temperature) / 50.0,
            float(max(0.0, temperature - band_high)),
            float(max(0.0, band_low - temperature)),
            float(has_ac),
            float(occupancy_fraction), float(attention_available(occupancy_fraction)),
            float(tariff.next_unit_rate(kwh)) / 10.0,
            float(mode == MODE_GRID_IMPORT), float(mode == MODE_SELF_SUFFICIENT),
            float(mode == MODE_ISLANDED),
            float(flows['self_sufficient_fraction']),
            float(flows['battery_kwh'] / household.battery_capacity_kwh)
            if household.battery_capacity_kwh > 0 else 0.0,
            float(flows['pv_direct_kw']), float(flows['unserved_kw']),
            float(flows['grid_absent']),
            float(override_total) / 10.0]
        device = []
        e = elapsed if e is None else e
        for j in range(MAX_DEVICES):
            device.extend([float(b[j] / 4), float(power[j] / 1000), float(c[j]),
                           float(protected[j]), float(cycle[j]), float(e[j] / 96),
                           float(preferred[j, t]) if t < 96 else 0.0,
                           float(is_ac[j]), float(shed_memory[j] / 96),
                           float(override_counts[j]) / 10.0]
                          if j < n else [0.0] * len(DEVICE_FEATURES))
        vec = globals_ + device
        check(np.isfinite(vec).all(), 'Nonfinite observation')
        return {'features': vec, 'device_present': [j < n for j in range(MAX_DEVICES)]}

    # Resolve the opening power state so the first observation is truthful.
    opening_flows = dispatch(demand_kw=0.0, pv_kw=0.0, battery_kwh=battery_kwh,
                             household=household, grid_absent=outage[0])

    for t in range(96):
        occupancy_fraction = float(occupancy_day[t])
        present = attention_available(occupancy_fraction)
        grid_absent = bool(outage[t])
        pv_kw = pv_generation_kw(household, irradiance[t])
        step_battery_preview = battery_kwh

        devices = []
        for j in range(n):
            if is_ac[j]:
                available, active_cycle, must_run = True, False, False
            else:
                available = bool(budget[j] > 1e-9)
                active_cycle = bool(cycle[j] and current[j] and elapsed[j] < 4 and available)
                must_run = bool(protected[j] and preferred[j, t] > 0 and available)
            # During a blackout nothing can be mandatory: a protected load that
            # cannot be powered is simply off, and the shield must be free to
            # shed it to fit the inverter. The service goes UNSERVED, which is
            # the honest penalty, rather than being billed as a grid import the
            # household could not physically have drawn.
            if grid_absent:
                must_run = False
                active_cycle = False
            rated = float(power[j]) if is_ac[j] else float(power[j] * min(1.0, budget[j]))
            devices.append(Device(device_ids[j], bool(current[j]), available,
                                  rated, must_run=must_run,
                                  supports_reduced=bool(supports_reduced[j]),
                                  estimated_reduced_w=float(rated * reduced_fraction[j]),
                                  noninterruptible_cycle_active=active_cycle,
                                  elapsed_state_steps=int(elapsed[j]),
                                  min_on_steps=(0 if grid_absent
                                                else (4 if (cycle[j] and not is_ac[j]) else 0)),
                                  min_off_steps=0,
                                  shed_priority=10 if not protected[j] else 0))

        wanted = (preferred[:, t] > 0) & (budget > 1e-9)
        wanted = np.where(is_ac, indoor > setpoint, wanted)
        # What the OCCUPANT wants, before any policy or shield decision. An
        # override is the user's correction of the controller, so denial must be
        # measured against this, not against what the policy already chose.
        user_wanted = wanted.copy()

        from sharp_power_system import INVERTER_MAX_DISCHARGE_KW, STEP_HOURS
        battery_available_kw = (min(INVERTER_MAX_DISCHARGE_KW,
                                    step_battery_preview / STEP_HOURS)
                                if household.battery_capacity_kwh > 0 else 0.0)
        non_grid_w = (pv_kw + battery_available_kw) * 1000.0
        import_ceiling_w = 0.0 if grid_absent else limit
        severity = float(day.loc[t, 'obs_grid_peak_severity'])
        rng_local = np.random.default_rng(
            int(hashlib.sha256(f'{episode}|{t}'.encode()).hexdigest()[:8], 16))
        if policy == 'peak_aware' and severity > 0.5:
            # Under grid stress: dim anything that can be dimmed, shed only
            # discretionary loads that cannot. A necessity is never shed, but it
            # CAN be dimmed - a fan on a lower speed or a dimmed light keeps the
            # service while cutting demand, which is the whole reason the third
            # action level exists.
            wanted_level = np.where(
                supports_reduced & wanted, 2,
                np.where(protected, wanted.astype(int), 0))
        elif policy == 'random_binary':
            draw = rng_local.random(n)
            choice = np.where(draw < 0.34, 0, np.where(draw < 0.67, 1, 2))
            choice = np.where(supports_reduced, choice, np.minimum(choice, 1))
            wanted_level = np.where(is_ac, (draw < 0.5).astype(int),
                                    choice * (budget > 1e-9))
        else:
            wanted_level = wanted.astype(int)
        # During an outage only the inverter circuit is alive: fans, lights and
        # the router. Everything else is unpowered.
        #
        # Demand for those essentials does NOT follow the normal schedule. A
        # power cut is exactly when someone switches the fan on, and riding the
        # cut is the whole reason the household bought an inverter. So while an
        # occupant is present, every essential load the household owns is wanted,
        # whether or not the clock says it is a preferred slot. Outside the
        # inverter circuit nothing is wanted, because nothing can run.
        if grid_absent:
            essential_wanted = on_inverter & (budget > 1e-9)
            if present:
                essential_wanted = essential_wanted | (on_inverter & (budget > 1e-9))
            else:
                essential_wanted = essential_wanted & (preferred[:, t] > 0)
            wanted = np.where(on_inverter, essential_wanted, False)
            user_wanted = np.where(on_inverter, essential_wanted, False)
            wanted_level = wanted.astype(int)

        # Unmodelled load is still load: it stops when the grid does.
        step_background_kw = 0.0 if grid_absent else float(background_kw)
        step_base_w = step_background_kw * 1000.0
        step_indoor = indoor
        step_battery = battery_kwh
        last = {}

        def physics(action):
            level = np.asarray(action, int)
            # Level 1 is full power, level 2 the reduced setting, level 0 off.
            action = np.where(level == 1, 1.0,
                              np.where(level == 2, reduced_fraction, 0.0))
            running = (level > 0).astype(float)
            ac_enabled = bool((running * is_ac).sum() > 0) if has_ac else False
            thermal_result = thermal_step(step_indoor, float(outdoor[t]), thermal,
                                          ac_enabled=ac_enabled, setpoint_c=setpoint)
            duty = thermal_result['compressor_duty_fraction']
            next_temperature = thermal_result['next_temperature_c']

            # A dimmed appliance still delivers its service, so the budget is
            # consumed at the full rate; only the power drawn is reduced.
            served = np.where(is_ac, 0.0, np.minimum(1.0, budget) * running)
            after = np.where(is_ac, budget, np.maximum(0.0, budget - served))
            powers = np.where(is_ac, power * action * duty,
                              power * np.minimum(1.0, budget) * action)

            flows = dispatch(demand_kw=float(powers.sum()) / 1000.0 + step_background_kw,
                             pv_kw=pv_kw,
                             battery_kwh=step_battery, household=household,
                             grid_absent=grid_absent)

            timers = advance_timers(devices, level.tolist())
            nxt = []
            for j, (d, z) in enumerate(zip(devices, timers)):
                if is_ac[j]:
                    nxt.append(replace(d, current_on=z['current_on'],
                                       elapsed_state_steps=z['elapsed_state_steps'],
                                       available=True, estimated_on_w=float(power[j]),
                                       estimated_reduced_w=float(power[j]
                                                                 * reduced_fraction[j]),
                                       must_run=False,
                                       noninterruptible_cycle_active=False))
                else:
                    next_absent = outage[min(t + 1, 95)]
                    nxt.append(replace(
                        d, current_on=z['current_on'],
                        elapsed_state_steps=z['elapsed_state_steps'],
                        available=bool(after[j] > 1e-9),
                        # Reduced power must track the rescaled ON power, or the
                        # next step sees a reduced setting above full power.
                        estimated_reduced_w=float(power[j] * min(1.0, after[j])
                                                  * reduced_fraction[j]),
                        # Minimum-on-time only binds when there is power to run on.
                        min_on_steps=(0 if next_absent
                                      else (4 if (cycle[j] and not is_ac[j]) else 0)),
                        estimated_on_w=float(power[j] * min(1.0, after[j])),
                        must_run=bool(protected[j] and t < 95
                                      and preferred[j, t + 1] > 0 and after[j] > 1e-9
                                      and not outage[min(t + 1, 95)]),
                        noninterruptible_cycle_active=bool(
                            cycle[j] and z['current_on']
                            and z['elapsed_state_steps'] < 4 and after[j] > 1e-9
                            and not outage[min(t + 1, 95)])))
            discomfort = (max(0.0, next_temperature - band_high)
                          + max(0.0, band_low - next_temperature)) if has_ac else 0.0
            unmet = float(after[~is_ac].sum() / 4) if t == 95 else 0.0
            next_shed = np.where(running.astype(bool), 0,
                                 np.minimum(96, steps_since_shed + 1))
            result = {
                'next_devices': nxt, 'appliance_power_w': powers.tolist(),
                'next_state': observe(t + 1, after, running.astype(bool), ledger.kwh,
                                      next_temperature, flows,
                                      False, next_shed, device_override_count,
                                      np.array([z['elapsed_state_steps'] for z in timers])),
                'discomfort_units': float(discomfort),
                'unmet_service_units': unmet,
                '_next_temperature_c': next_temperature, '_compressor_duty': duty,
                '_flows': flows, '_next_shed': next_shed}
            last['result'] = result
            return result

        # Sink-aware shield mask: if shedding saves nothing, the shed action is
        # removed rather than merely penalised, as the project spec requires.
        probe_flows = dispatch(
            demand_kw=float((power * np.where(is_ac, wanted.astype(float),
                                              np.minimum(1.0, budget) * wanted)).sum()) / 1000.0
                      + step_background_kw,
            pv_kw=pv_kw, battery_kwh=step_battery, household=household,
            grid_absent=grid_absent)
        marginal_rate = float(tariff.next_unit_rate(ledger.kwh))
        sink = shed_value_per_kw(flows=probe_flows, import_rate_inr_kwh=marginal_rate)
        worthless = bool(sink['shed_is_worthless'] and not grid_absent
                         and probe_flows['self_sufficient_fraction'] > 0)
        if worthless:
            # Express the sink mask through the REQUEST, not by mutating the
            # device. Setting must_run here made the device state at step t+1
            # disagree with what step t's dynamics produced, because the
            # previous step cannot know whether shedding will be worthless next
            # interval. Keeping the running load requested achieves the same
            # outcome and leaves device state continuous.
            wanted = np.where(current & ~is_ac, True, wanted)
            wanted_level = np.where(current & ~is_ac,
                                    np.maximum(wanted_level, 1), wanted_level)

        state = observe(t, budget, current, ledger.kwh, indoor,
                        opening_flows if t == 0 else previous_flows,
                        worthless, steps_since_shed, device_override_count)

        # What the policy alone would do, before any human request.
        wanted_level = np.asarray(wanted_level, int)
        policy_only = physics(wanted_level.tolist())
        from sharp_action_shield import apply_shield
        projected = apply_shield(devices, wanted_level.tolist(),
                                 max_import_w=import_ceiling_w, base_load_w=step_base_w,
                                 available_solar_w=non_grid_w)['executed_actions']

        decided, events, _ = decide_overrides(
            episode_id=episode, step_id=t, device_ids=device_ids,
            wanted=user_wanted.astype(int).tolist(), executed=projected,
            home_fraction=occupancy_fraction,
            degrees_outside_band=max(0.0, indoor - band_high),
            remaining_service_hours=(budget / 4).tolist(),
            is_air_conditioner=is_ac.tolist(), protected=protected.tolist())

        for event in events:
            arrival = t + int(event.latency_steps)
            if arrival <= 95:
                pending.setdefault(arrival, {})[event.device_id] = event

        # Only the overrides whose reaction time has elapsed act now.
        arrived = pending.pop(t, {})
        overrides = {device_id: 1 for device_id in arrived}
        for event in arrived.values():
            device_override_count[device_ids.index(event.device_id)] += 1
            override_total += 1

        record = transition_step(
            ledger=ledger, weights=weights, episode_id=episode, step_id=t,
            devices=devices, requested=wanted_level.tolist(),
            state=state, physics_step=physics, max_import_w=import_ceiling_w,
            base_load_w=step_base_w, available_solar_w=non_grid_w,
            grid_peak_severity=severity,
            policy_source=policy, data_release_version=RELEASE_VERSION,
            human_actions=overrides or None,
            intervention_seen=[present] * n,
            response_window_complete=[True] * n,
            terminated=t == 95)

        applied = last['result']
        executed_level = np.asarray(record['shield']['executed_actions'], int)
        executed = (executed_level > 0).astype(float)
        flows = applied['_flows']
        previous_flows = flows
        indoor = applied['_next_temperature_c']
        battery_kwh = flows['battery_kwh']
        steps_since_shed = applied['_next_shed']

        for event in arrived.values():
            index = device_ids.index(event.device_id)
            honoured = bool(record['shield']['human_override_honored'].get(
                event.device_id, False))
            pairs.append({
                'episode_id': episode, 'step_id': t, 'split': split,
                'household_id': tid, 'device_id': event.device_id,
                'decided_at_step': t - int(event.latency_steps),
                'appliance_type': ds.appliance_type.iloc[index],
                'proposed_action': 0,
                'proposed_action_at_arrival': int(projected[index]),
                'preferred_action': int(event.requested_on),
                'override_honoured': honoured,
                'override_probability': event.probability,
                'pressure_source': event.pressure_source,
                'degrees_outside_band': event.degrees_outside_band,
                'remaining_service_hours': event.remaining_service_hours,
            'policy_denied_user_request': True,
                'attention_available': event.attention_available,
                'latency_steps': event.latency_steps,
                'occupancy_adult_home_fraction': occupancy_fraction,
                'operating_mode': flows['operating_mode_name'],
                'marginal_tariff_inr_kwh': marginal_rate,
                'grid_peak_severity': severity,
                # Occupancy gating weights the pair: a user who was clearly home
                # gives stronger evidence than one who was marginally present.
                # Occupancy gating times pressure, discounted by how long the
                # user took to react: a fast correction is stronger evidence.
                'preference_weight': float(occupancy_fraction) * float(event.probability)
                                     / (1.0 + float(event.latency_steps)),
                'is_synthetic': True})

        budget = np.where(is_ac, budget,
                          np.maximum(0.0, budget - np.minimum(1.0, budget) * executed))
        current = executed_level > 0
        elapsed = np.array([x['elapsed_state_steps'] for x in record['next_device_state']], int)
        kwh_index = GLOBAL_FEATURES.index('month_to_date_kwh_div500')
        rate_index = GLOBAL_FEATURES.index('marginal_tariff_inr_kwh_div10')
        record['next_state']['features'][kwh_index] = float(ledger.kwh) / 500.0
        record['next_state']['features'][rate_index] = float(
            tariff.next_unit_rate(ledger.kwh)) / 10.0

        if records and records[-1]['next_state'] != record['state']:
            raise ValueError('State/next-state continuity failed')
        if records and records[-1]['next_device_state'] != record['device_state']:
            for a, b in zip(records[-1]['next_device_state'], record['device_state']):
                if a != b:
                    diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
                    raise ValueError(
                        f'Device-state continuity failed at step {t} '
                        f'device {a["device_id"]}: {diff}')
            raise ValueError('Device-state continuity failed')
        records.append(record)
        invalid_capacity += int(not record['constraint_feasible'])
        reward_sum += record['reward']['reward']

        rows.append({
            'episode_id': episode, 'step_id': t, 'split': split,
            'household_id': tid, 'date': date, 'season': season, 'policy': policy,
            'timestamp_ist': day.loc[t, 'timestamp_ist'].isoformat(),
            'state': record['state']['features'],
            'next_state': record['next_state']['features'],
            'action': record['shield']['executed_actions'],
            'requested_action': wanted_level.tolist(),
            'policy_action_before_human': projected,
            'device_present': [j < n for j in range(MAX_DEVICES)],
            'reward': record['reward']['reward'],
            'reward_cost_inr': record['reward']['raw_components']['cost_inr'],
            'reward_grid_peak_kwh': record['reward']['raw_components']['grid_peak_kwh'],
            'reward_discomfort': applied['discomfort_units'],
            'indoor_temperature_c': float(step_indoor),
            'next_indoor_temperature_c': float(indoor),
            'outdoor_temperature_c': float(outdoor[t]),
            'compressor_duty_fraction': float(applied['_compressor_duty']),
            'grid_import_kwh': record['grid_import_kwh'],
            'aggregate_power_kw': record['aggregate_power_w'] / 1000.0,
            'background_load_kw': step_background_kw,
            'controllable_power_kw': (record['aggregate_power_w'] / 1000.0
                                      - step_background_kw),
            'month_to_date_kwh': float(ledger.kwh),
            'marginal_tariff_inr_kwh': marginal_rate,
            'occupancy_adult_home_fraction': occupancy_fraction,
            'attention_available': present,
            'override_count': len(arrived),
            'operating_mode': flows['operating_mode_name'],
            'grid_absent': grid_absent,
            'self_sufficient_fraction': flows['self_sufficient_fraction'],
            'battery_kwh': flows['battery_kwh'],
            'pv_generation_kw': flows['pv_direct_kw'],
            'unserved_demand_kw': flows['unserved_kw'],
            'shed_is_worthless': worthless,
            'sink_reason': sink['sink_reason'],
            'terminated': record['terminated'], 'truncated': record['truncated'],
            'done': record['done'],
            'constraint_feasible': record['constraint_feasible']})

    billed = tariff.components(ledger.kwh, float(home.sanctioned_load_kw))['tariff_subtotal_inr']
    booked = ledger.booked_since_initialization
    opening_booked = tariff.components(opening_kwh, float(home.sanctioned_load_kw))['tariff_subtotal_inr']
    expected = billed - (opening_booked if day_of_month > 1 else 0)
    if booked != expected:
        raise ValueError(f'{episode}: billing reconciliation failed '
                         f'({booked} vs {expected})')

    summary = {'split': split, 'household_id': tid, 'date': date, 'season': season,
               'policy': policy, 'devices': n, 'has_air_conditioner': has_ac,
               'steps': 96, 'opening_kwh': opening_kwh,
               'closing_kwh': float(ledger.kwh),
               'day_import_kwh': float(ledger.kwh) - opening_kwh,
               'marginal_rate_end_inr_kwh': float(tariff.next_unit_rate(ledger.kwh)),
               'reward_sum': reward_sum, 'overrides': override_total,
               'outage_steps': int(sum(outage)),
               'has_inverter': household.has_inverter,
               'pv_capacity_kw': household.pv_capacity_kw,
               'unserved_kwh': float(sum(r['unserved_demand_kw'] for r in rows) * 0.25),
               'unserved_hours': float(budget[~is_ac].sum() / 4),
               'capacity_violation_steps': invalid_capacity}
    return rows, summary, pairs


def run(root, households, days_per_split, policies, seed, pv_scenario_kw,
        export_allowed):
    base = root / 'data/processed/simulator_devices_v1/unknown_quantity_one'
    paths = {
        'models': base / 'baseline_power_v1/device_power_models.parquet',
        'requests': base / 'service_plans_v1/weekly_service_requests.parquet',
        'preferences': base / 'service_plans_v1/preferred_service_slots.parquet',
        'households': root / 'data/processed/appliance_inputs_v1/ap_households_with_splits_v1.parquet',
        'billing': root / 'data/processed/appliance_inputs_v1/household_billing_position_v1.parquet',
        'background': root / 'data/processed/appliance_inputs_v1/household_background_load_v1.parquet',
        'occupancy': root / 'data/processed/location_scenarios_v1/adult_location_weekly_proxy.parquet',
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
    billing = pd.read_parquet(paths['billing']).set_index('template_id')
    background = (pd.read_parquet(paths['background']).set_index('template_id')
                  .background_kw.to_dict() if paths['background'].exists() else {})
    context = pd.read_parquet(paths['context']).sort_values('timestamp_ist')
    context['timestamp_ist'] = pd.to_datetime(context.timestamp_ist)
    tariff = Tariff.load(paths['tariff_config'])
    thermal = load_config(paths['thermal_config'])

    occupancy_frame = pd.read_parquet(paths['occupancy'])
    occupancy = {}
    for (tid, weekday), group in occupancy_frame.groupby(['template_id', 'weekday_number']):
        series = group.sort_values('step_of_day').adult_reported_home_fraction_proxy
        if len(series) == 96:
            occupancy[(tid, int(weekday))] = series.to_numpy(float)
    check(occupancy, 'No complete occupancy days')

    check(int(hh.groupby('template_id').split.nunique().max()) == 1,
          'A household appears in more than one split')

    out = root / 'data/processed/sharp_rl_transitions_v2'
    out.mkdir(parents=True, exist_ok=True)
    day_by_split = {s: pick_days(context, s, days_per_split, seed)
                    for s in ['train', 'validation', 'test']}
    print('Context days per split:', {s: len(v) for s, v in day_by_split.items()},
          flush=True)

    weights = RewardWeights(cost_per_inr=1, grid_peak_per_kwh=1,
                            discomfort_per_unit=0.5, switching_per_event=0.01,
                            unmet_service_per_unit=10)
    frames, summaries, all_pairs, skipped = [], [], [], 0
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
            if not 0 < len(ds) <= MAX_DEVICES or home.template_id not in billing.index:
                skipped += 1
                continue
            power_row = {'template_id': home.template_id,
                         'grid_supply_hours_daily': home.grid_supply_hours_daily,
                         'evening_supply_hours': home.evening_supply_hours,
                         'inverter_battery_available': home.inverter_battery_available,
                         'solar_home_system_capacity_w': home.solar_home_system_capacity_w}
            billing_row = billing.loc[home.template_id]
            for date in dates:
                day = context.loc[context.timestamp_ist.dt.strftime('%Y-%m-%d').eq(date)
                                  & context.split.eq(split)].reset_index(drop=True)
                if len(day) != 96:
                    continue
                for policy in policies:
                    result = run_episode(
                        ds=ds, home=home, power_row=power_row, billing_row=billing_row,
                        occupancy=occupancy, day=day, date=date, split=split,
                        policy=policy, tariff=tariff, weights=weights,
                        thermal=thermal, requests=requests, preferences=preferences,
                        pv_scenario_kw=pv_scenario_kw, export_allowed=export_allowed,
                        background_kw=float(background.get(home.template_id, 0.0)))
                    if result is None:
                        skipped += 1
                        continue
                    rows, summary, pairs = result
                    frames.extend(rows)
                    summaries.append(summary)
                    all_pairs.extend(pairs)
            if count % 25 == 0:
                print(f'  {split}: {count}/{len(pool)} households, '
                      f'{len(frames):,} transitions, {len(all_pairs):,} pairs',
                      flush=True)

    check(frames, 'No transitions were generated')
    transitions = pd.DataFrame(frames)
    episodes = pd.DataFrame(summaries)
    pairs = pd.DataFrame(all_pairs)

    check(int(transitions.groupby('household_id').split.nunique().max()) == 1,
          'Household leakage across splits')
    check(int(transitions.groupby('date').split.nunique().max()) == 1,
          'Context date leakage across splits')
    width = len(GLOBAL_FEATURES) + MAX_DEVICES * len(DEVICE_FEATURES)
    check(transitions.state.map(len).eq(width).all(), 'Ragged state vectors')
    check(transitions.next_state.map(len).eq(width).all(), 'Ragged next_state')
    lo = float(transitions.indoor_temperature_c.min())
    hi = float(transitions.next_indoor_temperature_c.max())
    check(5.0 < lo and hi < 55.0, f'Indoor temperature out of range: {lo} to {hi}')

    transitions.to_parquet(out / 'rl_transitions.parquet', index=False,
                           compression='zstd')
    episodes.to_csv(out / 'episode_summary.csv', index=False)
    if len(pairs):
        pairs.to_parquet(out / 'override_preference_pairs.parquet', index=False,
                         compression='zstd')

    schema = {
        'global_features': GLOBAL_FEATURES, 'device_features': DEVICE_FEATURES,
        'maximum_device_slots': MAX_DEVICES, 'feature_count': width,
        'device_order': 'device_id ascending within household',
        'padding': 'zero features; device_present column carries the mask',
        'action_space': 'per-device binary enable; AC enable is a thermostat call',
        'thermal_config': thermal['config_id'], 'thermal_scenario': thermal['scenario'],
    }
    (out / 'feature_schema.json').write_text(json.dumps(schema, indent=2), encoding='utf-8')

    report = {
        'status': 'SHARP_RL_TRANSITIONS_V2_GENERATED',
        'release_version': RELEASE_VERSION,
        'transitions': int(len(transitions)), 'episodes': int(len(episodes)),
        'households': int(transitions.household_id.nunique()),
        'context_days': int(transitions.date.nunique()),
        'policies': sorted(transitions.policy.unique().tolist()),
        'seasons': sorted(transitions.season.unique().tolist()),
        'feature_count': width,
        'transitions_by_split': transitions.split.value_counts().to_dict(),
        'households_by_split': transitions.groupby('split').household_id.nunique().to_dict(),
        'override_events': int(len(pairs)),
        'override_pairs_by_split': (pairs.split.value_counts().to_dict()
                                    if len(pairs) else {}),
        'override_honoured': int(pairs.override_honoured.sum()) if len(pairs) else 0,
        'episodes_with_outage': int((episodes.outage_steps > 0).sum()),
        'households_with_inverter': int(episodes[episodes.has_inverter].household_id.nunique()),
        'operating_mode_steps': transitions.operating_mode.value_counts().to_dict(),
        'marginal_rate_distribution': transitions.marginal_tariff_inr_kwh.value_counts().sort_index().to_dict(),
        'mean_occupancy': float(transitions.occupancy_adult_home_fraction.mean()),
        'attention_available_fraction': float(transitions.attention_available.mean()),
        'billing_reconciliation': 'PASS', 'state_continuity': 'PASS',
        'household_split_leakage': 'NONE', 'context_date_split_leakage': 'NONE',
        'capacity_violation_steps': int((~transitions.constraint_feasible).sum()),
        'unserved_kwh_total': float(episodes.unserved_kwh.sum()),
        'indoor_temperature_c': {'min': lo,
                                 'mean': float(transitions.indoor_temperature_c.mean()),
                                 'max': hi},
        'mean_reward': float(transitions.reward.mean()),
        'skipped_episodes': int(skipped),
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
            'Infeasible transitions are preserved and flagged, never silently dropped.',
        ],
        'full_simulator_ready': True, 'master_release_ready': True,
    }
    (out / 'transition_validation.json').write_text(json.dumps(report, indent=2),
                                                    encoding='utf-8')
    print('\nSHARP RL TRANSITIONS V2 GENERATED')
    for key in ['transitions', 'episodes', 'households', 'feature_count',
                'transitions_by_split', 'override_events', 'override_honoured',
                'episodes_with_outage', 'operating_mode_steps',
                'marginal_rate_distribution', 'attention_available_fraction',
                'capacity_violation_steps', 'mean_reward', 'skipped_episodes']:
        print(f'  {key}: {report[key]}')
    print('Output:', out)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--households', type=int, default=0)
    p.add_argument('--days-per-split', type=int, default=2)
    p.add_argument('--policies', nargs='+', default=['serve_preferred', 'random_binary'])
    p.add_argument('--seed', type=int, default=20260914)
    p.add_argument('--pv-scenario-kw', type=float, default=0.0,
                   help='rooftop PV overlay in kW; 0 keeps IRES-reported solar only')
    p.add_argument('--export-allowed', action='store_true')
    a = p.parse_args()
    run(a.root.resolve(), a.households, a.days_per_split, a.policies, a.seed,
        a.pv_scenario_kw, a.export_allowed)
