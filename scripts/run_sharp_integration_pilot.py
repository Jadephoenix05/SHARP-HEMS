"""Actual-input integration pilot, not a validated thermal/home simulator.
Runs one deterministic household per split for a finite 24-hour service task.
Source-derived and assumed inputs retain their upstream provenance.
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

OBS=['obs_T2M','obs_RH2M','obs_ALLSKY_SFC_SW_DWN','obs_WS10M',
     'obs_grid_percentile','obs_grid_peak_severity']
MAX_DEVICES=28

def run(root):
    base=root/'data/processed/simulator_devices_v1/unknown_quantity_one'
    paths={
      'models':base/'baseline_power_v1/device_power_models.parquet',
      'requests':base/'service_plans_v1/weekly_service_requests.parquet',
      'preferences':base/'service_plans_v1/preferred_service_slots.parquet',
      'households':root/'data/processed/appliance_inputs_v1/ap_households_with_splits_v1.parquet',
      'context':root/'data/processed/simulator_context_v1/regional_grid_guntur_weather_15min_v1.parquet'}
    models=pd.read_parquet(paths['models']);requests=pd.read_parquet(paths['requests'])
    preferences=pd.read_parquet(paths['preferences']);hh=pd.read_parquet(paths['households'])
    context=pd.read_parquet(paths['context']).sort_values('timestamp_ist')
    context['timestamp_ist']=pd.to_datetime(context.timestamp_ist)
    tariff=Tariff.load(root/'configs/tariffs/apcpdcl_2025_26_verified_components.json')
    out=root/'data/processed/rl_integration_pilot_v1';out.mkdir(parents=True,exist_ok=True)
    output=[];episode_reports=[]
    # Explicit test weights. They have NOT been learned or selected for release.
    weights=RewardWeights(cost_per_inr=1,grid_peak_per_kwh=1,discomfort_per_unit=0,
                          switching_per_event=.01,unmet_service_per_unit=10)
    for split,year in [('train',2022),('validation',2023),('test',2024)]:
        candidates=hh.loc[hh.split.eq(split)].sort_values('template_id')
        # No invented connection load: use a positive reported value for pilot.
        candidates=candidates.loc[pd.to_numeric(candidates.sanctioned_load_kw,errors='coerce').gt(0)]
        if candidates.empty:raise ValueError(f'No usable reported connection load in {split}')
        home=candidates.iloc[0];tid=home.template_id
        date=f'{year}-03-01'
        day=context.loc[context.timestamp_ist.dt.strftime('%Y-%m-%d').eq(date)&context.split.eq(split)].reset_index(drop=True)
        if len(day)!=96:raise ValueError(f'Expected complete 96-step context: {date}')
        ds=models.loc[models.template_id.eq(tid)].sort_values('device_id').reset_index(drop=True)
        n=len(ds)
        if not 0<n<=MAX_DEVICES:raise ValueError('Device padding bound exceeded')
        weekday=int(day.timestamp_ist.iloc[0].weekday());season='mar_jun'
        req=requests.loc[requests.template_id.eq(tid)&requests.season.eq(season)&requests.weekday_number.eq(weekday)].set_index('device_id')
        pref=preferences.loc[preferences.template_id.eq(tid)&preferences.season.eq(season)&preferences.weekday_number.eq(weekday)].set_index('device_id')
        remaining=np.array([float(req.loc[d,'requested_hours'])*4 for d in ds.device_id])
        preferred=np.stack([np.asarray(pref.loc[d,'preferred_service_fraction'],float) for d in ds.device_id])
        power=ds.operating_power_proxy_w.to_numpy(float)
        protected=ds.service_role.eq('protected_service').to_numpy(bool)
        cycle=ds.dynamics_family.eq('cycle').to_numpy(bool)
        limit=float(home.sanctioned_load_kw)*1000
        if not math.isfinite(limit):raise ValueError('Nonfinite connection limit')
        for policy in ['serve_preferred','peak_aware','random_binary']:
            budget=remaining.copy();current=np.zeros(n,bool);elapsed=np.full(n,4,int)
            rng=np.random.default_rng(int(hashlib.sha256((tid+policy).encode()).hexdigest()[:8],16))
            ledger=BillingLedger(tariff,tid,f'pilot_calendar_month_{year}-03',float(home.sanctioned_load_kw),
                opening_kwh=0,opening_charges_already_booked=False)
            episode=f'{split}:{tid}:{date}:{policy}'
            invalid_capacity=0;reward_sum=0.;records=[]
            def observe(t,b,c,kwh,e=None):
                # No future forcing is placed in policy observations.
                # Terminal sentinel context is zero; terminated=True prevents bootstrapping.
                globals_=day.loc[t,OBS].to_numpy(float).tolist() if t<96 else [0.]*len(OBS)
                globals_ += [t/96,float(kwh),limit/1000]
                device=[]
                e=elapsed if e is None else e
                for j in range(MAX_DEVICES):
                    device.extend([float(b[j]/4),float(power[j]/1000),float(c[j]),float(protected[j]),float(cycle[j]),float(e[j]/96),float(preferred[j,t]) if t<96 else 0.]
                    if j<n else [0.]*7)
                vec=globals_+device
                if not np.isfinite(vec).all():raise ValueError('Nonfinite observation')
                return {'features':vec,'device_present':[j<n for j in range(MAX_DEVICES)]}
            for t in range(96):
                # Budget service model only: no physical room/fridge/water temperatures.
                devices=[]
                for j,r in ds.iterrows():
                    available=bool(budget[j]>1e-9)
                    active_cycle=bool(cycle[j] and current[j] and elapsed[j]<4 and available)
                    devices.append(Device(str(r.device_id),bool(current[j]),available,
                        float(power[j]*min(1.,budget[j])),must_run=bool(protected[j] and preferred[j,t]>0 and available),
                        noninterruptible_cycle_active=active_cycle,elapsed_state_steps=int(elapsed[j]),
                        min_on_steps=4 if cycle[j] else 0,min_off_steps=0,shed_priority=10 if not protected[j] else 0))
                wanted=(preferred[:,t]>0)&(budget>1e-9)
                if policy=='peak_aware' and float(day.loc[t,'obs_grid_peak_severity'])>.5:
                    wanted &= protected
                elif policy=='random_binary':wanted=(rng.random(n)<.5)&(budget>1e-9)
                def physics(action):
                    delivered=np.minimum(1.,budget)*np.asarray(action,float)
                    after=np.maximum(0.,budget-delivered)
                    timers=advance_timers(devices,action)
                    nxt=[replace(d,current_on=z['current_on'],elapsed_state_steps=z['elapsed_state_steps'],
                        available=bool(after[j]>1e-9),estimated_on_w=float(power[j]*min(1.,after[j])),
                        must_run=bool(protected[j] and t<95 and preferred[j,t+1]>0 and after[j]>1e-9),
                        noninterruptible_cycle_active=bool(cycle[j] and z['current_on'] and z['elapsed_state_steps']<4 and after[j]>1e-9))
                        for j,(d,z) in enumerate(zip(devices,timers))]
                    # next_state billing feature is patched after atomic billing commits.
                    return {'next_devices':nxt,'appliance_power_w':(power*delivered).tolist(),
                        'next_state':observe(t+1,after,np.asarray(action,bool),ledger.kwh,np.array([z['elapsed_state_steps'] for z in timers])),
                        'discomfort_units':0.,
                        'unmet_service_units':float(after.sum()/4) if t==95 else 0.}
                record=transition_step(ledger=ledger,weights=weights,episode_id=episode,step_id=t,
                    devices=devices,requested=wanted.astype(int).tolist(),state=observe(t,budget,current,ledger.kwh),
                    physics_step=physics,max_import_w=limit,base_load_w=0.,available_solar_w=0.,
                    grid_peak_severity=float(day.loc[t,'forcing_grid_peak_severity']),
                    policy_source=policy,data_release_version='PILOT_NOT_RELEASE',terminated=t==95)
                budget=np.maximum(0.,budget-np.minimum(1.,budget)*np.asarray(record['shield']['executed_actions']))
                current=np.array(record['shield']['executed_actions'],bool)
                elapsed=np.array([x['elapsed_state_steps'] for x in record['next_device_state']],int)
                record['next_state']['features'][7]=float(ledger.kwh)
                record.update({'timestamp_ist':day.loc[t,'timestamp_ist'].isoformat(),'split':split,
                    'appliance_types':ds.appliance_type.tolist(),'dynamics_scope':'FINITE_DAILY_SERVICE_BUDGET_NOT_THERMAL',
                    'attention_model':'ABSENT_ALL_NONOVERRIDE_EVIDENCE_CENSORED'})
                if records and records[-1]['next_state']!=record['state']:
                    raise ValueError('State/next-state continuity failed')
                if records and records[-1]['next_device_state']!=record['device_state']:
                    raise ValueError('Device-state continuity failed')
                records.append(record)
                invalid_capacity+=int(not record['constraint_feasible']);reward_sum+=record['reward']['reward']
            # Store nested complete records as JSON; flat features are included for later loaders.
            for record in records:
                output.append({'episode_id':episode,'step_id':record['step_id'],'split':split,
                    'state':record['state']['features'],'next_state':record['next_state']['features'],
                    'reward':record['reward']['reward'],'terminated':record['terminated'],
                    'constraint_feasible':record['constraint_feasible'],
                    'transition_json':json.dumps(record,allow_nan=False,separators=(',',':'))})
            billed=tariff.components(ledger.kwh,float(home.sanctioned_load_kw))['tariff_subtotal_inr']
            if billed!=ledger.booked_since_initialization:raise ValueError('Pilot billing failed reconciliation')
            episode_reports.append({'split':split,'template_id':tid,'policy':policy,'devices':n,
                'steps':96,'import_kwh':float(ledger.kwh),'tariff_components_inr':float(billed),
                'reward_sum':reward_sum,'unserved_hours':float(budget.sum()/4),
                'capacity_violation_steps':invalid_capacity})
            print(f'{split} | {policy} | 96 steps | unserved {budget.sum()/4:.2f} hours | capacity violations {invalid_capacity}',flush=True)
    transitions=pd.DataFrame(output)
    if len(transitions)!=864:raise ValueError('Unexpected pilot transition count')
    transitions.to_parquet(out/'pilot_transitions.parquet',index=False,compression='zstd')
    schema={'global_features':OBS+['fraction_of_day','billing_period_kwh','connection_limit_kw'],
        'device_features':['remaining_service_hours','power_proxy_kw','current_on','protected_service','cycle_type','elapsed_state_steps_divided_by_96','preferred_service_fraction'],
        'maximum_device_slots':MAX_DEVICES,'device_order':'device_id ascending within household',
        'padding':'zero features; device_present mask in transition_json state',
        'actions_and_masks':'unpadded vectors in transition_json shield; loader must pad and mask consistently',
        'task':'finite daily service-budget integration test, not a final observation contract'}
    (out/'pilot_feature_schema.json').write_text(json.dumps(schema,indent=2),encoding='utf-8')
    pd.DataFrame(episode_reports).to_csv(out/'episode_summary.csv',index=False)
    report={'status':'PILOT_TRANSITIONS_GENERATED','households':3,'episodes':9,'transitions':len(transitions),
        'feature_count':len(output[0]['state']),'billing_reconciliation':'PASS','state_continuity':'PASS',
        'capacity_violation_steps':int((~transitions.constraint_feasible).sum()),
        'assumptions':['Finite 24-hour service-budget task, not validated appliance thermal dynamics.',
          'One March day per split; this does not establish seasonal evaluation.',
          'Calendar-month billing starts with zero consumption; FY2025-26 tariff is a scenario applied to other-year context.',
          'No solar, battery, outage model, human-attention model or learned preferences.',
          'No hidden background load; only modelled devices contribute demand.',
          'Cycles use a four-step minimum-on assumption; no complete-cycle source validation.',
          'Thermal comfort is not modelled; discomfort reward is zero, not evidence of comfort.',
          'Daily remaining service is penalised at the finite task termination.',
          'Infeasible transitions are preserved and flagged, not passed as safe training examples.',
          'REFIT, eMARC and RESIDE are not integrated into this pilot dynamics model.'],
        'reward_weights':{'cost':1,'peak':1,'discomfort':0,'switching':.01,'unmet_service_hours':10},
        'inputs_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        'full_simulator_ready':False,'master_release_ready':False}
    (out/'pilot_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nSHARP INTEGRATION PILOT');print(json.dumps(report,indent=2));print('\nOutput:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();run(a.root.resolve())
