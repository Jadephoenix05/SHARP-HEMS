"""Synthetic weekly service requests, with survey durations where available.
These are inputs to a scheduler, not executed demand or measured load traces.
Three four-month seasons follow the IRES AC-use question periods.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd

# Explicit development assumptions (hours per day), not empirical averages.
HOURS={'air_conditioner':4,'air_cooler':6,'ceiling_fan':8,'table_fan':4,
 'cfl_bulb':4,'cfl_tube':4,'incandescent_bulb':4,'led_bulb':4,'led_tube':4,
 'desktop':2,'electric_iron':.25,'electric_kettle':.1,'electric_rice_cooker':1,
 'geyser':.5,'laptop_tablet':2,'mixer_grinder':.1,'modem_router':24,
 'refrigerator':8,'television':3,'washing_machine':1,'water_pump':.5,'water_purifier':.5}
SEASONS=['mar_jun','jul_oct','nov_feb']
TASKS={'electric_iron','electric_kettle','electric_rice_cooker','geyser',
 'mixer_grinder','washing_machine','water_pump','water_purifier'}
LIGHTS={'cfl_bulb','cfl_tube','incandescent_bulb','led_bulb','led_tube'}

def hashint(s):return int(hashlib.sha256(s.encode()).hexdigest(),16)

def build(root):
    base=root/'data/processed/simulator_devices_v1/unknown_quantity_one'
    mp=base/'baseline_power_v1/device_power_models.parquet'; up=base/'survey_usage_constraints.parquet'
    lp=root/'data/processed/location_scenarios_v1/adult_location_weekly_proxy.parquet'
    models=pd.read_parquet(mp).sort_values(['template_id','appliance_type','unit_index'])
    usage=pd.read_parquet(up);locations=pd.read_parquet(lp)
    lookup={}
    for r in usage.to_dict('records'):
        if r['availability']=='SOURCE_REPORTED':lookup[(r['target_id'],r['metric'],r['season'])]=float(r['source_value'])
    proxy={(t,int(w)):g.sort_values('step_of_day').adult_reported_home_fraction_proxy.to_numpy(float)
        for (t,w),g in locations.groupby(['template_id','weekday_number'])}
    # Measured Indian activity timing from the TUS 2024 diaries, where an
    # activity plainly requires the appliance. This replaces hardcoded clock
    # heuristics for the appliances it covers: comparing against eMARC mainline
    # meters showed the heuristics produced no morning peak at all, which is the
    # most characteristic feature of Indian household load.
    tus_path=root/'data/processed/tus_appliance_clock_v1/appliance_clock_profiles.csv'
    tus={}
    if tus_path.exists():
        frame=pd.read_csv(tus_path)
        for column in frame.columns:
            if column in ('step_of_day','clock'):continue
            values=frame[column].to_numpy(float)
            if len(values)==96 and values.sum()>0:
                tus[column]=values/values.max()
        print('TUS clock profiles applied to:',sorted(tus),flush=True)
    else:
        print('No TUS clock profiles found; using heuristics only',flush=True)
    schedules=[];requests=[]
    clock=np.arange(96)/4
    def allocate(hours,score):
        # Fractional final slot preserves sub-15-minute reported durations.
        target=hours*4;order=np.argsort(-score,kind='stable');v=np.zeros(96)
        for i in order:
            amount=min(1.,target);v[i]=amount;target-=amount
            if target<=1e-10:break
        if not np.isclose(v.sum()/4,hours):raise ValueError('Duration allocation mismatch')
        return v
    for r in models.to_dict('records'):
        tid=r['template_id'];app=r['appliance_type'];did=r['device_id']
        if app not in HOURS:raise ValueError(f'Missing duration assumption: {app}')
        siblings=models.loc[models.template_id.eq(tid)&models.appliance_type.eq(app)]
        n=len(siblings)
        for season in SEASONS:
            hours=HOURS[app];basis='EXPLICIT_DEVELOPMENT_DURATION_ASSUMPTION'
            seasonal_months=lookup.get((did,'reported_months_per_year','unspecified'))
            key=(did,'reported_hours_daily','unspecified')
            if key in lookup:
                hours=lookup[key];basis='SURVEY_REPORTED_DEVICE_DAILY_HOURS'
            elif app=='air_conditioner' and r['unit_index']==1:
                key=(tid+':primary_ac','daily_use_hours',season)
                if key in lookup:
                    hours=lookup[key];basis='SURVEY_MOST_USED_AC_HOURS_ROLE_ASSIGNED_TO_UNIT_1_IN_SCENARIO'
            elif app=='water_pump':
                key=(tid+':water_pump','daily_use_minutes','unspecified')
                if key in lookup:
                    hours=lookup[key]/60/n;basis='SURVEY_GROUP_MINUTES_DIVIDED_EQUALLY_ACROSS_SCENARIO_UNITS'
            if not math.isfinite(hours) or not 0<=hours<=24:raise ValueError('Invalid duration')
            # Washing uses are weekly; distribute complete one-hour jobs.
            # One hour per cycle is an explicit assumption, not a survey value.
            weekly=lookup.get((tid+':washing_machine','weekly_uses','unspecified'),3.) if app=='washing_machine' else None
            jobs_by_day=np.zeros(7,dtype=int)
            if weekly is not None:
                if not math.isfinite(weekly) or weekly<0:raise ValueError('Invalid weekly washing frequency')
                # Deterministic stochastic rounding preserves expectation, not an exact fractional weekly count.
                target=weekly/n;count=int(target)
                count+=((hashint(did+season+':round')%1000000)/1000000 < target-count)
                order=sorted(range(7),key=lambda k:hashint(did+season+str(k)))
                for j in range(count):jobs_by_day[order[j%7]]+=1
                if jobs_by_day.max()>24:raise ValueError('Washing demand exceeds daily capacity')
            for day in range(7):
                home=proxy[(tid,day)]
                if len(home)!=96 or not np.isfinite(home).all():raise ValueError('Invalid location proxy')
                h=float(jobs_by_day[day]) if weekly is not None else hours
                score=home*.5+np.sin((clock-6)/24*2*np.pi)*.05
                profile_key = 'lighting' if (app in LIGHTS and 'lighting' in tus) else app
                if profile_key in tus:
                    # Measured activity timing outweighs the presence prior.
                    score=score+tus[profile_key]*2
                    clock_basis=('TUS_2024_SLEEP_AND_DARKNESS' if profile_key=='lighting'
                                 else 'TUS_2024_MEASURED_ACTIVITY_TIMING')
                elif app in LIGHTS:
                    score=score+((clock>=18)|(clock<6))*2
                    clock_basis='SYNTHETIC_HEURISTIC_DARKNESS_AND_PRESENCE'
                elif app in {'television','desktop','laptop_tablet'}:
                    score=score+((clock>=18)&(clock<23))*2
                    clock_basis='SYNTHETIC_HEURISTIC_EVENING_USE'
                elif app=='air_conditioner':
                    score=score+((clock>=21)|(clock<6))*1.5
                    clock_basis='SYNTHETIC_HEURISTIC_NIGHT_COOLING'
                elif app in TASKS:
                    score=score+((clock>=6)&(clock<10))*2
                    clock_basis='SYNTHETIC_HEURISTIC_MORNING_TASK'
                else:
                    clock_basis='SYNTHETIC_HEURISTIC_PRESENCE_ONLY'
                # Device-specific tiny deterministic jitter breaks ties.
                rng=np.random.default_rng(hashint(did+season+str(day))%(2**32))
                score=score+rng.uniform(0,.001,96)
                fractions=allocate(h,score)
                if weekly is not None:
                    this_basis='SURVEY_WEEKLY_FREQUENCY_AND_ASSUMED_ONE_HOUR_CYCLES' if (tid+':washing_machine','weekly_uses','unspecified') in lookup else 'ASSUMED_THREE_WEEKLY_ONE_HOUR_CYCLES'
                else:this_basis=basis
                request={'device_id':did,'template_id':tid,'split':r['split'],'appliance_type':app,
                    'season':season,'weekday_number':day,'requested_hours':h,
                    'duration_basis':this_basis,'source_active_months_per_year':seasonal_months,
                    'active_calendar_months_assigned':False,
                    'task_count':int(jobs_by_day[day]) if weekly is not None else None,
                    'clock_preference_basis':clock_basis,
                    'preferred_slots_are_executed_load':False,'is_synthetic':True}
                requests.append(request)
                schedules.append({'device_id':did,'template_id':tid,'split':r['split'],
                    'season':season,'weekday_number':day,'preferred_service_fraction':fractions.tolist()})
    req=pd.DataFrame(requests);pref=pd.DataFrame(schedules)
    keys=['device_id','season','weekday_number']
    if req.duplicated(keys).any():raise ValueError('Duplicate service requests')
    if len(req)!=len(models)*3*7:raise ValueError('Incomplete seasonal service plan')
    out=base/'service_plans_v1';out.mkdir(parents=True,exist_ok=True)
    req.to_parquet(out/'weekly_service_requests.parquet',index=False,compression='zstd')
    pref.to_parquet(out/'preferred_service_slots.parquet',index=False,compression='zstd')
    assumption={'daily_hours_fallback':HOURS,'seasons':SEASONS,
      'important_limits':['These are requested service and preferred time slots, not executed power.',
        'Fractional slots describe within-interval service duration, not separate binary RL actions.',
        'Preferred slots may be noncontiguous; noninterruptible cycles must be enforced by appliance dynamics.',
        'Fans retain reported months-per-year; active months have not been inferred or applied.',
        'Refrigerator eight-hour compressor-time budget is an unvalidated engineering assumption.',
        'Washing machine cycle length is assumed; complete cycles are not measured by this planner.',
        'A primary-AC role is assigned to scenario unit 1, not identified as a physical survey device.',
        'Other AC units use fallback duration rather than copying the most-used AC hours.',
        'One adult being away does not imply an empty household; service remains possible while proxy is away.',
        'No thermal comfort, attention, human preference or appliance-cycle dynamics are inferred here.']}
    (out/'service_assumptions.json').write_text(json.dumps(assumption,indent=2),encoding='utf-8')
    report={'status':'SEASONAL_WEEKLY_SERVICE_PLANS_BUILT','devices':len(models),
       'service_request_rows':len(req),'seasons':3,'preferred_slots_per_request':96,
       'duration_basis_counts':{k:int(v) for k,v in req.duration_basis.value_counts().items()},
       'input_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [mp,up,lp]},
       'full_simulator_ready':False,'master_release_ready':False}
    (out/'service_plan_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('\nOutput:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();build(a.root.resolve())
