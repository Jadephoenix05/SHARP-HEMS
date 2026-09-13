"""Bind survey usage constraints to simulator devices without inventing clocks.
Reported daily duration does not establish when an appliance operates.
Group-level reports must not be multiplied by the number of devices.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import pandas as pd


def build(root):
    base=root/'data/processed/appliance_inputs_v1'
    devpath=root/'data/processed/simulator_devices_v1/unknown_quantity_one/device_instances.parquet'
    hhpath=base/'ap_households_with_splits_v1.parquet'
    fanpath=root/'data/processed/household_templates/ap_fan_usage_units_v1.parquet'
    devices=pd.read_parquet(devpath); hh=pd.read_parquet(hhpath); fans=pd.read_parquet(fanpath)
    if len(devices)!=4124 or len(hh)!=498:
        raise ValueError('Unexpected device/household population')
    if not hh.template_id.is_unique or not devices.device_id.is_unique:
        raise ValueError('Duplicate household or device keys')
    if fans.duplicated(['template_id','appliance_type','survey_fan_number']).any():
        raise ValueError('Duplicate fan-use keys')
    fan_devices=devices.loc[devices.appliance_type.isin(['ceiling_fan','table_fan'])]
    matched=fans.merge(fan_devices[['template_id','appliance_type','unit_index','device_id','split']],
        left_on=['template_id','appliance_type','survey_fan_number'],
        right_on=['template_id','appliance_type','unit_index'],how='outer',validate='one_to_one',indicator=True)
    if not matched['_merge'].eq('both').all():
        raise ValueError('Fan usage and individual-device inventory do not match')
    rows=[]
    def add(template,app,scope,target,metric,value,unit,source_field,season='unspecified'):
        missing=pd.isna(value)
        if not missing:
            value=float(value)
            if not math.isfinite(value) or value<0:
                raise ValueError(f'Invalid usage: {template} {source_field}')
            upper={'hours_per_day':24,'months_per_year':12,'minutes_per_day':1440}.get(unit)
            if upper is not None and value>upper:
                raise ValueError(f'Out-of-range usage: {template} {source_field}: {value}')
        rows.append({'template_id':template,'appliance_type':app,'target_scope':scope,
            'target_id':target,'metric':metric,'source_value':None if missing else value,
            'unit':unit,'season':season,'source_field':source_field,
            'availability':'SOURCE_MISSING' if missing else 'SOURCE_REPORTED',
            'clock_schedule_established':False,'usage_is_synthetic':False})
    for r in matched.to_dict('records'):
        for field,unit in [('reported_hours_daily','hours_per_day'),('reported_months_per_year','months_per_year')]:
            add(r['template_id'],r['appliance_type'],'DEVICE',r['device_id'],field,r[field],unit,'fan_usage:'+field)
    groups=set(zip(devices.template_id,devices.appliance_type))
    for r in hh.to_dict('records'):
        tid=r['template_id']
        if (tid,'air_conditioner') in groups:
            # Survey refers to the most-used AC, not every AC in the house.
            # Keep that role unresolved rather than assigning arbitrary unit 1.
            for season,col in [('mar_jun','ac_hours_daily_mar_jun'),('jul_oct','ac_hours_daily_jul_oct'),('nov_feb','ac_hours_daily_nov_feb')]:
                add(tid,'air_conditioner','MOST_USED_DEVICE_ROLE_UNASSIGNED',tid+':primary_ac',
                    'daily_use_hours',r[col],'hours_per_day',col,season)
            add(tid,'air_conditioner','MOST_USED_DEVICE_ROLE_UNASSIGNED',tid+':primary_ac',
                'cooling_capacity',r['primary_ac_capacity_ton'],'cooling_ton','primary_ac_capacity_ton')
        if (tid,'washing_machine') in groups:
            add(tid,'washing_machine','HOUSEHOLD_APPLIANCE_GROUP',tid+':washing_machine',
                'weekly_uses',r['washing_machine_uses_weekly'],'uses_per_week','washing_machine_uses_weekly')
        if (tid,'water_pump') in groups:
            add(tid,'water_pump','HOUSEHOLD_APPLIANCE_GROUP',tid+':water_pump',
                'daily_use_minutes',r['water_pump_daily_minutes'],'minutes_per_day','water_pump_daily_minutes')
            add(tid,'water_pump','HOUSEHOLD_APPLIANCE_GROUP',tid+':water_pump',
                'reported_capacity',r['water_pump_capacity_hp'],'source_hp','water_pump_capacity_hp')
    result=pd.DataFrame(rows)
    keys=['template_id','appliance_type','target_scope','target_id','metric','season']
    if result.duplicated(keys).any():raise ValueError('Duplicate usage constraints')
    result=result.merge(hh[['template_id','split']],on='template_id',validate='many_to_one')
    out=root/'data/processed/simulator_devices_v1/unknown_quantity_one'
    result.to_parquet(out/'survey_usage_constraints.parquet',index=False)
    report={'status':'SURVEY_USAGE_CONSTRAINTS_BOUND','fan_devices_matched':len(matched),
      'usage_constraint_rows':len(result),'missing_source_values':int(result.source_value.isna().sum()),
      'rows_by_appliance':{k:int(v) for k,v in result.appliance_type.value_counts().items()},
      'notes':['Fan number is matched to the corresponding within-household device index.',
               'AC usage describes the most-used AC; physical unit assignment remains unresolved.',
               'Pump and washing-machine reports are retained at household-appliance-group scope.',
               'Cooling tons and pump horsepower have not been converted to electrical watts.',
               'Monthly fan-use duration does not identify the active months.',
               'Clock schedules and operating power still require explicit simulator models.'],
      'inputs_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [devpath,hhpath,fanpath]},
      'full_simulator_ready':False,'master_release_ready':False}
    p=out/'survey_usage_binding_report.json';p.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('\nOutput:',out/'survey_usage_constraints.parquet')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();build(a.root.resolve())
