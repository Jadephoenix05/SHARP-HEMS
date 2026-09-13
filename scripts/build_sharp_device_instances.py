"""Expand AP inventory into versioned per-device simulator scenario inputs.
Preserves survey semantics. Scenario quantities never overwrite source counts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import pandas as pd

THERMAL = {'air_conditioner','refrigerator','geyser'}
CYCLES = {'washing_machine','electric_rice_cooker'}
CONTINUOUS = {'ceiling_fan','table_fan','air_cooler','led_bulb','led_tube',
              'cfl_bulb','cfl_tube','incandescent_bulb','modem_router'}

def expand(inventory, unknown_quantity):
    if inventory.duplicated(['template_id','appliance_type']).any():
        raise ValueError('Duplicate inventory keys')
    rows=[]; decisions=[]
    for r in inventory.to_dict('records'):
        appliance=r['appliance_type']; status=r['inventory_status']
        q=0; basis=''; assumption=False; coverage='INCLUDED'
        if bool(r.get('ownership_count_conflict',False)) or bool(r.get('invalid_count',False)):
            raise ValueError(f'Unresolved count issue: {r["template_id"]} {appliance}')
        if appliance=='geyser':
            if r.get('geyser_use_source_code')==1:
                q=r.get('geyser_reported_units_used')
                basis='REPORTED_GEYSER_UNITS_USED_NOT_OWNERSHIP'
            else:
                coverage='EXCLUDED_GEYSER_NONUSE' if r.get('geyser_use_source_code')==0 else 'EXCLUDED_GEYSER_USE_UNKNOWN'
        elif status=='NOT_OWNED':
            coverage='EXCLUDED_REPORTED_NOT_OWNED'
        elif status=='OWNED_QUANTITY_KNOWN':
            q=r['quantity'];basis='SOURCE_REPORTED_QUANTITY'
        elif status=='OWNED_QUANTITY_UNKNOWN':
            if unknown_quantity=='one':
                q=1;basis='SCENARIO_ONE_UNIT_QUANTITY_UNREPORTED';assumption=True
            else:
                coverage='EXCLUDED_QUANTITY_UNKNOWN'
        elif status=='OWNERSHIP_UNKNOWN':
            coverage='EXCLUDED_OWNERSHIP_UNKNOWN'
        else:
            raise ValueError(f'Unrecognised inventory status: {status}')
        if coverage=='INCLUDED':
            if pd.isna(q) or not math.isfinite(float(q)) or float(q)<=0 or not float(q).is_integer():
                raise ValueError(f'Invalid included count: {r["template_id"]} {appliance}: {q}')
            q=int(q)
        else:
            q=0
        decisions.append({'template_id':r['template_id'],'appliance_type':appliance,
            'split':r['split'],'source_inventory_status':status,'scenario_device_count':q,
            'quantity_basis':basis,'quantity_is_assumed':assumption,'inclusion_status':coverage})
        for unit in range(1,q+1):
            key=f'{r["template_id"]}:{appliance}:{unit}'
            rows.append({'device_id':'sharp_'+hashlib.sha256(key.encode()).hexdigest()[:24],
                'template_id':r['template_id'],'source_profile_id':r['source_profile_id'],
                'split':r['split'],'appliance_type':appliance,'unit_index':unit,
                'source_inventory_status':status,'scenario_category_quantity':q,
                'quantity_basis':basis,'quantity_is_assumed':assumption,
                'model_family_candidate':'thermal' if appliance in THERMAL else 'cycle' if appliance in CYCLES else 'continuous' if appliance in CONTINUOUS else 'event_or_task',
                'model_family_basis':'SIMULATOR_DESIGN_NOT_SURVEY_FIELD',
                'control_permission':'UNASSIGNED','power_model_status':'PENDING',
                'schedule_model_status':'PENDING','timing_constraints_status':'PENDING',
                'trace_assignment_status':'PENDING','is_simulation_instance':True,
                'survey_source':'IRES_2020','simulator_ready':False})
    devices=pd.DataFrame(rows); decisions=pd.DataFrame(decisions)
    if devices.empty or not devices.device_id.is_unique:
        raise ValueError('No devices or device-ID collision')
    assert len(devices)==int(decisions.scenario_device_count.sum())
    return devices,decisions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--unknown-quantity',choices=['one','exclude'],required=True,
        help='Explicit scenario rule for reported ownership without quantity')
    args=p.parse_args();root=args.root.resolve()
    source=root/'data/processed/appliance_inputs_v1/ap_inventory_with_splits_v1.parquet'
    inv=pd.read_parquet(source)
    if len(inv)!=12948 or inv.template_id.nunique()!=498:
        raise ValueError('Unexpected AP inventory population')
    d,decisions=expand(inv,args.unknown_quantity)
    # Different uncertainty rules have separate output folders.
    out=root/f'data/processed/simulator_devices_v1/unknown_quantity_{args.unknown_quantity}'
    out.mkdir(parents=True,exist_ok=True)
    d.to_parquet(out/'device_instances.parquet',index=False)
    decisions.to_parquet(out/'inventory_decisions.parquet',index=False)
    counts=d.groupby(['template_id','split']).size().rename('device_count').reset_index()
    households=inv[['template_id','split']].drop_duplicates().merge(counts,on=['template_id','split'],how='left',validate='one_to_one')
    households.device_count=households.device_count.fillna(0).astype(int)
    households.to_parquet(out/'household_device_counts.parquet',index=False)
    report={'status':'DEVICE_INSTANCES_BUILT','source_inventory_rows':len(inv),
        'households':len(households),'device_instances':len(d),
        'maximum_devices_per_household':int(households.device_count.max()),
        'households_without_modelled_devices':int(households.device_count.eq(0).sum()),
        'assumed_quantity_devices':int(d.quantity_is_assumed.sum()),
        'geyser_use_instances':int(d.appliance_type.eq('geyser').sum()),
        'unknown_quantity_policy':args.unknown_quantity,
        'inclusion_decisions':{str(k):int(v) for k,v in decisions.inclusion_status.value_counts().items()},
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'notes':['One device is one candidate action branch; padding masks are required for variable household sizes.',
                 'Excluded unknown ownership is not evidence of nonownership.',
                 'One-unit substitutions are explicit scenario assumptions, not imputed survey facts.',
                 'Geyser inclusion is based on reported use, not established ownership.',
                 'Power, schedules, permissions and constraints still require model integration.'],
        'full_simulator_ready':False,'master_release_ready':False}
    (out/'device_build_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('\nDEVICES BY TYPE:')
    print(d.groupby('appliance_type').size().to_string());print('\nOutput:',out)

if __name__=='__main__':main()
