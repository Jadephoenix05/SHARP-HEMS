"""Explicit baseline simulation parameters; not validated Indian appliance ratings.
No values in this file authorize control of physical household equipment.
The baseline must undergo sensitivity analysis and aggregate-load validation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import pandas as pd

# Illustrative electrical W assumptions chosen for simulator development.
# They are NOT manufacturer ratings, population averages or calibrated values.
# Mode names describe candidate simulation behaviour, not actual user consent.
BASELINE = {
 'air_conditioner': (1500,'thermostatic','temperature_service'),
 'air_cooler': (150,'continuous','comfort_service'),
 'ceiling_fan': (60,'continuous','comfort_service'),
 'table_fan': (40,'continuous','comfort_service'),
 'cfl_bulb': (15,'continuous','lighting_service'),
 'cfl_tube': (36,'continuous','lighting_service'),
 'incandescent_bulb': (60,'continuous','lighting_service'),
 'led_bulb': (9,'continuous','lighting_service'),
 'led_tube': (18,'continuous','lighting_service'),
 'desktop': (150,'continuous','user_session'),
 'electric_iron': (1000,'task','user_session'),
 'electric_kettle': (1500,'task','user_session'),
 'electric_rice_cooker': (500,'cycle','noninterruptible_when_started'),
 'geyser': (2000,'thermostatic','hot_water_service'),
 'laptop_tablet': (40,'continuous','user_session'),
 'mixer_grinder': (500,'task','user_session'),
 'modem_router': (10,'continuous','protected_service'),
 'refrigerator': (100,'thermostatic','protected_service'),
 'television': (75,'continuous','user_session'),
 'washing_machine': (500,'cycle','noninterruptible_when_started'),
 'water_pump': (750,'task','water_service'),
 'water_purifier': (25,'task','water_service'),
}
# Keep air conditioners as two distinct observed examples.
EMPIRICAL = {'air_conditioner':[4,5], 'refrigerator':[3],
             'laptop_tablet':[7], 'television':[10]}

# Power source precedence, highest first:
#   1. REFIT single-appliance channels from the SAME split (canonical electrical
#      source per the project registry; split-aware so no split informs another).
#   2. iAWE observed proxy channels (Indian, but one household only).
#   3. The explicit engineering assumptions above.
# A REFIT value is a measured UK appliance applied to an Indian household: a
# documented transfer assumption, never a measured Indian rating.
REFIT_LIBRARY = ('data/processed/refit_power_library_v3/refit_power_library.json')


def build(root):
    base=root/'data/processed/simulator_devices_v1/unknown_quantity_one'
    device_path=base/'device_instances.parquet'
    params_path=root/'data/processed/appliance_inputs_v1/iawe_empirical_power_parameters_v1.csv'
    usage_path=base/'survey_usage_constraints.parquet'
    devices=pd.read_parquet(device_path)
    parameters=pd.read_csv(params_path)
    if parameters.channel_id.duplicated().any():raise ValueError('Duplicate calibration channel')
    lookup=parameters.set_index('channel_id')
    unknown=set(devices.appliance_type)-set(BASELINE)
    if unknown:raise ValueError(f'Unconfigured appliance categories: {unknown}')
    for channel in [3,4,5,7,10]:
        row=lookup.loc[channel]
        value=float(row.above_threshold_median_w)
        if not math.isfinite(value) or value<=0 or row.above_threshold_intervals<=0:
            raise ValueError(f'Invalid calibration channel {channel}')
    refit_path=root/REFIT_LIBRARY
    refit={} 
    if refit_path.exists():
        refit=json.loads(refit_path.read_text(encoding='utf-8'))['library']
        print('REFIT split-aware library loaded:',sorted(refit))
    else:
        print('No REFIT library found; falling back to iAWE proxies and assumptions')
    rows=[]
    for r in devices.to_dict('records'):
        app=r['appliance_type']; watt,mode,service=BASELINE[app]
        provenance='EXPLICIT_ILLUSTRATIVE_SIMULATION_ASSUMPTION'
        channel=None
        low,high=0.75,1.25
        sensitivity='ILLUSTRATIVE_STRESS_TEST_NOT_CONFIDENCE_INTERVAL'
        refit_entry=refit.get(app,{}).get(r['split'])
        if refit_entry:
            watt=float(refit_entry['median_on_power_w'])
            if not math.isfinite(watt) or watt<=0:
                raise ValueError(f'Invalid REFIT power for {app}/{r["split"]}')
            provenance='REFIT_SAME_SPLIT_SINGLE_APPLIANCE_MEDIAN_ON_POWER_TRANSFER'
            low=float(refit_entry['p25_on_power_w'])/watt
            high=float(refit_entry['p75_on_power_w'])/watt
            sensitivity='REFIT_SAME_SPLIT_ON_POWER_P25_TO_P75'
        elif app in EMPIRICAL:
            options=EMPIRICAL[app]
            channel=options[int(hashlib.sha256(r['device_id'].encode()).hexdigest(),16)%len(options)]
            watt=float(lookup.loc[channel,'above_threshold_median_w'])
            provenance='IAWE_15MIN_ABOVE_5W_MEDIAN_USED_AS_SIMULATION_PROXY'
        rows.append({**r,'scenario_model_version':'baseline_power_v1',
            'operating_power_proxy_w':float(watt),'power_parameter_basis':provenance,
            'calibration_channel_id':channel,'is_measured_device_rating':False,
            'power_sensitivity_low_multiplier':low,'power_sensitivity_high_multiplier':high,
            'sensitivity_range_basis':sensitivity,
            'trace_assignment_status':('REFIT_SAME_SPLIT_TYPE_MATCHED' if refit_entry
                                       else 'NO_SAME_SPLIT_REFIT_EQUIVALENT'),
            'refit_grounded':bool(refit_entry),
            'refit_channels_used':int(refit_entry['refit_channels']) if refit_entry else 0,
            'dynamics_family':mode,'service_role':service,
            'power_model_status':'PROXY_PARAMETER_ASSIGNED_NOT_VALIDATED',
            'control_permission':'SIMULATION_ONLY_POLICY_PENDING',
            'schedule_model_status':'PENDING','timing_constraints_status':'PENDING',
            'simulator_ready':False})
    models=pd.DataFrame(rows)
    assert len(models)==4124 and models.device_id.is_unique
    assert models.operating_power_proxy_w.gt(0).all()
    out=base/'baseline_power_v1';out.mkdir(parents=True,exist_ok=True)
    models.to_parquet(out/'device_power_models.parquet',index=False)
    category=models.groupby(['appliance_type','power_parameter_basis']).agg(
        devices=('device_id','size'),minimum_power_proxy_w=('operating_power_proxy_w','min'),
        maximum_power_proxy_w=('operating_power_proxy_w','max')).reset_index()
    category.to_csv(out/'power_model_summary.csv',index=False)
    config={'scenario_id':'baseline_power_v1','parameter_policy':'EXPLICIT_ASSUMPTIONS_AND_OBSERVED_PROXIES',
      'engineering_power_assumptions_w':{k:v[0] for k,v in BASELINE.items()},
      'empirical_proxy_channels':EMPIRICAL,'seed_rule':'SHA256(device_id) modulo channel count',
      'evidence_limits':['Observed 15-minute means are not instantaneous ON power or nameplate ratings.',
        'iAWE represents one Indian household, not an Indian population distribution.',
        'Shared iAWE calibration does not provide independent heldout Indian-household evaluation.',
        'Laptop readings are a proxy for the broader laptop/tablet category.',
        'Water-filter observations are not used as a universal water-purifier model.',
        'Engineering values and sensitivity ranges are explicit development assumptions.',
        'REFIT power is a same-split UK single-appliance median, not an Indian rating.',
        'Appliance and split combinations with no REFIT channel keep declared assumptions.',
        'Power parameters alone do not specify cycle timing, thermal behaviour or usage schedules.'],
      'units':'electrical watts','hardware_control_authorized':False,
      'full_simulator_ready':False,'master_release_ready':False}
    (out/'model_assumptions.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    report={'status':'BASELINE_POWER_PARAMETERS_ASSIGNED','devices':len(models),
        'observed_proxy_devices':int(models.calibration_channel_id.notna().sum()),
        'explicit_assumption_devices':int(models.calibration_channel_id.isna().sum()),
        'source_hashes':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [device_path,params_path,usage_path]},
        'parameter_validation_scope':'structure_and_finite_values_only',
        'aggregate_load_validation':'PENDING','full_simulator_ready':False,'master_release_ready':False}
    (out/'power_model_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('\n',category.to_string(index=False));print('\nOutput:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args();build(args.root.resolve())
