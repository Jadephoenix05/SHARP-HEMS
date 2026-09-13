"""Documented-unit RESIDE inputs and adjacent temperature-response pairs.
Source: https://doi.org/10.1186/s42162-022-00225-4
Not an identified causal cooling model; no current-to-power conversion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

SOURCE='https://link.springer.com/article/10.1186/s42162-022-00225-4'

def canonical(data, house):
    d=data.copy()
    if not d.candidate_house_id.eq(house).all():raise ValueError('House identity mismatch')
    t=pd.to_datetime(d.interval_start_ist,utc=True,errors='raise')
    if t.isna().any() or t.duplicated().any():raise ValueError('Invalid/duplicate timestamps')
    d['timestamp_utc']=t;d=d.sort_values('timestamp_utc').reset_index(drop=True)
    d['room_temperature_c']=pd.to_numeric(d.temperature_sample_mean_source_units,errors='raise')
    d['room_relative_humidity_pct']=pd.to_numeric(d.humidity_sample_mean_source_units,errors='raise')
    d['primary_ac_label_fraction']=pd.to_numeric(d.ac_status_sample_mean,errors='raise')
    if not d.primary_ac_label_fraction.dropna().between(0,1).all():raise ValueError('AC fractions outside [0,1]')
    if not d.room_relative_humidity_pct.dropna().between(0,100).all():raise ValueError('Humidity outside [0,100]')
    for phase in ['r','y','b']:
        d[f'phase_{phase}_current_a']=pd.to_numeric(d[f'phase_{phase}_sample_mean_source_units'],errors='raise')/1000
    d['mapping_status']='HOUSE_SUFFIX_CONFIRMED_BY_PRIMARY_PAPER'
    d['measurement_units_status']='C_A_PERCENT_CONFIRMED_BY_PRIMARY_PAPER'
    d['temperature_location']='ROOM_WITH_MOST_FREQUENTLY_USED_AC'
    d['ac_label_provenance']='MANUAL_TAGGING_USING_CURRENT_AND_TEMPERATURE_CHANGE'
    d['sensor_clock_drift_note']='PAPER_REPORTS_POSSIBLE_ENVILOG_SHIFT_OF_A_FEW_MINUTES'
    d['semantic_source']=SOURCE
    d['release_ready']=False
    fields=['room_temperature_c','room_relative_humidity_pct','primary_ac_label_fraction',
            'phase_r_current_a','phase_y_current_a','phase_b_current_a']
    if np.isinf(d[fields].to_numpy(float)).any():raise ValueError('Infinite measurements')
    # Keep exactly adjacent complete-bin temperature pairs; never span gaps.
    valid=(d.temperature_valid_samples.eq(3)&d.temperature_conflict_timestamps.eq(0)
           &d.ac_status_valid_samples.eq(15)&d.ac_status_conflict_timestamps.eq(0)
           &d.room_temperature_c.notna()&d.primary_ac_label_fraction.notna())
    adjacent=d.timestamp_utc.shift(-1).sub(d.timestamp_utc).eq(pd.Timedelta(minutes=15))
    keep=valid&valid.shift(-1,fill_value=False)&adjacent
    pairs=pd.DataFrame({'candidate_house_id':house,'timestamp_utc':d.timestamp_utc,
        'next_timestamp_utc':d.timestamp_utc.shift(-1),
        'room_temperature_c':d.room_temperature_c,
        'next_room_temperature_c':d.room_temperature_c.shift(-1),
        'temperature_change_c':d.room_temperature_c.shift(-1)-d.room_temperature_c,
        'primary_ac_label_fraction':d.primary_ac_label_fraction}).loc[keep].copy()
    pairs['causal_cooling_effect_established']=False
    pairs['time_step_minutes']=15
    return d,pairs


def build(root):
    source=root/'data/interim/reside_ac_clean/timezone_aligned'
    out=root/'data/processed/reside_thermal_inputs_v1';out.mkdir(parents=True,exist_ok=True)
    all_pairs=[];reports=[];hashes={}
    files=sorted(source.glob('house_*_15min_v1.parquet'))
    if len(files)!=11:raise ValueError('Expected eleven RESIDE files')
    seen=set()
    for p in files:
        house=int(p.name.split('_')[1]);seen.add(house)
        raw=pd.read_parquet(p)
        if len(raw)!=1824:raise ValueError(f'{p.name}: full 1824 intervals required, not uploaded samples')
        d,pairs=canonical(raw,house)
        d.to_parquet(out/f'house_{house:02d}_documented_units.parquet',index=False)
        all_pairs.append(pairs)
        reports.append({'house':house,'intervals':len(d),'adjacent_complete_pairs':len(pairs),
            'missing_temperature':int(d.room_temperature_c.isna().sum()),
            'source_year_discrepancy_preserved':bool(d.source_year_discrepancy.any())})
        hashes[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        print(f'House {house:02d}: {len(pairs):,} adjacent temperature pairs',flush=True)
    if seen!=set(range(1,12)):raise ValueError('Unexpected house IDs')
    pairs=pd.concat(all_pairs,ignore_index=True)
    if pairs.duplicated(['candidate_house_id','timestamp_utc']).any():raise ValueError('Duplicate pair keys')
    pairs.to_parquet(out/'observed_temperature_response_pairs.parquet',index=False)
    evidence={'primary_paper':SOURCE,'doi':'10.1186/s42162-022-00225-4',
      'confirmed':['Gxx and Exx suffixes identify the same house.',
                   'Phase current is in milliamperes; canonical current is amperes.',
                   'Envilog measures temperature in Celsius and relative humidity in percent in the most-used AC room.',
                   'Binary AC labels were manually checked using current and temperature changes.'],
      'remaining':['Paper abstract states May 2019, while Data records states May 2021; CSV dates remain unchanged.',
                   'Potential Envilog timing drift prevents assuming exact physical response timing.',
                   'Temperature-informed AC labels cannot independently validate causal cooling response.',
                   'Current alone does not establish active electrical power.'],
      'review_scope':'Documented units and file mapping; not thermal-model validation'}
    (out/'source_semantics.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    report={'status':'DOCUMENTED_RESIDE_THERMAL_INPUTS_BUILT','houses':len(files),
      'intervals':sum(r['intervals'] for r in reports),'temperature_pairs':len(pairs),
      'house_reports':reports,'input_sha256':hashes,'causal_thermal_model_validated':False,
      'full_simulator_ready':False,'master_release_ready':False}
    (out/'thermal_input_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nRESIDE THERMAL INPUTS BUILT');print('Houses:',len(files));print('Intervals:',report['intervals'])
    print('Temperature pairs:',len(pairs));print('Causal thermal model validated: False');print('Output:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();build(a.root.resolve())
