"""Reproducible TUS adult-location donor scenarios for AP household templates.
These are synthetic pairings, not observed occupancy in the target households.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
VERSION='adult_location_scenario_v1'

def h(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

def build(root):
    hp=root/'data/processed/appliance_inputs_v1/ap_households_with_splits_v1.parquet'
    tp=root/'data/interim/occupancy_profiles/tus2024_ap_complete_location_diaries_v1.parquet'
    hh=pd.read_parquet(hp);tus=pd.read_parquet(tp)
    if len(hh)!=498 or not hh.template_id.is_unique:raise ValueError('Invalid household population')
    if not tus.profile_id.is_unique:raise ValueError('Duplicate TUS diary IDs')
    columns=[f'home_fraction_{i//4:02d}{i%4*15:02d}' for i in range(96)]
    if not set(columns).issubset(tus.columns):raise ValueError('Missing 15-minute location fields')
    tus=tus.loc[pd.to_numeric(tus.age,errors='coerce').ge(18)
        & tus.complete_location_diary.eq(True)
        & tus.day_type.astype(str).str.strip().str.casefold().eq('normal day')].copy()
    if not tus.source_person_id.astype(str).str.match(r'^\d+_\d+$').all():
        raise ValueError('Unexpected source person identifier; grouping requires review')
    values=tus[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('Invalid location fractions')
    # Group all people sharing the mirror's person-ID prefix. Ambiguous merged
    # prefixes stay together; this is a conservative proxy, not verified HH IDs.
    tus['donor_group_id']=tus.source_person_id.astype(str).str.rsplit('_',n=1).str[0].map(lambda s:'tusgroup_'+h(s)[:24])
    def group_split(group):
        value=int(h(VERSION+':'+group),16)%100
        return 'train' if value<70 else 'validation' if value<85 else 'test'
    tus['donor_split']=tus.donor_group_id.map(group_split)
    tus['sector_key']=tus.sector.astype(str).str.strip().str.casefold()
    hh['sector_key']=hh.urban_rural.astype(str).str.strip().str.casefold()
    pools={(s,r,d):g.sort_values('profile_id').reset_index(drop=True)
        for (s,r,d),g in tus.groupby(['donor_split','sector_key','day_of_week'])}
    assignments=[];parts=[]
    for row in hh.sort_values('template_id').itertuples(index=False):
        for day_number,day in enumerate(DAYS):
            key=(row.split,row.sector_key,day)
            pool=pools.get(key)
            if pool is None or pool.empty:raise ValueError(f'No matching adult diary pool: {key}')
            index=int(h(VERSION+':'+row.template_id+':'+day),16)%len(pool)
            donor=pool.iloc[index]
            assignments.append({'template_id':row.template_id,'split':row.split,
                'weekday':day,'weekday_number':day_number,'sector':row.sector_key,
                'donor_profile_id':donor.profile_id,'donor_group_id':donor.donor_group_id,
                'donor_age':int(donor.age),'donor_day_type':donor.day_type,
                'pool_diaries':len(pool),'assignment_is_synthetic':True})
            parts.append(pd.DataFrame({'template_id':row.template_id,'split':row.split,
                'weekday_number':day_number,'step_of_day':range(96),
                'adult_reported_home_fraction_proxy':donor[columns].to_numpy(dtype=float),
                'donor_profile_id':donor.profile_id,
                'occupancy_scope':'ONE_ADULT_LOCATION_PROXY_NOT_WHOLE_HOUSEHOLD',
                'attention_status':'NOT_MODELLED','is_synthetic_assignment':True}))
    mapping=pd.DataFrame(assignments);profiles=pd.concat(parts,ignore_index=True)
    if mapping.groupby('donor_group_id').split.nunique().gt(1).any():
        raise ValueError('Donor group crossed training/evaluation splits')
    assert len(mapping)==498*7 and len(profiles)==498*7*96
    assert not profiles.duplicated(['template_id','weekday_number','step_of_day']).any()
    out=root/'data/processed/location_scenarios_v1';out.mkdir(parents=True,exist_ok=True)
    frozen=out/'donor_assignments.parquet'
    if frozen.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(frozen),mapping,check_dtype=False)
    else: mapping.to_parquet(frozen,index=False)
    profiles.to_parquet(out/'adult_location_weekly_proxy.parquet',index=False,compression='zstd')
    report={'status':'ADULT_LOCATION_SCENARIOS_BUILT','households':len(hh),
        'weekday_assignments':len(mapping),'quarter_hour_rows':len(profiles),
        'eligible_adult_normal_day_diaries':len(tus),'used_donor_diaries':int(mapping.donor_profile_id.nunique()),
        'donor_groups_crossing_splits':0,'weekday_convention':'Monday=0, Sunday=6',
        'survey_weighting':'UNWEIGHTED_DONOR_SAMPLING',
        'notes':['Seven independently selected diary days form a synthetic week, not an observed week.',
            'One adult location proxy does not establish whole-household occupancy or headcount.',
            'Being home does not establish awareness of a notification or consent to an action.',
            'No absence-of-override acceptance labels are generated.',
            '2024 survey diaries paired with other-year weather are scenario inputs, not historical observations.',
            'Normal-day adult diaries are a selected sample, not population-representative.',
            'Donor grouping uses a conservative mirror-ID prefix; original household linkage remains unverified.'],
        'input_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [hp,tp]},
        'full_simulator_ready':False,'master_release_ready':False}
    (out/'location_scenario_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('\nOutput:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();build(a.root.resolve())
