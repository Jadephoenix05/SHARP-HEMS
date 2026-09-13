"""Freeze source-household splits and catalogue existing electrical summaries.
No ratings, schedules, ownership counts or physical OFF labels are invented.
Run from the SHARP project root. Existing frozen split assignments are immutable.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

VERSION = 'sharp_appliance_inputs_v1'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def assignments(ids, source, fractions=(.7,.15), labels=('train','validation','test')):
    ids = sorted(set(map(str, ids)), key=lambda x: hashlib.sha256((VERSION+':'+source+':'+x).encode()).hexdigest())
    n = len(ids); a = int(n*fractions[0]); b = a+int(n*fractions[1])
    return pd.DataFrame({'source':source, 'household_id':ids,
        'split':[labels[0] if i<a else labels[1] if i<b else labels[2] for i in range(n)]})

def stats(v):
    v = pd.to_numeric(v, errors='coerce'); good = v[np.isfinite(v) & v.ge(0)]
    return {'rows':len(v), 'nonnegative_finite_rows':len(good),
        'missing_or_invalid_rows':len(v)-len(good),
        'zero_rows':int(good.eq(0).sum()),
        'mean_of_interval_means_w':float(good.mean()) if len(good) else None,
        'p50_interval_mean_w':float(good.quantile(.5)) if len(good) else None,
        'p95_interval_mean_w':float(good.quantile(.95)) if len(good) else None,
        'maximum_interval_mean_w':float(good.max()) if len(good) else None}

def main(root):
    out=root/'data/processed/appliance_inputs_v1'; out.mkdir(parents=True,exist_ok=True)
    reports=root/'reports'; reports.mkdir(exist_ok=True)
    hp=root/'data/processed/household_templates/ap_household_templates_v2.parquet'
    ip=root/'data/processed/household_templates/ap_appliance_ownership_v3.parquet'
    mp=root/'data/registry/refit_appliance_mapping.csv'
    ep=root/'data/interim/emarc/emarc_deployment_registry_v1.csv'
    hh=pd.read_parquet(hp); inv=pd.read_parquet(ip); mapping=pd.read_csv(mp); em=pd.read_csv(ep)
    assert len(hh)==498 and hh.template_id.is_unique, 'Unexpected AP household population'
    assert len(inv)==12948 and not inv.duplicated(['template_id','appliance_type']).any(), 'Inventory key/count mismatch'
    assert set(inv.template_id)==set(hh.template_id), 'Inventory household mismatch'
    assert len(mapping)==200 and not mapping.duplicated(['house_id','channel']).any(), 'REFIT mapping incomplete'
    splits=pd.concat([assignments(hh.template_id,'IRES_AP'),
        assignments(mapping.house_id,'REFIT'),
        assignments(em.household_id,'EMARC',(.8,0),('calibration','unused','heldout'))],ignore_index=True)
    sp=out/'household_splits_v1.csv'
    if sp.exists():
        old=pd.read_csv(sp,dtype=str)
        pd.testing.assert_frame_equal(old.reset_index(drop=True),splits.astype(str).reset_index(drop=True))
    else: splits.to_csv(sp,index=False)
    ap=splits.loc[splits.source.eq('IRES_AP'),['household_id','split']].rename(columns={'household_id':'template_id'})
    hh.merge(ap,on='template_id',validate='one_to_one').to_parquet(out/'ap_households_with_splits_v1.parquet',index=False)
    inv.merge(ap,on='template_id',validate='many_to_one').to_parquet(out/'ap_inventory_with_splits_v1.parquet',index=False)
    files=[hp,ip,mp,ep]; rows=[]; checks=[]
    rs=splits[splits.source.eq('REFIT')].set_index('household_id')['split'].to_dict()
    for house in sorted(mapping.house_id.unique()):
        p=root/f'data/processed/refit_canonical_v1/refit_house_{int(house):02d}_channels_v1.parquet'
        d=pd.read_parquet(p,columns=['source_house_id','channel_id','appliance_name_source','timestamp_source','power_w','channel_coverage_known'])
        assert d.source_house_id.eq(house).all(), f'Wrong household: {p}'
        expected=set(mapping.loc[mapping.house_id.eq(house),'channel'])
        observed=set(d.channel_id.unique()); complete=observed==expected
        checks.append({'house':int(house),'all_ten_channels_present':complete,'rows':len(d)})
        for ch,g in d.groupby('channel_id'):
            t=pd.to_datetime(g.timestamp_source,errors='coerce')
            r={'source':'REFIT','household_id':str(house),'channel_id':int(ch),
               'source_label':str(g.appliance_name_source.iloc[0]),'split':rs[str(house)],
               'path':p.relative_to(root).as_posix(),'time_basis':'SOURCE_CLOCK_UNVERIFIED',
               'channel_coverage_verified':bool(g.channel_coverage_known.fillna(False).all()),
               'duplicate_timestamp_rows':int(t.duplicated(keep=False).sum()),
               'invalid_timestamp_rows':int(t.isna().sum()),
               'first_timestamp':str(t.min()),'last_timestamp':str(t.max()),
               'trace_assignment_status':'CANDIDATE_REQUIRES_APPLIANCE_MATCH_REVIEW',**stats(g.power_w)}
            rows.append(r)
        files.append(p); print(f'REFIT house {house:02}: {len(d):,} channel rows',flush=True)
    iawe_files=sorted((root/'data/interim/iawe/15min').glob('iawe_channel_*_15min_v1.parquet'))
    assert len(iawe_files)==12, 'Expected 12 iAWE channel summaries'
    for p in iawe_files:
        d=pd.read_parquet(p); ch=int(d.channel_id.iloc[0]); t=pd.to_datetime(d.interval_start_utc,utc=True)
        # iAWE is one calibration household; never claim independent heldout homes.
        r={'source':'IAWE','household_id':'iawe_01','channel_id':ch,
           'source_label':str(d.appliance_name.iloc[0]),'split':'calibration_only',
           'path':p.relative_to(root).as_posix(),'time_basis':'UTC',
           'channel_coverage_verified':False,
           'duplicate_timestamp_rows':int(t.duplicated(keep=False).sum()),
           'invalid_timestamp_rows':int(t.isna().sum()),
           'first_timestamp':str(t.min()),'last_timestamp':str(t.max()),
           'sample_screen_bins':int(d.passes_sample_count_screen.fillna(False).sum()),
           'trace_assignment_status':'CANDIDATE_REQUIRES_APPLIANCE_MATCH_REVIEW',**stats(d.power_sample_mean_w)}
        rows.append(r); files.append(p)
    cat=pd.DataFrame(rows); cat.to_csv(out/'electrical_trace_catalog_v1.csv',index=False)
    assert not cat.duplicated(['source','household_id','channel_id']).any()
    inv.groupby(['appliance_type','inventory_status']).size().rename('rows').reset_index().to_csv(out/'ap_inventory_coverage_v1.csv',index=False)
    structural=all(x['all_ten_channels_present'] for x in checks) and cat.duplicate_timestamp_rows.eq(0).all() and cat.invalid_timestamp_rows.eq(0).all()
    report={'status':'INPUT_CATALOG_BUILT' if structural else 'FAIL_INCOMPLETE_OR_DUPLICATE_TRACES',
        'households':len(hh),'inventory_rows':len(inv),'refit_channels':int(cat.source.eq('REFIT').sum()),
        'iawe_channels':int(cat.source.eq('IAWE').sum()),'split_counts':splits.groupby(['source','split']).size().to_dict(),
        'refit_house_checks':checks,'master_release_ready':False,
        'remaining':['Review and match source channels to Indian appliance types',
          'Assign documented simulation ratings and usage models where measured traces are unavailable',
          'Build and test action constraints, reward, transitions and BDQ training',
          'Verify redistribution scope and clean-download reproducibility'],
        'notes':['Splits are household-disjoint and frozen; temporal context splits must also be enforced.',
          'Catalog statistics include all available nonnegative interval means, including low-coverage bins.',
          'Zero readings are not asserted to be physical OFF states.',
          'REFIT remains UK data; iAWE remains one Indian calibration household.',
          'No electrical trace has yet been assigned to an AP household.'],
        'source_sha256':{p.relative_to(root).as_posix():digest(p) for p in files}}
    report['split_counts']={':'.join(k):int(v) for k,v in report['split_counts'].items()}
    (reports/'sharp_appliance_inputs_v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nSHARP APPLIANCE INPUTS'); print(report['status']); print(json.dumps(report['split_counts'],indent=2))
    print(f'REFIT channels: {report["refit_channels"]}; iAWE channels: {report["iawe_channels"]}')
    print('Report: reports/sharp_appliance_inputs_v1.json')
    if not structural: raise SystemExit('Input completeness check failed; do not use sample-only traces for the full build.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args(); main(args.root.resolve())
