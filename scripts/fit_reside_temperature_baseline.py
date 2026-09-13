"""Descriptive AC/temperature response baseline with chronological validation.
This is not a causal thermal calibration or a validated control model.
AC labels in RESIDE were partly informed by temperature behaviour.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

FEATURES=['intercept','temperature_t_c','ac_label_fraction','sin_clock','cos_clock']

def design(d):
    # Clock is local IST, not UTC hour.
    clock=d.timestamp_utc.dt.tz_convert('Asia/Kolkata')
    hours=clock.dt.hour.to_numpy()+clock.dt.minute.to_numpy()/60
    return np.column_stack([np.ones(len(d)),d.room_temperature_c.to_numpy(float),
        d.primary_ac_label_fraction.to_numpy(float),np.sin(2*np.pi*hours/24),np.cos(2*np.pi*hours/24)])

def fit_one(d):
    d=d.copy().sort_values('timestamp_utc').reset_index(drop=True)
    d['timestamp_utc']=pd.to_datetime(d.timestamp_utc,utc=True)
    d['next_timestamp_utc']=pd.to_datetime(d.next_timestamp_utc,utc=True)
    first=d.timestamp_utc.min().tz_convert('Asia/Kolkata').normalize()
    validation_start=first+pd.Timedelta(days=12)
    test_start=first+pd.Timedelta(days=16)
    train=d.loc[d.next_timestamp_utc<validation_start].copy()
    val=d.loc[(d.timestamp_utc>=validation_start)&(d.next_timestamp_utc<test_start)].copy()
    test=d.loc[d.timestamp_utc>=test_start]
    # Boundaries are excluded rather than using targets from the next split.
    if len(train)<100 or len(val)<50:raise ValueError('Insufficient train/validation pairs')
    x=design(train);y=train.next_room_temperature_c.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid regression data')
    beta,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
    vx=design(val);truth=val.next_room_temperature_c.to_numpy(float)
    prediction=vx@beta;persistence=val.room_temperature_c.to_numpy(float)
    mae=float(np.abs(prediction-truth).mean());pmae=float(np.abs(persistence-truth).mean())
    # Open-loop 24-hour rollouts use heldout, OBSERVED AC labels, not forecasts.
    # Do not bridge gaps, missing pairs or validation boundaries.
    roll_errors=[];roll_persistence=[];windows=0
    local_dates=val.timestamp_utc.dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d')
    for _,g in val.groupby(local_dates):
        if len(g)!=96:continue
        ts=g.timestamp_utc
        if not ts.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():continue
        if not g.next_timestamp_utc.iloc[:-1].reset_index(drop=True).equals(ts.iloc[1:].reset_index(drop=True)):continue
        state=float(g.room_temperature_c.iloc[0]);start=state
        gx=design(g);actual=g.next_room_temperature_c.to_numpy(float)
        for i in range(len(g)):
            v=gx[i].copy();v[1]=state;state=float(v@beta)
            roll_errors.append(abs(state-actual[i]));roll_persistence.append(abs(start-actual[i]))
        windows+=1
    flags=[]
    if rank<len(FEATURES):flags.append('RANK_DEFICIENT')
    if not 0<=beta[1]<1:flags.append('TEMPERATURE_COEFFICIENT_OUTSIDE_0_TO_1')
    if beta[2]>=0:flags.append('AC_COEFFICIENT_NONNEGATIVE')
    if mae>=pmae:flags.append('NO_ONE_STEP_IMPROVEMENT_OVER_PERSISTENCE')
    if not windows:flags.append('NO_COMPLETE_VALIDATION_DAY_ROLLOUT')
    if windows and np.mean(roll_errors)>=np.mean(roll_persistence):flags.append('NO_ROLLOUT_IMPROVEMENT_OVER_PERSISTENCE')
    result={'training_pairs':len(train),'validation_pairs':len(val),'reserved_test_pairs':len(test),
        'validation_start_ist':validation_start.isoformat(),'reserved_test_start_ist':test_start.isoformat(),
        'rank':int(rank),'coefficients':{k:float(v) for k,v in zip(FEATURES,beta)},
        'validation_one_step_mae_c':mae,'persistence_one_step_mae_c':pmae,
        'validation_one_step_rmse_c':float(np.sqrt(np.mean((prediction-truth)**2))),
        'validation_rollout_days':windows,
        'validation_rollout_mae_c':float(np.mean(roll_errors)) if windows else None,
        'persistence_rollout_mae_c':float(np.mean(roll_persistence)) if windows else None,
        'training_ac_off_pairs':int(train.primary_ac_label_fraction.eq(0).sum()),
        'training_ac_on_pairs':int(train.primary_ac_label_fraction.eq(1).sum()),
        'review_flags':flags,'test_evaluated':False,
        'causal_cooling_effect_established':False,'approved_for_control_simulation':False}
    predictions=val[['candidate_house_id','timestamp_utc','next_timestamp_utc','room_temperature_c',
        'next_room_temperature_c','primary_ac_label_fraction']].copy()
    predictions['predicted_next_temperature_c']=prediction
    predictions['persistence_prediction_c']=persistence
    return result,predictions


def build(root):
    source=root/'data/processed/reside_thermal_inputs_v1/observed_temperature_response_pairs.parquet'
    data=pd.read_parquet(source)
    if data.candidate_house_id.nunique()!=11:raise ValueError('Expected eleven houses')
    if data.duplicated(['candidate_house_id','timestamp_utc']).any():raise ValueError('Duplicate thermal pair key')
    out=root/'data/processed/reside_temperature_baseline_v1';out.mkdir(parents=True,exist_ok=True)
    summary=[];predictions=[]
    for house,g in data.groupby('candidate_house_id'):
        result,pred=fit_one(g);result['house']=int(house)
        (out/f'house_{int(house):02d}_model.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        predictions.append(pred)
        summary.append({k:v for k,v in result.items() if k not in ['coefficients','review_flags']} |
                       {'review_flags':'|'.join(result['review_flags'])})
        print(f'House {int(house):02d}: validation MAE {result["validation_one_step_mae_c"]:.4f} C | persistence {result["persistence_one_step_mae_c"]:.4f} C | flags {len(result["review_flags"])}',flush=True)
    table=pd.DataFrame(summary);table.to_csv(out/'model_comparison.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'validation_predictions.parquet',index=False)
    report={'status':'DESCRIPTIVE_TEMPERATURE_BASELINES_FITTED','houses':len(summary),
      'houses_beating_one_step_persistence':int(table.validation_one_step_mae_c.lt(table.persistence_one_step_mae_c).sum()),
      'houses_with_review_flags':int(table.review_flags.ne('').sum()),
      'test_evaluated':False,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'limitations':['Observational regression, not causal cooling physics.',
        'Source AC tags partly used temperature changes, limiting independent validation.',
        'No outdoor temperature, internal heat gains, solar gains or room geometry fitted.',
        'Rollouts condition on observed AC labels, not forecast controls.',
        'Source date discrepancy and possible clock drift remain documented upstream.',
        'Predictions are not a basis for hardware safety decisions.'],
      'approved_for_control_simulation':False,'full_simulator_ready':False,'master_release_ready':False}
    (out/'temperature_baseline_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n',json.dumps(report,indent=2));print('\nOutput:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();build(a.root.resolve())
