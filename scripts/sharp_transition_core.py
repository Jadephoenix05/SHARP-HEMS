"""Connect SHARP shield, appliance dynamics, billing and reward atomically.
The caller supplies calibrated/scenario-labelled appliance dynamics and states.
This module does not invent power ratings, attention, comfort or occupancy.
"""
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import json
import math

from sharp_action_shield import Device, apply_shield, advance_timers, validate_device
from sharp_reward_billing import BillingLedger, RewardWeights, reward_components
from sharp_apcpdcl_tariff import Tariff


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def feedback_status(*, changed, seen, window_complete, override_requested):
    if type(seen) is not bool or type(window_complete) is not bool:
        raise ValueError('Attention and completed-window flags must be explicit booleans')
    if override_requested:
        return 'EXPLICIT_OVERRIDE_REQUEST'
    if not changed:
        return 'NO_INTERVENTION'
    if not seen:
        return 'CENSORED_NOT_SEEN'
    if not window_complete:
        return 'CENSORED_RESPONSE_WINDOW_OPEN'
    return 'SEEN_NO_OVERRIDE_WEAK_EVIDENCE'


def transition_step(*, ledger, weights, episode_id, step_id, devices, requested,
                    state, physics_step, max_import_w, base_load_w,
                    available_solar_w, grid_peak_severity, policy_source,
                    data_release_version, human_actions=None,
                    intervention_seen=None, response_window_complete=None,
                    terminated=False, truncated=False):
    """physics_step(executed_actions) returns a dictionary with:
      next_devices: list[Device], appliance_power_w: list[float], next_state: dict,
      discomfort_units: float, unmet_service_units: float.
    The callback must be pure: do not mutate external states or input devices.
    state/next_state are caller-defined observations, not raw future forcing.
    These 15-minute transitions use mean appliance power, not instant power.
    """
    if not episode_id or not policy_source or not data_release_version:
        raise ValueError('Episode, policy and release identifiers required')
    if type(step_id) is not int or step_id < 0:
        raise ValueError('Invalid episode step ID')
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ValueError('Explicit termination flags required')
    if terminated and truncated:
        raise ValueError('Choose termination or time-limit truncation')
    if not isinstance(state, dict):
        raise ValueError('Observation state must be a dictionary')
    # Reject NaN observations before any accounting changes.
    json.dumps(state, allow_nan=False)
    weights.validate()
    shield = apply_shield(devices, requested, max_import_w=max_import_w,
        base_load_w=base_load_w,available_solar_w=available_solar_w,
        human_actions=human_actions)
    n = len(devices)
    seen = [False]*n if intervention_seen is None else list(intervention_seen)
    complete = [False]*n if response_window_complete is None else list(response_window_complete)
    if len(seen) != n or len(complete) != n:
        raise ValueError('Feedback vector length mismatch')
    # The projected policy action before human requests identifies interventions.
    policy_shield = apply_shield(devices, requested, max_import_w=max_import_w,
        base_load_w=base_load_w,available_solar_w=available_solar_w)
    feedback = [feedback_status(changed=d.current_on != bool(a),seen=s,
        window_complete=w,override_requested=d.device_id in (human_actions or {}))
        for d,a,s,w in zip(devices,policy_shield['executed_actions'],seen,complete)]
    physical = physics_step(list(shield['executed_actions']))
    next_devices = physical['next_devices']
    powers = physical['appliance_power_w']
    if len(next_devices) != n or len(powers) != n:
        raise ValueError('Appliance dynamics length mismatch')
    timers = advance_timers(devices, shield['executed_actions'])
    for d,new,timer in zip(devices,next_devices,timers):
        validate_device(new)
        if new.device_id != d.device_id:
            raise ValueError('Dynamics changed device order or identity')
        if new.current_on != timer['current_on'] or new.elapsed_state_steps != timer['elapsed_state_steps']:
            raise ValueError('Dynamics state does not match executed action/timers')
    powers = [finite_nonnegative(x,'appliance_power_w') for x in powers]
    if not isinstance(physical['next_state'], dict):
        raise ValueError('Next observation must be a dictionary')
    json.dumps(physical['next_state'], allow_nan=False)
    demand_w = base_load_w + sum(powers)
    import_w = max(0.0,demand_w-available_solar_w)
    export_w = max(0.0,available_solar_w-demand_w)
    # A bad power forecast must not masquerade as a feasible transition.
    actual_excess_w = max(0.0,import_w-max_import_w)
    trial = BillingLedger.restore(ledger.tariff,ledger.state())
    before = ledger.state()
    billing = trial.charge(step_index=ledger.last_step+1,import_kwh=import_w*.25/1000)
    switching = sum(d.current_on != bool(a) for d,a in zip(devices,shield['executed_actions']))
    reward = reward_components(billing=billing,grid_peak_severity=grid_peak_severity,
        discomfort_units=physical['discomfort_units'],switching_events=switching,
        unmet_service_units=physical['unmet_service_units'],weights=weights)
    record = {
        'schema_version':'sharp_transition_core_v1','episode_id':episode_id,
        'step_id':step_id,'household_id':ledger.household_id,'interval_minutes':15,
        'state':deepcopy(state),'next_state':deepcopy(physical['next_state']),
        'device_state':[asdict(d) for d in devices],
        'next_device_state':[asdict(d) for d in next_devices],
        'shield':shield,'policy_action_before_human':policy_shield['executed_actions'],
        'feedback_evidence':feedback,'intervention_seen':seen,'response_window_complete':complete,
        'appliance_power_w':powers,'aggregate_power_w':demand_w,
        'grid_import_kwh':import_w*.25/1000,'grid_export_kwh':export_w*.25/1000,
        'actual_capacity_excess_w':actual_excess_w,
        'constraint_feasible':shield['capacity_feasible'] and actual_excess_w<=1e-9,
        'billing_before':before,'billing_after':trial.state(),'billing_components':billing,
        'reward':reward,'reward_weights':asdict(weights),
        'terminated':terminated,'truncated':truncated,'done':terminated or truncated,
        'policy_source':policy_source,'data_release_version':data_release_version,
        'simulator_version':'transition_core_v1','is_synthetic':True,
        'export_compensation_included':False,
    }
    # Entire record must serialize before committing the billing step.
    json.dumps(record,allow_nan=False)
    ledger.__dict__.update(trial.__dict__)
    return record


def self_test():
    from dataclasses import replace
    tariff = Tariff.load()
    ledger = BillingLedger(tariff,'fixture_house','fixture_period',2,
        opening_kwh=0,opening_charges_already_booked=False)
    ds = [Device('required',True,True,100,must_run=True),
          Device('flexible',True,True,1000,shed_priority=10)]
    def physics(action):
        times=advance_timers(ds,action)
        nxt=[replace(d,current_on=t['current_on'],elapsed_state_steps=t['elapsed_state_steps']) for d,t in zip(ds,times)]
        return {'next_devices':nxt,'appliance_power_w':[float(d.estimated_on_w if a else d.estimated_off_w) for d,a in zip(ds,action)],
                'next_state':{'fixture':1},'discomfort_units':0.,'unmet_service_units':0.}
    args=dict(ledger=ledger,weights=RewardWeights(1,1,1,1,1),episode_id='fixture',step_id=0,
        devices=ds,requested=[0,1],state={'fixture':0},physics_step=physics,
        max_import_w=500.,base_load_w=0.,available_solar_w=0.,grid_peak_severity=.5,
        policy_source='TEST_FIXTURE_NOT_TRAINING',data_release_version='TEST_ONLY')
    record=transition_step(**args)
    assert record['shield']['executed_actions']==[1,0]
    assert record['feedback_evidence'][1]=='CENSORED_NOT_SEEN'
    assert record['constraint_feasible'] and record['grid_import_kwh']==.025
    assert record['billing_after']['last_step']==0
    assert feedback_status(changed=True,seen=True,window_complete=False,override_requested=False)=='CENSORED_RESPONSE_WINDOW_OPEN'
    assert feedback_status(changed=True,seen=True,window_complete=True,override_requested=False)=='SEEN_NO_OVERRIDE_WEAK_EVIDENCE'
    original=ledger.state()
    bad=dict(args);bad['physics_step']=lambda a:{**physics(a),'discomfort_units':-1.}
    try: transition_step(**bad)
    except ValueError: pass
    else: raise AssertionError('Invalid discomfort accepted')
    assert ledger.state()==original, 'Failed transition changed billing'
    override=dict(args);override['human_actions']={'flexible':1};override['step_id']=1
    second=transition_step(**override)
    assert second['feedback_evidence'][1]=='EXPLICIT_OVERRIDE_REQUEST'
    assert not second['shield']['human_override_honored']['flexible']
    assert second['billing_components']['fixed_inr']=='0'
    excessive=dict(args);excessive['step_id']=2
    excessive['physics_step']=lambda a:{**physics(a),'appliance_power_w':[600.,0.]}
    third=transition_step(**excessive)
    assert not third['constraint_feasible'] and third['actual_capacity_excess_w']==100.
    terminal=dict(args);terminal['step_id']=3;terminal['truncated']=True
    fourth=transition_step(**terminal)
    assert fourth['done'] and not fourth['terminated'] and fourth['truncated']
    report={'status':'PASS','module':'integrated_transition_core_v1',
      'tested':['shield-to-dynamics execution','power-to-energy conversion',
                'billing-to-reward integration','unseen feedback censoring',
                'open response-window censoring','constrained human override',
                'failed-step billing rollback','actual versus estimated capacity',
                'time-limit versus terminal flags'],
      'test_data':'synthetic fixtures only; not training transitions',
      'household_appliance_models_integrated':False,
      'full_simulator_ready':False,'master_release_ready':False}
    p=Path(__file__).resolve().parents[1]/'reports/sharp_transition_core_validation_v1.json'
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('Report:',p)

if __name__=='__main__': self_test()
