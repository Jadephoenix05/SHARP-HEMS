"""NumPy branching-dueling Double-Q compatibility test on SHARP pilot train rows.
Not a final offline-RL algorithm, policy evaluation or convergence claim.
Architecture reference: https://arxiv.org/abs/1711.08946
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sharp_action_shield import Device, local_mask, apply_shield

N=28

class Network:
    def __init__(self,features,hidden=64,seed=42):
        rng=np.random.default_rng(seed)
        self.p={'w':rng.normal(0,np.sqrt(2/features),(features,hidden)),
                'b':np.zeros(hidden),'v':rng.normal(0,.01,(hidden,1)),'vb':np.zeros(1),
                'a':rng.normal(0,.01,(hidden,N*2)),'ab':np.zeros(N*2)}
    def forward(self,x):
        h=np.maximum(0,x@self.p['w']+self.p['b'])
        a=(h@self.p['a']+self.p['ab']).reshape(-1,N,2)
        v=(h@self.p['v']+self.p['vb']).reshape(-1,1,1)
        return v+a-a.mean(2,keepdims=True),h
    def loss_grad(self,x,actions,target,present):
        q,h=self.forward(x);batch=len(x)
        chosen=np.take_along_axis(q,actions[:,:,None],axis=2)[:,:,0]
        error=chosen-target[:,None]
        weight=present/(present.sum(1,keepdims=True)*batch)
        loss=float((np.where(abs(error)<1,.5*error**2,abs(error)-.5)*weight).sum())
        dq=np.zeros_like(q)
        np.put_along_axis(dq,actions[:,:,None],(np.clip(error,-1,1)*weight)[:,:,None],axis=2)
        da=(dq-dq.mean(2,keepdims=True)).reshape(batch,-1);dv=dq.sum((1,2))[:,None]
        dh=(da@self.p['a'].T+dv@self.p['v'].T)*(h>0)
        return loss,{'w':x.T@dh,'b':dh.sum(0),'v':h.T@dv,'vb':dv.sum(0),'a':h.T@da,'ab':da.sum(0)}
    def update(self,x,a,y,m,lr=.001):
        loss,g=self.loss_grad(x,a,y,m)
        norm=np.sqrt(sum(np.square(v).sum() for v in g.values()))
        if not np.isfinite(norm) or not np.isfinite(loss):raise ValueError('Nonfinite training update')
        for k,v in g.items():self.p[k]-=lr*v*min(1.,10/(norm+1e-12))
        return loss


def gradient_test():
    rng=np.random.default_rng(9);net=Network(5,hidden=8)
    x=rng.normal(size=(2,5));a=np.zeros((2,N),int);y=np.array([.3,-.2])
    m=np.zeros((2,N));m[:,:2]=1
    loss,g=net.loss_grad(x,a,y,m)
    k='a';ix=(0,0);orig=net.p[k][ix];eps=1e-6
    net.p[k][ix]=orig+eps;up=net.loss_grad(x,a,y,m)[0]
    net.p[k][ix]=orig-eps;down=net.loss_grad(x,a,y,m)[0];net.p[k][ix]=orig
    assert np.isclose((up-down)/(2*eps),g[k][ix],atol=1e-6,rtol=1e-4)
    assert np.all(g['a'][:,4:]==0) and np.all(g['ab'][4:]==0),'Padded branches contributed to loss'


def fit(root,steps):
    if steps<1:raise ValueError('Positive update count required')
    gradient_test()
    source=root/'data/processed/rl_integration_pilot_v1/pilot_transitions.parquet'
    # Filter before decoding observations or fitting normalization.
    d=pd.read_parquet(source,filters=[('split','==','train')])
    if d.empty or not d.split.eq('train').all():raise ValueError('Expected train-only data')
    if not d.constraint_feasible.all():raise ValueError('Resolve infeasible training transitions before this smoke test')
    records=[json.loads(s) for s in d.transition_json]
    x=np.stack(d.state).astype(float);nx=np.stack(d.next_state).astype(float)
    rewards=d.reward.to_numpy(float)/10;terminal=d.terminated.to_numpy(bool)
    if not np.isfinite(x).all() or not np.isfinite(nx).all() or not np.isfinite(rewards).all():raise ValueError('Nonfinite training data')
    actions=np.zeros((len(d),N),int);present=np.zeros((len(d),N));next_devices=[];limits=[]
    for i,r in enumerate(records):
        if r['household_id']!=r['billing_before']['household_id']:raise ValueError('Household mismatch')
        if r['shield']['human_requested_actions']:raise ValueError('This test only supports pilot rows without human overrides')
        if r['state']['features']!=list(d.state.iloc[i]):raise ValueError('Flat/nested state mismatch')
        devices=[Device(**v) for v in r['device_state']]
        nxt=[Device(**v) for v in r['next_device_state']]
        n=len(devices);limit=float(r['state']['features'][8])*1000
        if not 0<n<=N:raise ValueError('Invalid device count')
        # Executed actions can label this pilot because replaying them through
        # its shield leaves them unchanged; no override or proposal penalty.
        executed=r['shield']['executed_actions']
        replay=apply_shield(devices,executed,max_import_w=limit)
        if replay['executed_actions']!=executed or not replay['capacity_feasible']:
            raise ValueError('Executed-action replay is not equivalent')
        actions[i,:n]=executed;present[i,:n]=1
        next_devices.append(nxt);limits.append(limit)
    mean=x.mean(0);std=x.std(0);std[std<1e-6]=1
    x=np.clip((x-mean)/std,-10,10);nx=np.clip((nx-mean)/std,-10,10)
    online=Network(x.shape[1]);target=Network(x.shape[1]);target.p={k:v.copy() for k,v in online.p.items()}
    rng=np.random.default_rng(42);losses=[]
    for step in range(steps):
        idx=rng.integers(0,len(d),size=min(64,len(d)))
        q,_=online.forward(nx[idx]);tq,_=target.forward(nx[idx]);bootstrap=np.zeros(len(idx))
        for j,i in enumerate(idx):
            if terminal[i]:continue
            ds=next_devices[i];mask=np.array([local_mask(v)[0] for v in ds],bool)
            proposed=np.where(mask,q[j,:len(ds)],-np.inf).argmax(1).tolist()
            # Joint capacity is not represented by independent branch masks.
            projection=apply_shield(ds,proposed,max_import_w=limits[i])
            if not projection['capacity_feasible']:raise ValueError('Infeasible next-state target')
            legal=projection['executed_actions']
            bootstrap[j]=np.mean(tq[j,np.arange(len(ds)),legal])
        y=rewards[idx]+.99*(~terminal[idx])*bootstrap
        loss=online.update(x[idx],actions[idx],y,present[idx]);losses.append(loss)
        if (step+1)%50==0:
            target.p={k:v.copy() for k,v in online.p.items()}
            print(f'Update {step+1}/{steps} | Huber loss {loss:.6f}',flush=True)
    out=root/'models/sharp_bdq_smoke_v1';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'checkpoint.npz',**online.p,normalization_mean=mean,normalization_std=std)
    saved=np.load(out/'checkpoint.npz',allow_pickle=False);restored=Network(x.shape[1])
    restored.p={k:saved[k] for k in online.p}
    assert np.array_equal(online.forward(x[:8])[0],restored.forward(x[:8])[0])
    report={'status':'TRAINING_COMPATIBILITY_PASS','training_rows':len(d),
        'training_households':len({r['household_id'] for r in records}),
        'feature_count':x.shape[1],'maximum_device_branches':N,'updates':steps,
        'last_batch_huber_loss':losses[-1],'mean_last_20_batch_loss':float(np.mean(losses[-20:])),
        'normalization_fit':'TRAIN_ONLY','validation_used':False,'test_used':False,
        'padded_branch_gradient_test':'PASS','finite_difference_gradient_test':'PASS',
        'checkpoint_reload':'PASS','reward_scale_divisor':10,'gamma':.99,
        'action_semantics':'EXECUTED_ACTION_WITH_SHIELD_REPLAY_EQUIVALENCE_CHECK',
        'next_action_selection':'LOCAL_MASKS_THEN_JOINT_SHIELD_PROJECTION',
        'training_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'architecture_reference':'https://arxiv.org/abs/1711.08946',
        'limitations':['Offline smoke test, not evidence of convergence or generalisation.',
            'Pilot has one training household; heldout behaviour is not evaluated.',
            'No prioritised replay or offline extrapolation correction.',
            'Device slot ordering is household-specific; appliance-type-aware shared modelling still needs design.',
            'Underlying appliance dynamics and human preference models are incomplete.'],
        'full_simulator_ready':False,'master_release_ready':False}
    (out/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nSHARP BDQ COMPATIBILITY TEST');print(json.dumps(report,indent=2));print('\nOutput:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--steps',type=int,default=100)
    args=p.parse_args();fit(args.root.resolve(),args.steps)
