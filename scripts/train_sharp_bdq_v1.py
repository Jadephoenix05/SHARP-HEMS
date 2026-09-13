"""Branching Dueling Double-Q training on the SHARP RL transition release.

Reads the flat release columns directly, so it runs on Kaggle against
rl_transitions.parquet without needing nested per-step records.

Honest scope. This is a working offline Double-Q training run with a held-out
validation split and train-only normalisation. It is NOT a claim of convergence,
of generalisation to real homes, or of deployable policy quality. Offline RL
without a behaviour-policy correction can overestimate out-of-distribution
actions, and no such correction is applied here.

Difference from the earlier smoke test: next-action selection masks padded
device branches but does NOT re-apply the joint safety shield, because the
release rows do not carry nested device state. Shield-consistent target
projection remains future work and is recorded in the limitations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

N = 28


class Network:
    def __init__(self, features, hidden=128, seed=42):
        rng = np.random.default_rng(seed)
        self.p = {'w': rng.normal(0, np.sqrt(2 / features), (features, hidden)),
                  'b': np.zeros(hidden),
                  'v': rng.normal(0, .01, (hidden, 1)), 'vb': np.zeros(1),
                  'a': rng.normal(0, .01, (hidden, N * 2)), 'ab': np.zeros(N * 2)}

    def forward(self, x):
        h = np.maximum(0, x @ self.p['w'] + self.p['b'])
        a = (h @ self.p['a'] + self.p['ab']).reshape(-1, N, 2)
        v = (h @ self.p['v'] + self.p['vb']).reshape(-1, 1, 1)
        return v + a - a.mean(2, keepdims=True), h

    def loss_grad(self, x, actions, target, present):
        q, h = self.forward(x)
        batch = len(x)
        chosen = np.take_along_axis(q, actions[:, :, None], axis=2)[:, :, 0]
        error = chosen - target[:, None]
        weight = present / (present.sum(1, keepdims=True) * batch)
        loss = float((np.where(abs(error) < 1, .5 * error ** 2, abs(error) - .5) * weight).sum())
        dq = np.zeros_like(q)
        np.put_along_axis(dq, actions[:, :, None],
                          (np.clip(error, -1, 1) * weight)[:, :, None], axis=2)
        da = (dq - dq.mean(2, keepdims=True)).reshape(batch, -1)
        dv = dq.sum((1, 2))[:, None]
        dh = (da @ self.p['a'].T + dv @ self.p['v'].T) * (h > 0)
        return loss, {'w': x.T @ dh, 'b': dh.sum(0), 'v': h.T @ dv, 'vb': dv.sum(0),
                      'a': h.T @ da, 'ab': da.sum(0)}

    def update(self, x, a, y, m, lr=.001):
        loss, g = self.loss_grad(x, a, y, m)
        norm = np.sqrt(sum(np.square(v).sum() for v in g.values()))
        if not np.isfinite(norm) or not np.isfinite(loss):
            raise ValueError('Nonfinite training update')
        for k, v in g.items():
            self.p[k] -= lr * v * min(1., 10 / (norm + 1e-12))
        return loss


def gradient_test():
    """Finite-difference check, and proof that padded branches never learn."""
    rng = np.random.default_rng(9)
    net = Network(5, hidden=8)
    x = rng.normal(size=(2, 5))
    a = np.zeros((2, N), int)
    y = np.array([.3, -.2])
    m = np.zeros((2, N))
    m[:, :2] = 1
    loss, g = net.loss_grad(x, a, y, m)
    key, index, eps = 'a', (0, 0), 1e-6
    original = net.p[key][index]
    net.p[key][index] = original + eps
    up = net.loss_grad(x, a, y, m)[0]
    net.p[key][index] = original - eps
    down = net.loss_grad(x, a, y, m)[0]
    net.p[key][index] = original
    assert np.isclose((up - down) / (2 * eps), g[key][index], atol=1e-6, rtol=1e-4)
    assert np.all(g['a'][:, 4:] == 0) and np.all(g['ab'][4:] == 0), \
        'Padded branches contributed to the loss'


def load(source, split):
    d = pd.read_parquet(source, filters=[('split', '==', split)])
    if d.empty:
        raise ValueError(f'No {split} rows')
    x = np.stack(d.state).astype(float)
    nx = np.stack(d.next_state).astype(float)
    present = np.stack(d.device_present).astype(float)
    actions = np.zeros((len(d), N), int)
    for i, a in enumerate(d.action.to_numpy()):
        a = np.asarray(a, int)
        actions[i, :len(a)] = a
    reward = d.reward.to_numpy(float)
    done = d.done.to_numpy(bool)
    for name, array in [('state', x), ('next_state', nx), ('reward', reward)]:
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite {name}')
    return x, nx, actions, present, reward, done


def fit(root, steps, batch, gamma, scale, seed):
    gradient_test()
    source = root / 'data/processed/sharp_rl_transitions_v1/rl_transitions.parquet'
    if not source.exists():
        raise FileNotFoundError(f'Missing {source}')

    x, nx, actions, present, reward, done = load(source, 'train')
    vx, vnx, vactions, vpresent, vreward, vdone = load(source, 'validation')

    # Normalisation is fitted on TRAIN ONLY and then applied to validation.
    mean = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0
    xn, nxn = (x - mean) / sd, (nx - mean) / sd
    vxn, vnxn = (vx - mean) / sd, (vnx - mean) / sd
    y = reward / scale
    vy = vreward / scale

    net = Network(x.shape[1], seed=seed)
    target = Network(x.shape[1], seed=seed)
    target.p = {k: v.copy() for k, v in net.p.items()}
    rng = np.random.default_rng(seed)
    losses, history = [], []

    for step in range(1, steps + 1):
        index = rng.integers(0, len(xn), batch)
        # Double Q: the online net selects, the target net evaluates.
        online_next, _ = net.forward(nxn[index])
        masked = np.where(present[index][:, :, None] > 0, online_next, -np.inf)
        best = np.argmax(masked, axis=2)
        target_next, _ = target.forward(nxn[index])
        chosen = np.take_along_axis(target_next, best[:, :, None], axis=2)[:, :, 0]
        per_branch = (chosen * present[index]).sum(1) / np.maximum(1, present[index].sum(1))
        bootstrap = np.where(done[index], 0.0, gamma * per_branch)
        losses.append(net.update(xn[index], actions[index], y[index] + bootstrap,
                                 present[index]))
        if step % 250 == 0:
            target.p = {k: v.copy() for k, v in net.p.items()}
        if step % 500 == 0 or step == steps:
            vq, _ = net.forward(vxn)
            vchosen = np.take_along_axis(vq, vactions[:, :, None], axis=2)[:, :, 0]
            vonline, _ = net.forward(vnxn)
            vmask = np.where(vpresent[:, :, None] > 0, vonline, -np.inf)
            vbest = np.argmax(vmask, axis=2)
            vtq, _ = target.forward(vnxn)
            vnext = np.take_along_axis(vtq, vbest[:, :, None], axis=2)[:, :, 0]
            vper = (vnext * vpresent).sum(1) / np.maximum(1, vpresent.sum(1))
            vtarget = vy + np.where(vdone, 0.0, gamma * vper)
            verror = (vchosen - vtarget[:, None]) * vpresent
            vtd = float(np.abs(verror).sum() / vpresent.sum())
            history.append({'step': step,
                            'train_loss_last_50': float(np.mean(losses[-50:])),
                            'validation_td_error': vtd})
            print(f"  step {step:5d} | train loss {history[-1]['train_loss_last_50']:.6f} "
                  f"| validation TD {vtd:.6f}", flush=True)

    out = root / 'models/sharp_bdq_v1'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'checkpoint.npz', mean=mean, sd=sd, **net.p)
    reloaded = np.load(out / 'checkpoint.npz')
    for k, v in net.p.items():
        if not np.array_equal(reloaded[k], v):
            raise ValueError('Checkpoint reload mismatch')

    report = {
        'status': 'BDQ_TRAINING_COMPLETED',
        'training_rows': int(len(xn)),
        'validation_rows': int(len(vxn)),
        'training_households': int(pd.read_parquet(
            source, columns=['split', 'household_id'],
            filters=[('split', '==', 'train')]).household_id.nunique()),
        'feature_count': int(x.shape[1]),
        'maximum_device_branches': N,
        'updates': steps,
        'batch_size': batch,
        'gamma': gamma,
        'reward_scale_divisor': scale,
        'final_train_loss_last_50': history[-1]['train_loss_last_50'],
        'final_validation_td_error': history[-1]['validation_td_error'],
        'history': history,
        'normalization_fit': 'TRAIN_ONLY',
        'test_used': False,
        'finite_difference_gradient_test': 'PASS',
        'padded_branch_gradient_test': 'PASS',
        'checkpoint_reload': 'PASS',
        'architecture_reference': 'https://arxiv.org/abs/1711.08946',
        'training_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'limitations': [
            'Offline Double-Q without any behaviour-policy or distribution-shift correction.',
            'Next-action selection masks padded branches but does not re-apply the joint safety shield.',
            'A falling TD error is not evidence of real-world policy quality.',
            'The test split has deliberately not been touched.',
            'Underlying appliance power values are proxies, and thermal parameters are declared assumptions.',
            'No human attention or learned preference model is present.',
        ],
        'approved_for_deployment': False,
    }
    (out / 'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nBDQ TRAINING COMPLETED')
    for key in ['training_rows', 'validation_rows', 'training_households',
                'feature_count', 'updates', 'final_train_loss_last_50',
                'final_validation_td_error']:
        print(f'  {key}: {report[key]}')
    print('  test_used:', report['test_used'])
    print('Output:', out)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--reward-scale', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    fit(a.root.resolve(), a.steps, a.batch, a.gamma, a.reward_scale, a.seed)
