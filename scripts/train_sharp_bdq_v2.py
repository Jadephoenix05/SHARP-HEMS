"""Conservative Branching Dueling Double-Q training on the SHARP release.

What this adds over train_sharp_bdq_v1.py, in the order the RL spec asks for.

1. Terminal and non-terminal TD error are reported SEPARATELY. The unmet-service
   penalty lands as one lump at step 95, so terminal rewards are roughly 77 times
   the size of step rewards. A single pooled TD number is dominated by the 1 per
   cent of rows that are terminal and flatters the policy.

2. Conservative Q-Learning. Offline Q-learning overestimates actions that rarely
   appear in the batch, the max in the Bellman target picks the most overestimated
   action rather than the best one, and with no environment to explore the error
   compounds. CQL adds a term that pushes Q down across levels and back up on the
   level actually logged.

3. A behaviour-cloning warm start. The CQL term is exactly the cross-entropy of
   the logged action under a softmax over that branch's Q values, so a warm start
   is the same loss with the temporal-difference term switched off. No second
   objective and no second head.

Honest scope. This is offline training measured by held-out TD error and by
agreement with the logged policy. Neither is evidence of real-world control
quality, and no result here is approved for deployment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

N = 28
LEVELS = 3   # 0 off, 1 on, 2 reduced


class Network:
    def __init__(self, features, hidden=128, seed=42):
        rng = np.random.default_rng(seed)
        self.p = {'w': rng.normal(0, np.sqrt(2 / features), (features, hidden)),
                  'b': np.zeros(hidden),
                  'v': rng.normal(0, .01, (hidden, 1)), 'vb': np.zeros(1),
                  'a': rng.normal(0, .01, (hidden, N * LEVELS)), 'ab': np.zeros(N * LEVELS)}

    def forward(self, x):
        h = np.maximum(0, x @ self.p['w'] + self.p['b'])
        a = (h @ self.p['a'] + self.p['ab']).reshape(-1, N, LEVELS)
        v = (h @ self.p['v'] + self.p['vb']).reshape(-1, 1, 1)
        return v + a - a.mean(2, keepdims=True), h

    def backward(self, x, h, dq):
        """Chain one gradient on Q back through the duelling heads to the trunk."""
        batch = len(x)
        da = (dq - dq.mean(2, keepdims=True)).reshape(batch, -1)
        dv = dq.sum((1, 2))[:, None]
        dh = (da @ self.p['a'].T + dv @ self.p['v'].T) * (h > 0)
        return {'w': x.T @ dh, 'b': dh.sum(0), 'v': h.T @ dv, 'vb': dv.sum(0),
                'a': h.T @ da, 'ab': da.sum(0)}

    def loss_grad(self, x, actions, target, present, alpha, use_td=True,
                  level_weight=None):
        q, h = self.forward(x)
        weight = present / (present.sum(1, keepdims=True) * len(x))
        dq = np.zeros_like(q)
        loss = 0.0
        td_loss = 0.0

        if use_td:
            chosen = np.take_along_axis(q, actions[:, :, None], axis=2)[:, :, 0]
            error = chosen - target[:, None]
            td_loss = float((np.where(abs(error) < 1, .5 * error ** 2,
                                      abs(error) - .5) * weight).sum())
            loss += td_loss
            np.put_along_axis(dq, actions[:, :, None],
                              (np.clip(error, -1, 1) * weight)[:, :, None], axis=2)

        # CQL / behaviour cloning. The log-sum-exp over levels minus the logged
        # level is the negative log probability of the logged action, so a single
        # term serves as the conservative penalty and as the cloning objective.
        cql_loss = 0.0
        if alpha > 0:
            shift = q.max(2, keepdims=True)
            exp = np.exp(q - shift)
            total = exp.sum(2)
            softmax = exp / total[:, :, None]
            logsumexp = shift[:, :, 0] + np.log(total)
            logged = np.take_along_axis(q, actions[:, :, None], axis=2)[:, :, 0]
            # Optional inverse-frequency weighting. Without it the term is a
            # class-imbalanced cross-entropy and collapses to OFF, which scores
            # well on raw agreement and is worthless as a controller.
            cql_weight = weight if level_weight is None else weight * level_weight[actions]
            cql_loss = float(((logsumexp - logged) * cql_weight).sum())
            loss += alpha * cql_loss
            grad = softmax.copy()
            np.put_along_axis(
                grad, actions[:, :, None],
                np.take_along_axis(grad, actions[:, :, None], axis=2) - 1.0, axis=2)
            dq += alpha * grad * cql_weight[:, :, None]

        return loss, td_loss, cql_loss, self.backward(x, h, dq)

    def update(self, x, a, y, m, alpha, lr=.001, use_td=True, level_weight=None):
        loss, td, cql, g = self.loss_grad(x, a, y, m, alpha, use_td, level_weight)
        norm = np.sqrt(sum(np.square(v).sum() for v in g.values()))
        if not np.isfinite(norm) or not np.isfinite(loss):
            raise ValueError('Nonfinite training update')
        for k, v in g.items():
            self.p[k] -= lr * v * min(1., 10 / (norm + 1e-12))
        return loss, td, cql


def gradient_test():
    """Finite differences on both loss terms, and proof padded branches never learn."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=(3, 5))
    a = rng.integers(0, LEVELS, (3, N))
    y = np.array([.3, -.2, .7])
    m = np.zeros((3, N))
    m[:, :2] = 1
    balance = np.array([0.4, 1.1, 1.5])
    for alpha, use_td, lw in [(0.0, True, None), (1.0, False, None),
                              (0.5, True, None), (0.5, True, balance),
                              (1.0, False, balance)]:
        net = Network(5, hidden=8)
        _, _, _, g = net.loss_grad(x, a, y, m, alpha, use_td, lw)
        for key, index in [('a', (0, 0)), ('w', (1, 3)), ('v', (2, 0))]:
            original = net.p[key][index]
            eps = 1e-6
            net.p[key][index] = original + eps
            up = net.loss_grad(x, a, y, m, alpha, use_td, lw)[0]
            net.p[key][index] = original - eps
            down = net.loss_grad(x, a, y, m, alpha, use_td, lw)[0]
            net.p[key][index] = original
            numeric = (up - down) / (2 * eps)
            if not np.isclose(numeric, g[key][index], atol=1e-6, rtol=1e-4):
                raise AssertionError(
                    f'Gradient mismatch alpha={alpha} td={use_td} balanced={lw is not None} at {key}{index}: '
                    f'finite difference {numeric} vs analytic {g[key][index]}')
        if not (np.all(g['a'][:, 2 * LEVELS:] == 0) and np.all(g['ab'][2 * LEVELS:] == 0)):
            raise AssertionError('Padded branches contributed to the loss')


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
        if a.size and (a.min() < 0 or a.max() >= LEVELS):
            raise ValueError(f'Action level outside 0..{LEVELS - 1}')
        actions[i, :len(a)] = a
    reward = d.reward.to_numpy(float)
    done = d.done.to_numpy(bool)
    for name, array in [('state', x), ('next_state', nx), ('reward', reward)]:
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite {name}')
    return x, nx, actions, present, reward, done


def bootstrap_target(net, target, nxn, present, reward, done, gamma, index):
    """Double Q: the online net selects the next level, the target net prices it."""
    online_next, _ = net.forward(nxn[index])
    masked = np.where(present[index][:, :, None] > 0, online_next, -np.inf)
    best = np.argmax(masked, axis=2)
    target_next, _ = target.forward(nxn[index])
    chosen = np.take_along_axis(target_next, best[:, :, None], axis=2)[:, :, 0]
    per_branch = (chosen * present[index]).sum(1) / np.maximum(1, present[index].sum(1))
    return reward[index] + np.where(done[index], 0.0, gamma * per_branch)


def evaluate(net, target, x, nx, actions, present, reward, done, gamma, chunk=20000):
    """Held-out TD error, terminal split from non-terminal, plus policy agreement."""
    absolute = np.zeros(len(x))
    counted = np.zeros(len(x))
    agree = np.zeros(LEVELS)
    logged_count = np.zeros(LEVELS)
    greedy_count = np.zeros(LEVELS)
    for start in range(0, len(x), chunk):
        index = np.arange(start, min(start + chunk, len(x)))
        q, _ = net.forward(x[index])
        chosen = np.take_along_axis(q, actions[index][:, :, None], axis=2)[:, :, 0]
        y = bootstrap_target(net, target, nx, present, reward, done, gamma, index)
        error = np.abs(chosen - y[:, None]) * present[index]
        absolute[index] = error.sum(1)
        counted[index] = present[index].sum(1)
        greedy = np.argmax(np.where(present[index][:, :, None] > 0, q, -np.inf), axis=2)
        live = present[index] > 0
        for level in range(LEVELS):
            picked = live & (actions[index] == level)
            logged_count[level] += picked.sum()
            agree[level] += (picked & (greedy == level)).sum()
            greedy_count[level] += (live & (greedy == level)).sum()

    def td(rows):
        return float(absolute[rows].sum() / max(1.0, counted[rows].sum()))

    total = max(1.0, logged_count.sum())
    recall = agree / np.maximum(1.0, logged_count)
    # Balanced accuracy is the headline, not raw agreement. 83 per cent of logged
    # actions are OFF, so a model that answers OFF unconditionally scores 0.83 on
    # raw agreement while having learned nothing about when to switch. The mean of
    # the per-level recalls cannot be fooled that way: it is 1/3 for that model.
    return {
        'balanced_accuracy': float(recall.mean()),
        'td_error_all': td(slice(None)),
        'td_error_non_terminal': td(~done),
        'td_error_terminal': td(done),
        'terminal_row_share': float(done.mean()),
        'agreement_with_logged_action': float(agree.sum() / total),
        'agreement_by_level': {str(i): float(agree[i] / max(1.0, logged_count[i]))
                               for i in range(LEVELS)},
        'logged_level_share': {str(i): float(logged_count[i] / total)
                               for i in range(LEVELS)},
        'greedy_level_share': {str(i): float(greedy_count[i] / total)
                               for i in range(LEVELS)},
    }


def fit(root, steps, warm_start, batch, gamma, scale, alpha, lr, seed, hidden,
        level_balance, tag):
    gradient_test()
    source = root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet'
    if not source.exists():
        raise FileNotFoundError(f'Missing {source}')

    x, nx, actions, present, reward, done = load(source, 'train')
    vx, vnx, vactions, vpresent, vreward, vdone = load(source, 'validation')

    # Normalisation is fitted on TRAIN ONLY and then applied to validation.
    features = x.shape[1]
    mean = x.mean(0)
    sd = x.std(0)
    zero_variance = int((sd < 1e-8).sum())
    sd[sd < 1e-8] = 1.0
    # float32 throughout: the training matrices are 279k x 305 and a float64
    # copy of state, next state and both normalised versions needs ~2.7 GB.
    xn = ((x - mean) / sd).astype(np.float32)
    nxn = ((nx - mean) / sd).astype(np.float32)
    vxn = ((vx - mean) / sd).astype(np.float32)
    vnxn = ((vnx - mean) / sd).astype(np.float32)
    del x, nx, vx, vnx
    y = reward / scale
    vy = vreward / scale

    # Inverse-frequency level weights, counted on TRAIN only and over live
    # branches only, so padded slots do not vote.
    live = present > 0
    counts = np.array([float(((actions == level) & live).sum()) for level in range(LEVELS)])
    share = counts / counts.sum()
    level_weight = None
    if level_balance:
        level_weight = (1.0 / np.maximum(share, 1e-9))
        level_weight = level_weight / level_weight.mean()

    net = Network(features, hidden=hidden, seed=seed)
    target = Network(features, hidden=hidden, seed=seed)
    target.p = {k: v.copy() for k, v in net.p.items()}
    rng = np.random.default_rng(seed)
    history = []
    recent = []

    def report(step, phase):
        metrics = evaluate(net, target, vxn, vnxn, vactions, vpresent, vy, vdone, gamma)
        metrics.update({'step': step, 'phase': phase,
                        'train_loss_last_50': float(np.mean(recent[-50:]))})
        history.append(metrics)
        print(f"  {phase:12s} step {step:5d} | train {metrics['train_loss_last_50']:9.6f} "
              f"| val TD step {metrics['td_error_non_terminal']:8.5f} "
              f"terminal {metrics['td_error_terminal']:9.5f} "
              f"| balanced {metrics['balanced_accuracy']:.4f} "
              f"agree {metrics['agreement_with_logged_action']:.4f}", flush=True)

    # Phase 1: behaviour cloning. No bootstrapping, so the network starts from a
    # policy that at least reproduces what the logged controllers actually did.
    if warm_start:
        print(f'Behaviour-cloning warm start: {warm_start} updates')
        for step in range(1, warm_start + 1):
            index = rng.integers(0, len(xn), batch)
            loss, _, _ = net.update(xn[index], actions[index], y[index], present[index],
                                    alpha=1.0, lr=lr, use_td=False,
                                    level_weight=level_weight)
            recent.append(loss)
            if step % 500 == 0 or step == warm_start:
                target.p = {k: v.copy() for k, v in net.p.items()}
                report(step, 'warm_start')
        target.p = {k: v.copy() for k, v in net.p.items()}

    # Phase 2: conservative Double-Q.
    print(f'Conservative Double-Q: {steps} updates, CQL alpha {alpha}')
    for step in range(1, steps + 1):
        index = rng.integers(0, len(xn), batch)
        bootstrap = bootstrap_target(net, target, nxn, present, y, done, gamma, index)
        loss, _, _ = net.update(xn[index], actions[index], bootstrap, present[index],
                                alpha=alpha, lr=lr, use_td=True,
                                level_weight=level_weight)
        recent.append(loss)
        if step % 250 == 0:
            target.p = {k: v.copy() for k, v in net.p.items()}
        if step % 500 == 0 or step == steps:
            report(step, 'conservative')

    final = history[-1]
    out = root / f'models/sharp_bdq_v2_{tag}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'checkpoint.npz', mean=mean, sd=sd,
             n_features=np.array(features), n_branches=np.array(N),
             n_levels=np.array(LEVELS), **net.p)
    reloaded = np.load(out / 'checkpoint.npz')
    for k, v in net.p.items():
        if not np.array_equal(reloaded[k], v):
            raise ValueError('Checkpoint reload mismatch')

    report_json = {
        'status': 'CONSERVATIVE_BDQ_TRAINING_COMPLETED',
        'training_rows': int(len(xn)),
        'validation_rows': int(len(vxn)),
        'feature_count': int(features),
        'zero_variance_features': zero_variance,
        'maximum_device_branches': N,
        'action_levels_per_branch': LEVELS,
        'action_level_meaning': {'0': 'off', '1': 'on', '2': 'reduced'},
        'hidden_units': hidden,
        'parameter_count': int(sum(v.size for v in net.p.values())),
        'warm_start_updates': warm_start,
        'conservative_updates': steps,
        'cql_alpha': alpha,
        'cql_level_balanced': bool(level_balance),
        'train_logged_level_share': {str(i): float(share[i]) for i in range(LEVELS)},
        'cql_level_weight': (None if level_weight is None else
                             {str(i): float(level_weight[i]) for i in range(LEVELS)}),
        'batch_size': batch,
        'learning_rate': lr,
        'gamma': gamma,
        'reward_scale_divisor': scale,
        'final': {k: final[k] for k in
                  ['balanced_accuracy', 'td_error_all', 'td_error_non_terminal',
                   'td_error_terminal',
                   'terminal_row_share', 'agreement_with_logged_action',
                   'agreement_by_level', 'logged_level_share', 'greedy_level_share']},
        'history': history,
        'normalization_fit': 'TRAIN_ONLY',
        'test_used': False,
        'finite_difference_gradient_test': 'PASS',
        'padded_branch_gradient_test': 'PASS',
        'checkpoint_reload': 'PASS',
        'architecture_reference': 'https://arxiv.org/abs/1711.08946',
        'cql_reference': 'https://arxiv.org/abs/2006.04779',
        'training_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'limitations': [
            'CQL is applied over all three levels because the flat release does '
            'not carry per-device supports_reduced, so the penalty also covers '
            'levels the shield would refuse anyway.',
            'Next-action selection masks padded branches but does not re-apply '
            'the joint safety shield.',
            'Agreement with the logged action measures imitation of three '
            'scripted behaviour policies, not control quality. Read balanced '
            'accuracy instead: 83 per cent of logged actions are OFF, so raw '
            'agreement near 0.83 means the model answers OFF unconditionally.',
            'A falling TD error is not evidence of real-world policy quality.',
            'The test split has deliberately not been touched.',
            'Appliance power values are proxies and thermal parameters are '
            'declared assumptions.',
            'The reward is the hand-weighted billing reward; the learned '
            'preference term is fitted separately and is not applied here.',
        ],
        'approved_for_deployment': False,
    }
    (out / 'training_report.json').write_text(json.dumps(report_json, indent=2),
                                              encoding='utf-8')
    print('\nCONSERVATIVE BDQ TRAINING COMPLETED')
    print(f"  parameters              {report_json['parameter_count']}")
    print(f"  validation TD step      {final['td_error_non_terminal']:.6f}")
    print(f"  validation TD terminal  {final['td_error_terminal']:.6f}")
    print(f"  balanced accuracy       {final['balanced_accuracy']:.4f}   "
          f"(1/3 = no discrimination)")
    print(f"  raw agreement           {final['agreement_with_logged_action']:.4f}   "
          f"(majority class is {max(final['logged_level_share'].values()):.4f})")
    for level, name in enumerate(['off', 'on', 'reduced']):
        print(f"    level {level} {name:8s} logged "
              f"{final['logged_level_share'][str(level)]:.4f}  "
              f"greedy {final['greedy_level_share'][str(level)]:.4f}  "
              f"recall {final['agreement_by_level'][str(level)]:.4f}")
    print('  test_used:', report_json['test_used'])
    print('Output:', out)
    return report_json


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--warm-start', type=int, default=1000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--reward-scale', type=float, default=10.0)
    p.add_argument('--alpha', type=float, default=0.5, help='CQL weight')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--level-balance', action='store_true',
                   help='inverse-frequency weighting on the CQL / cloning term')
    p.add_argument('--tag', default=None, help='output directory suffix')
    a = p.parse_args()
    tag = a.tag or (f'cql{a.alpha}' + ('_balanced' if a.level_balance else '')
                    + ('_bc' if a.warm_start else ''))
    fit(a.root.resolve(), a.steps, a.warm_start, a.batch, a.gamma, a.reward_scale,
        a.alpha, a.lr, a.seed, a.hidden, a.level_balance, tag)
