"""Conservative BDQ, version 3. Four changes, each measured on its own.

1. ADAM instead of plain SGD. v2 used raw gradient descent with norm clipping at
   a fixed 1e-3. That is the single largest unforced loss in the whole pipeline:
   the 305 input features have wildly different scales and sparsity, and one
   global step size serves none of them well. Adam gives each parameter its own
   effective rate.

2. THE LEGALITY MASK, ported from the Kaggle notebook. Level 2 is meaningless on
   an appliance that cannot dim, and roughly 40 per cent cannot. Without the mask
   the argmax wastes probability mass on impossible actions and the conservative
   penalty pushes down levels that were never available. In the notebook this
   alone lifted dim recall from 0.51 to 0.81. The local trainer never had it,
   which is why the two disagreed.

3. POLICY-FILTERED IMITATION, available and OFF BY DEFAULT because measuring it
   showed it hurts badly. Recorded here because the reasoning was plausible and
   the refutation is the useful part.

   The idea was: random_binary samples its action from a draw that is not in the
   state, so the imitation term is asking the network to predict noise on a third
   of every batch, and fitting noise blurs whatever was learnable.

   The measurement says otherwise. Filtering it out drops balanced accuracy from
   0.7389 to 0.6155, and dim recall from 0.647 to 0.192. The reason is that
   random_binary supplies 98.3 per cent of every level-2 example in the training
   set - serve_preferred and peak_aware between them contribute 780 out of 45,556.
   The sampling policy is not noise to be filtered out; it is the only place the
   model ever sees an appliance dimmed.

   The general lesson: before excluding a data source for being noisy, check what
   it is the only source of.

4. A WIDER TRUNK is now an option. v2 fixed hidden at 128 on the reasoning that
   larger would overfit 279k rows. Balanced accuracy was still climbing at 15,000
   updates, which is the signature of underfitting, not overfitting.

Honest scope. All of this still optimises agreement with scripted controllers and
held-out TD error. Neither is control quality, and no result here is approved for
deployment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

N = 28
LEVELS = 3
LEVEL_NAMES = ['off', 'on', 'reduced']

SCENARIOS = {
    'main': 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet',
    'pv': 'data/processed/sharp_rl_scenario_pv_v1/rl_transitions.parquet',
}
# Policies whose action is a function of the state, so cloning them is learnable.
DETERMINISTIC_POLICIES = ('serve_preferred', 'peak_aware')


def check(condition, message):
    if not condition:
        raise ValueError(message)


class Network:
    def __init__(self, features, hidden=128, seed=42):
        rng = np.random.default_rng(seed)
        self.p = {'w': rng.normal(0, np.sqrt(2 / features), (features, hidden)),
                  'b': np.zeros(hidden),
                  'v': rng.normal(0, .01, (hidden, 1)), 'vb': np.zeros(1),
                  'a': rng.normal(0, .01, (hidden, N * LEVELS)), 'ab': np.zeros(N * LEVELS)}
        # Adam state, one pair of moments per parameter tensor.
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.vv = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        h = np.maximum(0, x @ self.p['w'] + self.p['b'])
        a = (h @ self.p['a'] + self.p['ab']).reshape(-1, N, LEVELS)
        v = (h @ self.p['v'] + self.p['vb']).reshape(-1, 1, 1)
        return v + a - a.mean(2, keepdims=True), h

    def backward(self, x, h, dq):
        batch = len(x)
        da = (dq - dq.mean(2, keepdims=True)).reshape(batch, -1)
        dv = dq.sum((1, 2))[:, None]
        dh = (da @ self.p['a'].T + dv @ self.p['v'].T) * (h > 0)
        return {'w': x.T @ dh, 'b': dh.sum(0), 'v': h.T @ dv, 'vb': dv.sum(0),
                'a': h.T @ da, 'ab': da.sum(0)}

    def loss_grad(self, x, actions, target, present, legal, alpha,
                  use_td=True, level_weight=None, imitate=None):
        q, h = self.forward(x)
        weight = present / (present.sum(1, keepdims=True) * len(x))
        dq = np.zeros_like(q)
        loss = 0.0

        if use_td:
            chosen = np.take_along_axis(q, actions[:, :, None], axis=2)[:, :, 0]
            error = chosen - target[:, None]
            loss += float((np.where(abs(error) < 1, .5 * error ** 2,
                                    abs(error) - .5) * weight).sum())
            np.put_along_axis(dq, actions[:, :, None],
                              (np.clip(error, -1, 1) * weight)[:, :, None], axis=2)

        if alpha > 0:
            # Log-sum-exp over LEGAL levels only.
            masked = np.where(legal, q, -np.inf)
            shift = masked.max(2, keepdims=True)
            exponent = np.where(legal, np.exp(masked - shift), 0.0)
            total = exponent.sum(2)
            softmax = exponent / total[:, :, None]
            logsumexp = shift[:, :, 0] + np.log(total)
            logged = np.take_along_axis(q, actions[:, :, None], axis=2)[:, :, 0]

            cql_weight = weight if level_weight is None else weight * level_weight[actions]
            if imitate is not None:
                # Rows from a sampling policy keep their temporal-difference
                # contribution above but contribute nothing here: their action
                # is not a function of the state, so there is nothing to clone.
                cql_weight = cql_weight * imitate[:, None]
            loss += alpha * float(((logsumexp - logged) * cql_weight).sum())
            gradient = softmax.copy()
            np.put_along_axis(
                gradient, actions[:, :, None],
                np.take_along_axis(gradient, actions[:, :, None], axis=2) - 1.0, axis=2)
            dq += alpha * gradient * cql_weight[:, :, None]

        return loss, self.backward(x, h, dq)

    def update(self, x, a, y, m, legal, alpha, lr=1e-3, use_td=True,
               level_weight=None, imitate=None, beta1=0.9, beta2=0.999, eps=1e-8):
        loss, g = self.loss_grad(x, a, y, m, legal, alpha, use_td,
                                 level_weight, imitate)
        norm = np.sqrt(sum(np.square(v).sum() for v in g.values()))
        if not np.isfinite(norm) or not np.isfinite(loss):
            raise ValueError('Nonfinite training update')
        scale = min(1.0, 10.0 / (norm + 1e-12))
        self.t += 1
        for k, grad in g.items():
            grad = grad * scale
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * grad
            self.vv[k] = beta2 * self.vv[k] + (1 - beta2) * np.square(grad)
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.vv[k] / (1 - beta2 ** self.t)
            self.p[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return loss


def gradient_test():
    """Finite differences on every loss path, plus the padded-branch proof."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4, 5))
    a = rng.integers(0, LEVELS, (4, N))
    y = np.array([.3, -.2, .7, .1])
    m = np.zeros((4, N))
    m[:, :2] = 1
    legal = rng.random((4, N, LEVELS)) > 0.3
    legal[:, :, 1] = True
    np.put_along_axis(legal, a[:, :, None], True, axis=2)
    balance = np.array([0.4, 1.1, 1.5])
    imitate = np.array([1.0, 0.0, 1.0, 0.0])

    settings = [(0.0, True, None, None), (1.0, False, None, None),
                (0.5, True, balance, None), (1.0, False, balance, imitate),
                (0.5, True, balance, imitate)]
    for alpha, use_td, lw, im in settings:
        net = Network(5, hidden=8)
        _, g = net.loss_grad(x, a, y, m, legal, alpha, use_td, lw, im)
        for key, index in [('a', (0, 0)), ('w', (1, 3)), ('v', (2, 0)), ('ab', (5,))]:
            original = net.p[key][index]
            eps = 1e-6
            net.p[key][index] = original + eps
            up = net.loss_grad(x, a, y, m, legal, alpha, use_td, lw, im)[0]
            net.p[key][index] = original - eps
            down = net.loss_grad(x, a, y, m, legal, alpha, use_td, lw, im)[0]
            net.p[key][index] = original
            numeric = (up - down) / (2 * eps)
            check(np.isclose(numeric, g[key][index], atol=1e-6, rtol=1e-4),
                  f'Gradient mismatch alpha={alpha} td={use_td} '
                  f'balanced={lw is not None} filtered={im is not None} '
                  f'at {key}{index}: {numeric} vs {g[key][index]}')
        padded = slice(2 * LEVELS, None)
        check(np.all(g['a'][:, padded] == 0) and np.all(g['ab'][padded] == 0),
              'Padded branches contributed to the loss')


def load(source, split, slot_supports_reduced):
    d = pd.read_parquet(source, filters=[('split', '==', split)])
    check(not d.empty, f'No {split} rows in {source}')
    x = np.stack(d.state).astype(np.float32)
    nx = np.stack(d.next_state).astype(np.float32)
    present = np.stack(d.device_present).astype(np.float32)
    actions = np.zeros((len(d), N), int)
    for i, a in enumerate(d.action.to_numpy()):
        a = np.asarray(a, int)
        check(not a.size or (a.min() >= 0 and a.max() < LEVELS),
              f'Action level outside 0..{LEVELS - 1}')
        actions[i, :len(a)] = a

    # Per-row legality: level 2 only where the appliance can actually dim.
    legal = np.ones((len(d), N, LEVELS), bool)
    households = d.household_id.to_numpy()
    table = {h: slot_supports_reduced[h] for h in np.unique(households)}
    legal[:, :, 2] = np.stack([table[h] for h in households])
    violations = int((~np.take_along_axis(legal, actions[:, :, None], axis=2)[:, :, 0]
                      & (present > 0)).sum())
    check(violations == 0,
          f'{violations} logged {split} actions violate the legality mask')

    imitate = np.isin(d.policy.to_numpy(), DETERMINISTIC_POLICIES).astype(np.float32)
    reward = d.reward.to_numpy(float)
    done = d.done.to_numpy(bool)
    for name, array in [('state', x), ('next_state', nx), ('reward', reward)]:
        check(np.isfinite(array).all(), f'Nonfinite {name}')
    return x, nx, actions, present, legal, imitate, reward, done


def slot_table(root):
    """supports_reduced per household, laid out in the 28 state slots."""
    models = pd.read_parquet(
        root / 'data/processed/simulator_devices_v1/unknown_quantity_one'
               '/baseline_power_v1/device_power_models.parquet',
        columns=['template_id', 'device_id', 'supports_reduced'])
    models = models.sort_values(['template_id', 'device_id'])
    models['slot'] = models.groupby('template_id').cumcount()
    check(int(models.slot.max()) < N, 'A household has more devices than slots')
    table = {}
    for household, g in models.groupby('template_id'):
        row = np.zeros(N, bool)
        row[g.slot.to_numpy()] = g.supports_reduced.to_numpy(bool)
        table[household] = row
    return table


def load_many(root, scenarios, split, table):
    parts = []
    for name in scenarios:
        path = root / SCENARIOS[name]
        check(path.exists(), f'Missing scenario {name}: {path}')
        loaded = load(path, split, table)
        parts.append(loaded)
        print(f'  {split:10s} {name:4s} {len(loaded[0]):7,d} rows  '
              f'{100 * loaded[5].mean():4.1f}% clonable')
    if len(parts) == 1:
        return parts[0]
    return tuple(np.concatenate([p[i] for p in parts], axis=0)
                 for i in range(len(parts[0])))


def bootstrap_target(net, target, nxn, present, legal, reward, done, gamma, index):
    online, _ = net.forward(nxn[index])
    allowed = (present[index][:, :, None] > 0) & legal[index]
    best = np.argmax(np.where(allowed, online, -np.inf), axis=2)
    priced, _ = target.forward(nxn[index])
    chosen = np.take_along_axis(priced, best[:, :, None], axis=2)[:, :, 0]
    per_branch = (chosen * present[index]).sum(1) / np.maximum(1, present[index].sum(1))
    return reward[index] + np.where(done[index], 0.0, gamma * per_branch)


def evaluate(net, target, data, gamma, chunk=20000):
    xn, nxn, actions, present, legal, reward, done = data
    absolute = np.zeros(len(xn))
    counted = np.zeros(len(xn))
    agree = np.zeros(LEVELS)
    logged = np.zeros(LEVELS)
    greedy_total = np.zeros(LEVELS)
    illegal = 0
    for start in range(0, len(xn), chunk):
        index = np.arange(start, min(start + chunk, len(xn)))
        q, _ = net.forward(xn[index])
        chosen = np.take_along_axis(q, actions[index][:, :, None], axis=2)[:, :, 0]
        y = bootstrap_target(net, target, nxn, present, legal, reward, done,
                             gamma, index)
        absolute[index] = (np.abs(chosen - y[:, None]) * present[index]).sum(1)
        counted[index] = present[index].sum(1)
        allowed = (present[index][:, :, None] > 0) & legal[index]
        greedy = np.argmax(np.where(allowed, q, -np.inf), axis=2)
        live = present[index] > 0
        illegal += int((~np.take_along_axis(legal[index], greedy[:, :, None],
                                            axis=2)[:, :, 0] & live).sum())
        for level in range(LEVELS):
            picked = live & (actions[index] == level)
            logged[level] += picked.sum()
            agree[level] += (picked & (greedy == level)).sum()
            greedy_total[level] += (live & (greedy == level)).sum()

    def td(rows):
        return float(absolute[rows].sum() / max(1.0, counted[rows].sum()))

    recall = agree / np.maximum(1.0, logged)
    total = max(1.0, logged.sum())
    return {
        'balanced_accuracy': float(recall.mean()),
        'raw_agreement': float(agree.sum() / total),
        'td_error_non_terminal': td(~done),
        'td_error_terminal': td(done),
        'illegal_greedy_picks': illegal,
        'recall_by_level': {LEVEL_NAMES[i]: float(recall[i]) for i in range(LEVELS)},
        'logged_share': {LEVEL_NAMES[i]: float(logged[i] / total) for i in range(LEVELS)},
        'greedy_share': {LEVEL_NAMES[i]: float(greedy_total[i] / total)
                         for i in range(LEVELS)},
    }


def fit(root, *, steps, warm_start, batch, gamma, scale, alpha, lr, seed, hidden,
        level_balance, filter_imitation, scenarios, tag):
    gradient_test()
    table = slot_table(root)
    print(f'scenarios: {list(scenarios)}  filter_imitation: {filter_imitation}')
    (x, nx, actions, present, legal, imitate, reward, done) = load_many(
        root, scenarios, 'train', table)
    (vx, vnx, vactions, vpresent, vlegal, vimitate, vreward, vdone) = load_many(
        root, scenarios, 'validation', table)

    features = x.shape[1]
    mean = x.mean(0)
    sd = x.std(0)
    zero_variance = int((sd < 1e-8).sum())
    sd[sd < 1e-8] = 1.0
    xn = ((x - mean) / sd).astype(np.float32)
    nxn = ((nx - mean) / sd).astype(np.float32)
    vxn = ((vx - mean) / sd).astype(np.float32)
    vnxn = ((vnx - mean) / sd).astype(np.float32)
    del x, nx, vx, vnx
    y = reward / scale
    vy = vreward / scale
    validation = (vxn, vnxn, vactions, vpresent, vlegal, vy, vdone)

    live = present > 0
    counts = np.array([float(((actions == l) & live).sum()) for l in range(LEVELS)])
    share = counts / counts.sum()
    level_weight = None
    if level_balance:
        level_weight = 1.0 / np.maximum(share, 1e-9)
        level_weight = level_weight / level_weight.mean()
    imitation_mask = imitate if filter_imitation else None

    net = Network(features, hidden=hidden, seed=seed)
    target = Network(features, hidden=hidden, seed=seed)
    target.p = {k: v.copy() for k, v in net.p.items()}
    rng = np.random.default_rng(seed)
    history, recent = [], []

    best = {'balanced_accuracy': -1.0}
    best_parameters = None

    def report(step, phase):
        nonlocal best, best_parameters
        metrics = evaluate(net, target, validation, gamma)
        metrics.update({'step': step, 'phase': phase,
                        'train_loss': float(np.mean(recent[-100:]))})
        history.append(metrics)
        # Keep the best checkpoint, not the last one. Under Adam this matters:
        # the run peaks early and then the temporal-difference term pulls the
        # policy away from the logged behaviour again, so shipping the final
        # weights ships a model that training had already beaten.
        if metrics['balanced_accuracy'] > best['balanced_accuracy']:
            best = dict(metrics)
            best_parameters = {k: v.copy() for k, v in net.p.items()}
        print(f"  {phase:12s} {step:6d} | loss {metrics['train_loss']:9.6f} "
              f"| TD step {metrics['td_error_non_terminal']:7.5f} "
              f"| balanced {metrics['balanced_accuracy']:.4f} "
              f"| dim recall {metrics['recall_by_level']['reduced']:.3f}", flush=True)

    if warm_start:
        print(f'behaviour-cloning warm start: {warm_start} updates')
        for step in range(1, warm_start + 1):
            i = rng.integers(0, len(xn), batch)
            recent.append(net.update(xn[i], actions[i], y[i], present[i], legal[i],
                                     alpha=1.0, lr=lr, use_td=False,
                                     level_weight=level_weight,
                                     imitate=None if imitation_mask is None
                                     else imitation_mask[i]))
            if step % 1000 == 0 or step == warm_start:
                target.p = {k: v.copy() for k, v in net.p.items()}
                report(step, 'warm_start')
        target.p = {k: v.copy() for k, v in net.p.items()}

    print(f'conservative Double-Q: {steps} updates, alpha {alpha}')
    for step in range(1, steps + 1):
        i = rng.integers(0, len(xn), batch)
        bootstrap = bootstrap_target(net, target, nxn, present, legal, y, done,
                                     gamma, i)
        recent.append(net.update(xn[i], actions[i], bootstrap, present[i], legal[i],
                                 alpha=alpha, lr=lr, use_td=True,
                                 level_weight=level_weight,
                                 imitate=None if imitation_mask is None
                                 else imitation_mask[i]))
        if step % 250 == 0:
            target.p = {k: v.copy() for k, v in net.p.items()}
        if step % 2500 == 0 or step == steps:
            report(step, 'conservative')

    check(best_parameters is not None, 'No evaluation ran, so no checkpoint to keep')
    net.p = best_parameters
    final = best
    print(f"selected step {best['step']} ({best['phase']}), "
          f"balanced {best['balanced_accuracy']:.4f}; "
          f"final step was {history[-1]['balanced_accuracy']:.4f}")
    out = root / f'models/sharp_bdq_v3_{tag}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'checkpoint.npz', mean=mean, sd=sd,
             n_features=np.array(features), n_branches=np.array(N),
             n_levels=np.array(LEVELS), **net.p)
    reloaded = np.load(out / 'checkpoint.npz')
    for k, v in net.p.items():
        check(np.array_equal(reloaded[k], v), 'Checkpoint reload mismatch')

    report_json = {
        'status': 'BDQ_V3_TRAINING_COMPLETED',
        'optimiser': 'adam',
        'legality_mask': True,
        'imitation_filtered_to': list(DETERMINISTIC_POLICIES) if filter_imitation else 'all',
        'scenarios': list(scenarios),
        'training_rows': int(len(xn)),
        'validation_rows': int(len(vxn)),
        'clonable_share_of_train': float(imitate.mean()),
        'feature_count': int(features),
        'zero_variance_features': zero_variance,
        'hidden_units': hidden,
        'parameter_count': int(sum(v.size for v in net.p.values())),
        'warm_start_updates': warm_start, 'conservative_updates': steps,
        'cql_alpha': alpha, 'batch_size': batch, 'learning_rate': lr,
        'gamma': gamma, 'reward_scale_divisor': scale, 'seed': seed,
        'level_balanced': level_balance,
        'train_logged_level_share': {LEVEL_NAMES[i]: float(share[i])
                                     for i in range(LEVELS)},
        'final': final, 'history': history,
        'checkpoint_selection': 'BEST_VALIDATION_BALANCED_ACCURACY',
        'selected_step': final['step'],
        'last_step_balanced_accuracy': history[-1]['balanced_accuracy'],
        'normalization_fit': 'TRAIN_ONLY', 'test_used': False,
        'finite_difference_gradient_test': 'PASS',
        'padded_branch_gradient_test': 'PASS',
        'checkpoint_reload': 'PASS',
        'training_source_sha256': hashlib.sha256(
            (root / SCENARIOS[scenarios[0]]).read_bytes()).hexdigest(),
        'limitations': [
            'Balanced accuracy measures agreement with scripted behaviour '
            'policies, not control quality.',
            'Filtering the imitation term to deterministic policies raises that '
            'agreement partly by no longer being asked to predict noise. It does '
            'not make the controller better at controlling.',
            'A falling TD error is not evidence of policy quality.',
            'The test split has deliberately not been touched.',
        ],
        'approved_for_deployment': False,
    }
    (out / 'training_report.json').write_text(json.dumps(report_json, indent=2),
                                              encoding='utf-8')
    print('\nBDQ V3 COMPLETED')
    print(f"  parameters          {report_json['parameter_count']:,}")
    print(f"  balanced accuracy   {final['balanced_accuracy']:.4f}")
    print(f"  raw agreement       {final['raw_agreement']:.4f}")
    print(f"  illegal picks       {final['illegal_greedy_picks']}")
    for level in LEVEL_NAMES:
        print(f"    {level:8s} logged {final['logged_share'][level]:.4f}  "
              f"greedy {final['greedy_share'][level]:.4f}  "
              f"recall {final['recall_by_level'][level]:.4f}")
    print('Output:', out)
    return report_json


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--warm-start', type=int, default=2000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--reward-scale', type=float, default=10.0)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--level-balance', action='store_true', default=True)
    p.add_argument('--no-level-balance', dest='level_balance', action='store_false')
    # Off by default: filtering costs 0.12 balanced accuracy because
    # random_binary supplies 98.3% of all dim examples. See the module docstring.
    p.add_argument('--filter-imitation', action='store_true', default=False)
    p.add_argument('--no-filter-imitation', dest='filter_imitation',
                   action='store_false')
    p.add_argument('--scenarios', nargs='+', default=['main'], choices=sorted(SCENARIOS))
    p.add_argument('--tag', default='run')
    a = p.parse_args()
    fit(a.root.resolve(), steps=a.steps, warm_start=a.warm_start, batch=a.batch,
        gamma=a.gamma, scale=a.reward_scale, alpha=a.alpha, lr=a.lr, seed=a.seed,
        hidden=a.hidden, level_balance=a.level_balance,
        filter_imitation=a.filter_imitation, scenarios=tuple(a.scenarios), tag=a.tag)
