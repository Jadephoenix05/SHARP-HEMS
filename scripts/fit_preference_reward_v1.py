"""Fit the Bradley-Terry preference reward r_psi from SHARP override pairs (E2).

The idea. Every time the simulated occupant overrode the controller, the dataset
recorded two actions for the same device in the same state: the level the
controller had settled on, and the level the occupant asked for instead. That is
a preference comparison, and a preference comparison is enough to fit a reward
without ever writing down a comfort weight by hand.

    P(preferred beats proposed) = sigmoid( r_psi(s, i, a_pref) - r_psi(s, i, a_prop) )

Maximising the weighted log likelihood of the observed comparisons recovers
r_psi up to a per-branch constant, which is all a reward needs to be. The weight
is the dataset's own preference_weight: occupancy gating times pressure, divided
by one plus the response latency, so a fast override by an attentive occupant
counts for more than a slow one.

Splits. Pairs carry the same household-disjoint and date-disjoint split as the
transitions. The model is fitted on train pairs only, selected on validation
pairs, and the test pairs are not touched.

READ THIS BEFORE QUOTING AN ACCURACY. Every one of the 3,180 pairs compares
preferred = 1 (ON) against proposed = 0 (OFF). The direction never varies,
because a pair is only recorded when the occupant overrode a shed. So the
constant rule 'the occupant always wants it on' scores 100 per cent, and the
classification accuracy of this model is not a result. Do not put it in the
abstract.

What IS a result is the MARGIN. r_psi(s, i, ON) - r_psi(s, i, OFF) is a learned,
state-dependent measure of how badly the occupant wants that appliance back, and
that is the quantity a reward actually needs. This script therefore reports
whether the margin recovers the generator's own override pressure on held-out
households and dates: its correlation with override_probability, and whether it
falls as response latency rises. Those are the E2 numbers.

To make the sign of the comparison informative as well, the pair generator would
have to emit pairs in both directions, including cases where the occupant let a
shed stand. That is a change to the dataset, not to this script.

What this can and cannot claim. Every pair is synthetic, generated from the
stated behavioural rule in sharp_human_model.py. Recovering that rule shows a
preference reward is learnable from override comparisons of this kind. It is NOT
evidence about real occupants, and the paper must say so every time.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

N = 28
LEVELS = 3
LEVEL_NAMES = {0: 'off', 1: 'on', 2: 'reduced'}


def check(condition, message):
    if not condition:
        raise ValueError(message)


class RewardModel:
    """r_psi(s) -> (branches, levels). One shared trunk, one score per branch level."""

    def __init__(self, features, hidden=64, seed=7):
        rng = np.random.default_rng(seed)
        self.p = {'w': rng.normal(0, np.sqrt(2 / features), (features, hidden)),
                  'b': np.zeros(hidden),
                  'r': rng.normal(0, .01, (hidden, N * LEVELS)),
                  'rb': np.zeros(N * LEVELS)}

    def forward(self, x):
        h = np.maximum(0, x @ self.p['w'] + self.p['b'])
        return (h @ self.p['r'] + self.p['rb']).reshape(-1, N, LEVELS), h

    def loss_grad(self, x, branch, preferred, proposed, weight, decay):
        r, h = self.forward(x)
        rows = np.arange(len(x))
        good = r[rows, branch, preferred]
        bad = r[rows, branch, proposed]
        margin = good - bad
        # Numerically safe -log sigmoid(margin).
        loss = float((weight * (np.logaddexp(0.0, -margin))).sum() / weight.sum())
        # d/dmargin of -log sigmoid(margin) is -sigmoid(-margin).
        scale = -(weight / weight.sum()) / (1.0 + np.exp(margin))
        dr = np.zeros_like(r)
        dr[rows, branch, preferred] += scale
        dr[rows, branch, proposed] -= scale
        dr = dr.reshape(len(x), -1)
        dh = (dr @ self.p['r'].T) * (h > 0)
        grad = {'w': x.T @ dh + decay * self.p['w'], 'b': dh.sum(0),
                'r': h.T @ dr + decay * self.p['r'], 'rb': dr.sum(0)}
        loss += 0.5 * decay * float(np.square(self.p['w']).sum()
                                    + np.square(self.p['r']).sum())
        return loss, grad

    def update(self, x, branch, preferred, proposed, weight, lr, decay):
        loss, g = self.loss_grad(x, branch, preferred, proposed, weight, decay)
        norm = np.sqrt(sum(np.square(v).sum() for v in g.values()))
        check(np.isfinite(norm) and np.isfinite(loss), 'Nonfinite preference update')
        for k, v in g.items():
            self.p[k] -= lr * v * min(1.0, 10.0 / (norm + 1e-12))
        return loss

    def margin(self, x, branch, preferred, proposed, chunk=20000):
        out = np.zeros(len(x))
        for start in range(0, len(x), chunk):
            index = np.arange(start, min(start + chunk, len(x)))
            r, _ = self.forward(x[index])
            rows = np.arange(len(index))
            out[index] = (r[rows, branch[index], preferred[index]]
                          - r[rows, branch[index], proposed[index]])
        return out


def gradient_test():
    rng = np.random.default_rng(3)
    model = RewardModel(6, hidden=5)
    x = rng.normal(size=(4, 6))
    branch = np.array([0, 1, 0, 2])
    preferred = np.array([1, 2, 1, 0])
    proposed = np.array([0, 0, 2, 1])
    weight = np.array([1.0, .5, .25, 2.0])
    _, g = model.loss_grad(x, branch, preferred, proposed, weight, 1e-3)
    for key, index in [('r', (0, 0)), ('w', (2, 1)), ('rb', (4,)), ('b', (3,))]:
        original = model.p[key][index]
        eps = 1e-6
        model.p[key][index] = original + eps
        up = model.loss_grad(x, branch, preferred, proposed, weight, 1e-3)[0]
        model.p[key][index] = original - eps
        down = model.loss_grad(x, branch, preferred, proposed, weight, 1e-3)[0]
        model.p[key][index] = original
        numeric = (up - down) / (2 * eps)
        check(np.isclose(numeric, g[key][index], atol=1e-7, rtol=1e-4),
              f'Preference gradient mismatch at {key}{index}: '
              f'{numeric} vs {g[key][index]}')


def branch_index(models):
    """Slot index is the device's rank by device_id within its household.

    This mirrors the generator exactly: it sorts each household's devices by
    device_id before writing them into the 28 state slots, and the feature schema
    records that as 'device_id ascending within household'. Getting this wrong
    would silently attach every preference to the wrong branch.
    """
    ordered = models[['template_id', 'device_id']].sort_values(
        ['template_id', 'device_id'])
    ordered['slot'] = ordered.groupby('template_id').cumcount()
    check(int(ordered.slot.max()) < N, 'A household has more devices than state slots')
    return ordered.set_index('device_id').slot


def assemble(root, split, slot_of):
    pairs = pd.read_parquet(
        root / 'data/processed/sharp_rl_transitions_v2/override_preference_pairs.parquet')
    pairs = pairs[pairs.split.eq(split)]
    check(len(pairs) > 0, f'No {split} preference pairs')

    transitions = pd.read_parquet(
        root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet',
        columns=['episode_id', 'step_id', 'split', 'state', 'device_present'],
        filters=[('split', '==', split)])
    transitions = transitions.set_index(['episode_id', 'step_id'])

    key = pd.MultiIndex.from_arrays([pairs.episode_id, pairs.step_id])
    missing = ~key.isin(transitions.index)
    check(not missing.any(), f'{int(missing.sum())} {split} pairs have no transition row')
    joined = transitions.loc[key]

    x = np.stack(joined.state.to_numpy()).astype(np.float32)
    present = np.stack(joined.device_present.to_numpy()).astype(float)
    branch = pairs.device_id.map(slot_of).to_numpy()
    check(pd.notna(branch).all(), f'{split} pairs reference unknown devices')
    branch = branch.astype(int)
    live = present[np.arange(len(branch)), branch] > 0
    check(live.all(), f'{int((~live).sum())} {split} pairs point at a padded branch')

    preferred = pairs.preferred_action.to_numpy(int)
    proposed = pairs.proposed_action.to_numpy(int)
    check(((preferred >= 0) & (preferred < LEVELS)).all(), 'Preferred level out of range')
    check(((proposed >= 0) & (proposed < LEVELS)).all(), 'Proposed level out of range')
    same = preferred == proposed
    check(not same.any(), f'{int(same.sum())} {split} pairs compare a level with itself')

    weight = pairs.preference_weight.to_numpy(float)
    check(np.isfinite(weight).all() and (weight >= 0).all(), 'Bad preference weight')

    # Some comparisons carry weight exactly zero: the occupancy gate is zero
    # because the response latency pushed the override past the moment everyone
    # left the house. The dataset is saying those pairs carry no evidence, so
    # they are dropped rather than trained on at zero weight, which would leave
    # the weighted-mean denominators counting rows that contribute nothing.
    keep = weight > 0
    dropped = int((~keep).sum())
    x, branch = x[keep], branch[keep]
    preferred, proposed, weight = preferred[keep], proposed[keep], weight[keep]
    pairs = pairs[keep]
    check(len(weight) > 0, f'All {split} pairs were zero-weight')
    return (x, branch, preferred, proposed, weight,
            pairs.reset_index(drop=True), dropped)


def score(model, x, branch, preferred, proposed, weight):
    margin = model.margin(x, branch, preferred, proposed)
    correct = margin > 0
    return {
        'pairs': int(len(margin)),
        'accuracy': float(correct.mean()),
        'weighted_accuracy': float((weight * correct).sum() / weight.sum()),
        'mean_margin': float(margin.mean()),
        'log_likelihood_per_pair': float(-np.logaddexp(0.0, -margin).mean()),
    }, margin


def fit(root, steps, batch, lr, decay, hidden, seed):
    gradient_test()
    models = pd.read_parquet(
        root / 'data/processed/simulator_devices_v1/unknown_quantity_one'
               '/baseline_power_v1/device_power_models.parquet',
        columns=['template_id', 'device_id', 'appliance_type'])
    slot_of = branch_index(models)

    train = assemble(root, 'train', slot_of)
    validation = assemble(root, 'validation', slot_of)
    x, branch, preferred, proposed, weight, train_pairs, train_dropped = train
    vx, vbranch, vpreferred, vproposed, vweight, validation_pairs, validation_dropped = validation

    # Normalisation fitted on TRAIN pairs only.
    mean = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0
    xn = ((x - mean) / sd).astype(np.float32)
    vxn = ((vx - mean) / sd).astype(np.float32)

    model = RewardModel(x.shape[1], hidden=hidden, seed=seed)
    rng = np.random.default_rng(seed)
    history, best, best_parameters = [], None, None

    for step in range(1, steps + 1):
        index = rng.integers(0, len(xn), min(batch, len(xn)))
        loss = model.update(xn[index], branch[index], preferred[index],
                            proposed[index], weight[index], lr, decay)
        if step % 250 == 0 or step == steps:
            metrics, _ = score(model, vxn, vbranch, vpreferred, vproposed, vweight)
            metrics.update({'step': step, 'train_loss': float(loss)})
            history.append(metrics)
            # Selection is on validation weighted accuracy; test is never read.
            if best is None or metrics['weighted_accuracy'] > best['weighted_accuracy']:
                best = metrics
                best_parameters = {k: v.copy() for k, v in model.p.items()}
            print(f"  step {step:5d} | train {loss:.5f} | validation accuracy "
                  f"{metrics['accuracy']:.4f} weighted {metrics['weighted_accuracy']:.4f}",
                  flush=True)

    model.p = best_parameters
    train_metrics, train_margin = score(model, xn, branch, preferred, proposed, weight)
    validation_metrics, validation_margin = score(
        model, vxn, vbranch, vpreferred, vproposed, vweight)

    # Does the recovered reward track the generator's own override pressure? The
    # simulated occupant overrides with a probability driven by discomfort and
    # unmet service. If r_psi has recovered that rule rather than memorising
    # households, its margin should rise with that probability.
    recovery = float(np.corrcoef(
        validation_margin, validation_pairs.override_probability.to_numpy(float))[0, 1])
    by_latency = {int(k): float(v) for k, v in
                  pd.Series(validation_margin).groupby(
                      validation_pairs.latency_steps.to_numpy()).mean().items()}
    by_type = (pd.DataFrame({'appliance_type': validation_pairs.appliance_type,
                             'margin': validation_margin,
                             'correct': validation_margin > 0})
               .groupby('appliance_type')
               .agg(pairs=('margin', 'size'), accuracy=('correct', 'mean'),
                    mean_margin=('margin', 'mean'))
               .sort_values('pairs', ascending=False))

    # The degenerate-direction check. If every pair points the same way, a
    # constant rule scores perfectly and accuracy says nothing about the model.
    directions = (pd.Series(list(zip(validation_pairs.preferred_action,
                                     validation_pairs.proposed_action)))
                  .value_counts())
    majority = float(directions.iloc[0] / directions.sum())
    degenerate = len(directions) == 1

    # The real E2 evidence: does the learned margin track the generator's own
    # override pressure and its latency discounting, on held-out households?
    probability = validation_pairs.override_probability.to_numpy(float)
    spearman = float(pd.Series(validation_margin).corr(
        pd.Series(probability), method='spearman'))
    by_pressure = {str(k): float(v) for k, v in
                   pd.Series(validation_margin).groupby(
                       validation_pairs.pressure_source.to_numpy()).mean().items()}

    out = root / 'models/sharp_preference_reward_v1'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'reward_model.npz', mean=mean, sd=sd,
             n_features=np.array(x.shape[1]), n_branches=np.array(N),
             n_levels=np.array(LEVELS), **model.p)
    reloaded = np.load(out / 'reward_model.npz')
    for k, v in model.p.items():
        check(np.array_equal(reloaded[k], v), 'Reward model reload mismatch')

    source = (root / 'data/processed/sharp_rl_transitions_v2'
                     '/override_preference_pairs.parquet')
    report = {
        'status': 'PREFERENCE_REWARD_FITTED',
        'experiment': 'E2',
        'model': 'Bradley-Terry over per-branch, per-level reward scores',
        'parameter_count': int(sum(v.size for v in model.p.values())),
        'hidden_units': hidden,
        'weight_decay': decay,
        'updates': steps,
        'selected_at_step': best['step'],
        'zero_weight_pairs_dropped': {'train': train_dropped,
                                      'validation': validation_dropped},
        'zero_weight_reason': ('occupancy gating is zero because response latency '
                               'pushed the override past the occupants leaving'),
        'train': train_metrics,
        'validation': validation_metrics,
        'accuracy_is_meaningless_here': degenerate,
        'comparison_directions_observed': {f'{k[0]}_over_{k[1]}': int(v)
                                           for k, v in directions.items()},
        'constant_rule_baseline_accuracy': majority,
        'headline_metrics_use_the_margin_not_the_accuracy': {
            'margin_pearson_with_override_probability': recovery,
            'margin_spearman_with_override_probability': spearman,
            'mean_margin_by_latency_steps': by_latency,
            'mean_margin_by_pressure_source': by_pressure},
        'validation_by_appliance_type': json.loads(by_type.to_json(orient='index')),
        'history': history,
        'normalization_fit': 'TRAIN_PAIRS_ONLY',
        'model_selection': 'VALIDATION_WEIGHTED_ACCURACY',
        'test_used': False,
        'finite_difference_gradient_test': 'PASS',
        'reward_model_reload': 'PASS',
        'branch_index_rule': 'device_id ascending within household, matching feature_schema.json',
        'preference_weight_formula': 'occupancy gating x pressure / (1 + latency_steps)',
        'reference': 'Bradley and Terry 1952; Christiano et al. 2017 arXiv:1706.03741',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'limitations': [
            'Every pair compares ON against OFF in that one direction, so the '
            'classification accuracy is not a result: a constant rule scores 1.0. '
            'Report the margin metrics instead.',
            'Making the direction informative requires the pair generator to also '
            'emit comparisons where the occupant accepted the shed. That is a '
            'dataset regeneration, not a change to this script.',
            'Every preference pair is synthetic, generated from the stated rule in '
            'sharp_human_model.py. Recovering it shows the rule is learnable from '
            'override comparisons, not that real occupants behave this way.',
            'The validation split holds only a few hundred pairs, so the margin '
            'correlations carry wide uncertainty.',
            'Comparisons only exist where an override happened, so the pairs are '
            'concentrated on discretionary appliances under service pressure.',
            'The reward is identified only up to a constant per branch, which is '
            'sufficient for a policy but means absolute values are not meaningful.',
            'This reward is fitted here and is not yet substituted into the '
            'training reward.',
        ],
        'approved_for_deployment': False,
    }
    (out / 'preference_reward_report.json').write_text(json.dumps(report, indent=2),
                                                       encoding='utf-8')

    print('\nPREFERENCE REWARD FITTED (E2)')
    print(f"  train pairs           {train_metrics['pairs']}")
    print(f"  validation pairs      {validation_metrics['pairs']}")
    if degenerate:
        direction = list(directions.index)[0]
        print(f"  DIRECTION IS CONSTANT: every pair prefers level {direction[0]} "
              f"over level {direction[1]}.")
        print(f"  A constant rule therefore scores {majority:.4f}. "
              "Accuracy is NOT a result here.")
    print(f"  validation accuracy   {validation_metrics['accuracy']:.4f}  "
          "(reported for completeness only)")
    print('  --- the E2 evidence is the margin ---')
    print(f"  margin vs override probability  pearson {recovery:+.4f}  "
          f"spearman {spearman:+.4f}")
    print('  mean margin by latency steps:',
          {k: round(v, 3) for k, v in by_latency.items()})
    print('  mean margin by pressure source:',
          {k: round(v, 3) for k, v in by_pressure.items()})
    print('  test_used: False')
    print('Output:', out)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--decay', type=float, default=1e-4)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    fit(a.root.resolve(), a.steps, a.batch, a.lr, a.decay, a.hidden, a.seed)
