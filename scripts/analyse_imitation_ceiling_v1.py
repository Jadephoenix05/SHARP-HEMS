"""How high can agreement with the logged action actually go?

The question this answers. Balanced accuracy against the logged action is the
metric the trainer reports, and it is natural to want to push it toward 1.0. But
the logged action is whatever three scripted behaviour policies did, and one of
those three is `random_binary` - it samples. Its choices are not a function of
the state, so no model of the state can predict them, however large.

That puts a hard ceiling on the metric that has nothing to do with model quality.
This script measures where the ceiling is, by scoring the trained policy
separately against each behaviour policy, and by computing what a perfect
predictor would score.

The conclusion matters for how the result is reported: a number close to the
ceiling is a good model, and a number close to 1.0 would mean the dataset had
stopped containing any exploration at all.
"""
from pathlib import Path
import argparse
import sys
import json
import numpy as np
import pandas as pd

N = 28
LEVELS = 3
LEVEL_NAMES = ['off', 'on', 'reduced']


sys.path.insert(0, str(Path(__file__).resolve().parent))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def forward(p, x):
    h = np.maximum(0, x @ p['w'] + p['b'])
    advantage = (h @ p['a'] + p['ab']).reshape(-1, N, LEVELS)
    value = (h @ p['v'] + p['vb']).reshape(-1, 1, 1)
    return value + advantage - advantage.mean(2, keepdims=True)


def balanced_accuracy(actions, greedy, live):
    recall = []
    for level in range(LEVELS):
        picked = live & (actions == level)
        recall.append(float((picked & (greedy == level)).sum() / max(1, picked.sum())))
    return float(np.mean(recall)), recall


def analyse(root, checkpoint, clone_steps):
    weights = np.load(root / checkpoint / 'checkpoint.npz')
    p = {k: weights[k] for k in ['w', 'b', 'v', 'vb', 'a', 'ab']}
    mean, sd = weights['mean'], weights['sd']

    source = root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet'
    d = pd.read_parquet(source, filters=[('split', '==', 'validation')],
                        columns=['policy', 'household_id', 'state', 'action',
                                 'device_present'])
    x = ((np.stack(d.state).astype(np.float32) - mean) / sd).astype(np.float32)
    present = np.stack(d.device_present).astype(float)
    actions = np.zeros((len(d), N), int)
    for i, a in enumerate(d.action.to_numpy()):
        a = np.asarray(a, int)
        actions[i, :len(a)] = a

    greedy = np.zeros((len(d), N), int)
    for start in range(0, len(x), 20000):
        index = np.arange(start, min(start + 20000, len(x)))
        q = forward(p, x[index])
        greedy[index] = np.argmax(
            np.where(present[index][:, :, None] > 0, q, -np.inf), axis=2)

    live = present > 0
    policy = d.policy.to_numpy()
    overall, overall_recall = balanced_accuracy(actions, greedy, live)

    per_policy = {}
    for name in np.unique(policy):
        rows = policy == name
        score, recall = balanced_accuracy(actions[rows], greedy[rows], live[rows])
        per_policy[name] = {
            'balanced_accuracy': score,
            'recall_by_level': {LEVEL_NAMES[i]: recall[i] for i in range(LEVELS)},
            'share_of_validation': float(rows.mean()),
        }

    # The ceiling, measured rather than assumed. For each behaviour policy we
    # fit a dedicated behaviour cloner on that policy's own training rows and
    # score it on that policy's validation rows. A policy that is a function of
    # the state can be cloned well; one that samples cannot be, however much
    # capacity and data you give it. The gap between the three is the answer.
    from train_sharp_bdq_v2 import Network, load

    tx, _, tactions, tpresent, _, _ = load(source, 'train')
    tpolicy = pd.read_parquet(source, filters=[('split', '==', 'train')],
                              columns=['policy']).policy.to_numpy()
    check(len(tpolicy) == len(tx), 'Train policy labels misaligned with states')

    ceilings = {}
    for name in np.unique(policy):
        rows = tpolicy == name
        subset = ((tx[rows] - mean) / sd).astype(np.float32)
        subset_actions, subset_present = tactions[rows], tpresent[rows]
        live_subset = subset_present > 0
        counts = np.array([float(((subset_actions == lv) & live_subset).sum())
                           for lv in range(LEVELS)])
        weights_by_level = 1.0 / np.maximum(counts / counts.sum(), 1e-9)
        weights_by_level = weights_by_level / weights_by_level.mean()

        cloner = Network(subset.shape[1], hidden=128, seed=11)
        rng = np.random.default_rng(11)
        for _ in range(clone_steps):
            index = rng.integers(0, len(subset), 256)
            cloner.update(subset[index], subset_actions[index],
                          np.zeros(len(index)), subset_present[index],
                          alpha=1.0, lr=1e-3, use_td=False,
                          level_weight=weights_by_level)

        held_out = policy == name
        vx = x[held_out]
        cloned = np.zeros((held_out.sum(), N), int)
        for start_row in range(0, len(vx), 20000):
            index = np.arange(start_row, min(start_row + 20000, len(vx)))
            q, _ = cloner.forward(vx[index])
            cloned[index] = np.argmax(
                np.where(present[held_out][index][:, :, None] > 0, q, -np.inf), axis=2)
        score, _ = balanced_accuracy(actions[held_out], cloned, live[held_out])
        ceilings[name] = score
        print(f'  cloner for {name:18s} held-out balanced accuracy {score:.4f}',
              flush=True)

    report = {
        'status': 'IMITATION_CEILING_MEASURED',
        'checkpoint': str(checkpoint),
        'split': 'validation',
        'question': ('How high can balanced accuracy against the logged action '
                     'go, given that one of three behaviour policies samples?'),
        'model_balanced_accuracy_overall': overall,
        'model_recall_by_level': {LEVEL_NAMES[i]: overall_recall[i]
                                  for i in range(LEVELS)},
        'model_by_behaviour_policy': per_policy,
        'dedicated_cloner_per_policy': ceilings,
        'clone_steps': clone_steps,
        'chance_level': 1.0 / LEVELS,
        'interpretation': [
            'random_binary samples its action, so its choices are not a function '
            'of the state and no model can predict them beyond chance. It is one '
            'third of every split.',
            'A score near the ceiling is a good model. A score near 1.0 would '
            'mean the logged data contained no exploration, which would make the '
            'dataset worse, not the model better.',
            'Agreement with a scripted policy is not control quality. Cost '
            'against baseline, peak-to-average ratio and comfort hours are, and '
            'those need the simulator in the loop.',
        ],
    }
    out = root / 'reports/imitation_ceiling_v1.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('IMITATION CEILING (validation)')
    print(f'  chance                       {1 / LEVELS:.4f}')
    print(f'  model, all policies          {overall:.4f}')
    print()
    print('  by behaviour policy:')
    for name, metrics in sorted(per_policy.items()):
        print(f"    {name:18s} model {metrics['balanced_accuracy']:.4f}   "
              f"ceiling {ceilings[name]:.4f}")
    print()
    print('  ceiling = a model trained on that ONE policy, scored held out')
    print('Output:', out)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint', type=Path,
                   default=Path('models/sharp_bdq_v2_cql1.0_balanced_bc_long'))
    p.add_argument('--clone-steps', type=int, default=6000)
    a = p.parse_args()
    analyse(a.root.resolve(), a.checkpoint, a.clone_steps)
