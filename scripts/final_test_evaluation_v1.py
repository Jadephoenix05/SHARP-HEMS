"""The one and only evaluation on the TEST split. Run once, report, stop.

The test split has been untouched for the whole project: never read during
training, never used to pick a checkpoint, never used to choose a
hyperparameter. That is what makes these numbers worth quoting. It also means
they can only be earned once - every further look at test turns it into another
validation set, and the figure stops meaning anything.

So this script is deliberately separate from the trainer. Training cannot reach
the test split even by accident, because the notebook never opens that file. The
decision to look is an explicit, recorded act.

What it reports:

  behaviour   action agreement, per behaviour policy and balanced
  legality    actions the deployment mask would forbid
  safety      critical loads shed while the occupant wanted them
  control     replayed in the simulator against three baselines and the oracle

Test households and test dates are both disjoint from training.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_BRANCHES, N_LEVELS = 28, 3
LEVEL_NAMES = ['off', 'on', 'reduced']
GLOBAL_FEATURES, DEVICE_FEATURES = 25, 10
REMAINING_HOURS, PREFERRED_SERVICE = 0, 6


def check(condition, message):
    if not condition:
        raise ValueError(message)


def device_tables(root):
    m = pd.read_parquet(
        root / 'data/processed/simulator_devices_v1/unknown_quantity_one'
               '/baseline_power_v1/device_power_models.parquet',
        columns=['template_id', 'device_id', 'supports_reduced', 'is_necessity'])
    m = m.sort_values(['template_id', 'device_id'])
    m['slot'] = m.groupby('template_id').cumcount()
    check(int(m.slot.max()) < N_BRANCHES, 'A household exceeds the branch count')
    dim, need = {}, {}
    for household, g in m.groupby('template_id'):
        row = np.zeros(N_BRANCHES, bool)
        row[g.slot.to_numpy()] = g.supports_reduced.to_numpy(bool)
        dim[household] = row
        row = np.zeros(N_BRANCHES, bool)
        row[g.slot.to_numpy()] = g.is_necessity.to_numpy(bool)
        need[household] = row
    return dim, need


def evaluate(root, checkpoint, episodes, skip_simulator):
    started = time.time()
    weights = np.load(root / checkpoint, allow_pickle=False)
    p = {k: weights[k] for k in ['w', 'b', 'v', 'vb', 'a', 'ab']}
    mean, sd = weights['mean'], weights['sd']

    print('Reading the TEST split. This is the one look.\n')
    d = pd.read_parquet(
        root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet',
        filters=[('split', '==', 'test')],
        columns=['policy', 'household_id', 'state', 'action', 'device_present'])
    check(len(d) > 0, 'No test rows')

    train_homes = set(pd.read_parquet(
        root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet',
        filters=[('split', '==', 'train')], columns=['household_id']
    ).household_id.unique())
    test_homes = set(d.household_id.unique())
    overlap = train_homes & test_homes
    check(not overlap, f'{len(overlap)} households appear in train AND test')
    print(f'{len(d):,} rows, {len(test_homes)} households, '
          f'0 shared with training')

    raw = np.stack(d.state.to_numpy()).astype(np.float32)
    x = ((raw - mean) / sd).astype(np.float32)
    present = np.stack(d.device_present.to_numpy()).astype(bool)
    actions = np.zeros((len(d), N_BRANCHES), int)
    for i, a in enumerate(d.action.to_numpy()):
        a = np.asarray(a, int)
        actions[i, :len(a)] = a

    dim, need = device_tables(root)
    can_dim = np.stack([dim[h] for h in d.household_id])
    necessity = np.stack([need[h] for h in d.household_id])
    slot = np.arange(N_BRANCHES) * DEVICE_FEATURES + GLOBAL_FEATURES
    wanted = ((raw[:, slot + PREFERRED_SERVICE] > 0)
              & (raw[:, slot + REMAINING_HOURS] > 0))
    entitled = necessity & wanted

    # The deployment mask: never dim a critical load, never shed one in use.
    deployment = np.ones((len(d), N_BRANCHES, N_LEVELS), bool)
    deployment[:, :, 2] = can_dim & ~necessity
    deployment[:, :, 0] = ~entitled

    greedy = np.zeros((len(d), N_BRANCHES), int)
    for start in range(0, len(x), 20000):
        index = np.arange(start, min(start + 20000, len(x)))
        h = np.maximum(0, x[index] @ p['w'] + p['b'])
        advantage = (h @ p['a'] + p['ab']).reshape(-1, N_BRANCHES, N_LEVELS)
        q = ((h @ p['v'] + p['vb'])[:, :, None] + advantage
             - advantage.mean(2, keepdims=True))
        allowed = present[index][:, :, None] & deployment[index]
        greedy[index] = np.argmax(np.where(allowed, q, -np.inf), axis=2)

    live = present
    policy = d.policy.to_numpy()
    agreement = {}
    for name in sorted(set(policy)):
        rows = policy == name
        agreement[name] = float((greedy[rows][live[rows]]
                                 == actions[rows][live[rows]]).mean())
    deterministic = np.isin(policy, ['serve_preferred', 'peak_aware'])
    agreement['deterministic_only'] = float(
        (greedy[deterministic][live[deterministic]]
         == actions[deterministic][live[deterministic]]).mean())
    agreement['all_policies'] = float((greedy[live] == actions[live]).mean())

    recall = {}
    for level in range(N_LEVELS):
        picked = live & (actions == level)
        recall[LEVEL_NAMES[level]] = float(
            (picked & (greedy == level)).sum() / max(1, picked.sum()))

    illegal_dim = int(((greedy == 2) & ~can_dim & live).sum())
    critical_shed = int((entitled & live & (greedy == 0)).sum())
    critical_dimmed = int((necessity & live & (greedy == 2)).sum())

    print(f'\n{"=" * 60}\nTEST SPLIT RESULTS\n{"=" * 60}')
    print('Action agreement')
    for name in ['serve_preferred', 'peak_aware', 'random_binary']:
        if name in agreement:
            print(f'  {name:22s} {100 * agreement[name]:6.2f}%')
    print(f'  {"DETERMINISTIC ONLY":22s} {100 * agreement["deterministic_only"]:6.2f}%')
    print(f'  {"all policies":22s} {100 * agreement["all_policies"]:6.2f}%')
    print(f'\nBalanced accuracy       {np.mean(list(recall.values())):.4f}'
          f'   (chance {1 / N_LEVELS:.4f})')
    print('  recall ' + '  '.join(f'{k} {v:.3f}' for k, v in recall.items()))
    print(f'\nLegality and safety')
    print(f'  illegal dim attempts            {illegal_dim}')
    print(f'  critical shed while entitled    {critical_shed} '
          f'of {int((entitled & live).sum()):,}')
    print(f'  critical dimmed, ever           {critical_dimmed}')

    simulation = None
    if not skip_simulator:
        print('\nReplaying test households in the simulator...')
        import evaluate_sharp_policy_v1 as ev
        shipped = pd.read_parquet(
            root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet',
            columns=['episode_id', 'split', 'policy', 'household_id'])
        plain = ev.plain_grid_households(root)
        pool = sorted(shipped.loc[shipped.split.eq('test')
                                  & shipped.policy.eq('serve_preferred')
                                  & shipped.household_id.isin(plain)]
                      .episode_id.unique())
        check(len(pool) > 0, 'No plain-grid test episodes')
        rng = np.random.default_rng(11)
        chosen = list(rng.choice(pool, size=min(episodes, len(pool)), replace=False))
        import generate_sharp_rl_transitions_v2 as gen
        inputs = gen.load_inputs(root)
        learned = ev.load_policy(root / checkpoint)
        arms = {'no demand response': ('serve_preferred', None),
                'rule-based': ('peak_aware', None),
                'SHARP': ('learned', learned),
                'oracle ceiling': ('learned', ev.maximum_curtailment_oracle)}
        per_day = {}
        for label, (name, fn) in arms.items():
            frame, _, _ = gen.replay_episodes(root, chosen, policy_fn=fn,
                                              policy_override=name, inputs=inputs)
            per_day[label] = ev.summarise(frame, label)
            print(f'  {label:22s} {len(frame):6d} steps')
        base = per_day['no demand response']
        simulation = {}
        print(f'\n{"metric":18s} {"no DR":>9s} {"rule":>9s} {"SHARP":>9s} {"oracle":>9s}')
        for metric in ['cost_inr', 'peak_kw', 'peak_to_average', 'import_kwh']:
            values = {label: float(f[metric].mean()) for label, f in per_day.items()}
            simulation[metric] = values
            print(f'{metric:18s} {values["no demand response"]:9.3f} '
                  f'{values["rule-based"]:9.3f} {values["SHARP"]:9.3f} '
                  f'{values["oracle ceiling"]:9.3f}')
        head = (simulation['peak_kw']['no demand response']
                - simulation['peak_kw']['oracle ceiling'])
        got = (simulation['peak_kw']['no demand response']
               - simulation['peak_kw']['SHARP'])
        simulation['peak_share_of_achievable'] = (
            float(100 * got / head) if abs(head) > 1e-9 else float('nan'))
        simulation['peak_cut_percent'] = float(
            100 * got / simulation['peak_kw']['no demand response'])
        print(f"\nPeak cut {simulation['peak_cut_percent']:.2f}% "
              f"= the relief of blacking out "
              f"{simulation['peak_cut_percent']:.1f} homes in 100, with none cut off")
        print(f"That is {simulation['peak_share_of_achievable']:.1f}% of what is "
              f"physically achievable")

    report = {
        'status': 'TEST_SPLIT_EVALUATED_ONCE',
        'checkpoint': str(checkpoint),
        'evaluated_at': pd.Timestamp.now(tz='Asia/Kolkata').isoformat(),
        'rows': int(len(d)),
        'households': int(len(test_homes)),
        'households_shared_with_training': 0,
        'action_agreement': agreement,
        'balanced_accuracy': float(np.mean(list(recall.values()))),
        'recall_by_level': recall,
        'illegal_dim_attempts': illegal_dim,
        'critical_shed_while_entitled': critical_shed,
        'critical_dimmed_ever': critical_dimmed,
        'critical_protection_opportunities': int((entitled & live).sum()),
        'simulator': simulation,
        'seconds': round(time.time() - started, 1),
        'honest_scope': [
            'These numbers are from a simulator built on survey data, declared '
            'assumptions and a synthetic occupant. They are not field evidence.',
            'Action agreement measures reproduction of three scripted behaviour '
            'policies, not control quality.',
            'random_binary samples its action from a draw absent from the state, '
            'so no model can predict it; quote the deterministic figure with '
            'that stated.',
            'The test split has now been read. It cannot be read again without '
            'becoming a second validation set.',
        ],
        'approved_for_deployment': False,
    }
    out = root / 'reports/final_test_evaluation_v1.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nOutput: {out}   ({report["seconds"]}s)')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint', type=Path,
                   default=Path('handover/sharp_rl_v1/sharp_policy.npz'))
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--skip-simulator', action='store_true')
    a = p.parse_args()
    evaluate(a.root.resolve(), a.checkpoint, a.episodes, a.skip_simulator)
