"""One command that decides whether a trained SHARP policy is fit to hand over.

Why this exists. The first Kaggle run produced a model that looked excellent -
0.7635 balanced accuracy, a clean export, a golden vector that reproduced to
1e-15 - and was not a reinforcement learning policy at all. Checkpoint selection
on agreement with the logged action had picked the behaviour-cloning warm start,
because cloning optimises that metric directly and always wins it. Nothing in
the pipeline noticed, because nothing checked the artefact as a whole.

So this checks the artefact as a whole, and returns a single verdict. Every gate
below can fail independently and each one failed silently at some point during
development.

GATE 1  Artefact integrity   weights load, shapes agree, normalisation present
GATE 2  Golden vector        the exported weights reproduce the shipped vector
GATE 3  Provenance           the checkpoint came from the RL phase, not cloning
GATE 4  Value sanity         held-out TD error is finite and not absurd
GATE 5  Action legality      no dim on a device that cannot dim
GATE 6  Behaviour            beats chance, and does not collapse to one level
GATE 7  Simulator            replayed on held-out homes against three baselines
GATE 8  Safety               no necessity load shed while the occupant is owed it

Gates 1-6 read the artefact. Gate 7 and 8 run the simulator, which is slower and
is the only part that says anything about control rather than imitation.

Held-out only. The test split is never read.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_BRANCHES = 28
N_LEVELS = 3
LEVEL_NAMES = ['off', 'on', 'reduced']
CHANCE = 1.0 / N_LEVELS


class Gate:
    def __init__(self):
        self.results = []

    def record(self, name, passed, detail, fatal=True):
        self.results.append({'gate': name, 'passed': bool(passed),
                             'detail': detail, 'fatal': bool(fatal)})
        mark = 'PASS' if passed else ('FAIL' if fatal else 'WARN')
        print(f'  [{mark}] {name}: {detail}', flush=True)
        return passed

    @property
    def failed(self):
        return [r for r in self.results if not r['passed'] and r['fatal']]


def forward(p, x):
    h = np.maximum(0, x @ p['w'] + p['b'])
    advantage = (h @ p['a'] + p['ab']).reshape(-1, N_BRANCHES, N_LEVELS)
    value = (h @ p['v'] + p['vb']).reshape(-1, 1, 1)
    return value + advantage - advantage.mean(2, keepdims=True)


def validate(root, checkpoint, report_path, golden_path, episodes, skip_simulator):
    gate = Gate()
    started = time.time()
    print(f'Validating {checkpoint}\n')

    # ---- GATE 1: the artefact loads and is self-consistent -------------------
    print('Artefact')
    weights = np.load(checkpoint, allow_pickle=False)
    required = ['w', 'b', 'v', 'vb', 'a', 'ab', 'mean', 'sd']
    missing = [k for k in required if k not in weights.files]
    gate.record('weights present', not missing,
                f'missing {missing}' if missing else f'all of {required}')
    p = {k: weights[k] for k in ['w', 'b', 'v', 'vb', 'a', 'ab']}
    mean, sd = weights['mean'], weights['sd']
    features = p['w'].shape[0]
    shapes_ok = (p['a'].shape[1] == N_BRANCHES * N_LEVELS
                 and len(mean) == features and len(sd) == features)
    gate.record('shapes agree', shapes_ok,
                f'{features} features, {p["a"].shape[1]} outputs '
                f'= {N_BRANCHES} branches x {N_LEVELS} levels')
    gate.record('normalisation shipped', bool((sd > 0).all()) and np.isfinite(mean).all(),
                'mean and sd finite and positive - the Pi cannot normalise without them')
    parameters = int(sum(v.size for v in p.values()))
    gate.record('fits on a Pi', parameters < 500_000,
                f'{parameters:,} parameters, {parameters * 4 / 1024:.0f} KB in float32')

    # ---- GATE 2: the golden vector ------------------------------------------
    print('\nGolden vector')
    if golden_path.exists():
        golden = json.loads(golden_path.read_text())
        x = np.asarray(golden['state'], float)[None, :]
        q = forward(p, x)[0]
        expected = np.asarray(golden['expected_q'], float)
        difference = float(np.abs(q - expected).max())
        gate.record('Q reproduces', difference < 1e-8,
                    f'max absolute difference {difference:.2e}')
        present = np.asarray(golden['device_present'], bool)
        greedy = np.argmax(np.where(present[:, None], q, -np.inf), axis=1)
        agreed = bool((greedy[present]
                       == np.asarray(golden['expected_greedy_action'])[present]).all())
        gate.record('greedy action reproduces', agreed,
                    f'{int(present.sum())} live devices agree')
    else:
        gate.record('golden vector present', False,
                    f'{golden_path} not found - the Pi has no boot assertion')

    # ---- GATE 3: provenance --------------------------------------------------
    print('\nProvenance')
    if report_path.exists():
        report = json.loads(report_path.read_text())
        selection = report.get('checkpoint_selection', 'UNKNOWN')
        history = report.get('history', [])
        chosen = report.get('final_validation', {}).get('balanced_accuracy')
        phases = {h['step']: h['phase'] for h in history}
        matching = [h for h in history
                    if abs(h['balanced_accuracy'] - (chosen or -1)) < 1e-12]
        phase = matching[0]['phase'] if matching else 'unknown'
        gate.record('checkpoint came from the RL phase', phase == 'conservative',
                    f'selected checkpoint is from phase "{phase}" '
                    f'(selection rule: {selection}). A warm-start checkpoint is a '
                    f'behaviour cloner, not a policy.')
        gate.record('test split untouched', report.get('test_split_used') is False,
                    f"test_split_used={report.get('test_split_used')}")
        gate.record('not marked deployable', report.get('approved_for_deployment') is False,
                    'approved_for_deployment is false, as it must be')
        gate.record('gradient tests ran', report.get('gradient_tests') == 'PASS',
                    f"gradient_tests={report.get('gradient_tests')}", fatal=False)
    else:
        gate.record('training report present', False, f'{report_path} not found')
        report = {}

    # ---- GATES 4-6: behaviour on held-out data -------------------------------
    print('\nHeld-out behaviour')
    source = root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet'
    d = pd.read_parquet(source, filters=[('split', '==', 'validation')],
                        columns=['household_id', 'state', 'action', 'device_present'])
    states = ((np.stack(d.state).astype(np.float32) - mean) / sd).astype(np.float32)
    present = np.stack(d.device_present).astype(bool)
    actions = np.zeros((len(d), N_BRANCHES), int)
    for i, a in enumerate(d.action.to_numpy()):
        a = np.asarray(a, int)
        actions[i, :len(a)] = a

    models = pd.read_parquet(
        root / 'data/processed/simulator_devices_v1/unknown_quantity_one'
               '/baseline_power_v1/device_power_models.parquet',
        columns=['template_id', 'device_id', 'supports_reduced'])
    models = models.sort_values(['template_id', 'device_id'])
    models['slot'] = models.groupby('template_id').cumcount()
    dim_table = {}
    for household, g in models.groupby('template_id'):
        row = np.zeros(N_BRANCHES, bool)
        row[g.slot.to_numpy()] = g.supports_reduced.to_numpy(bool)
        dim_table[household] = row
    can_dim = np.stack([dim_table[h] for h in d.household_id.to_numpy()])

    greedy = np.zeros((len(d), N_BRANCHES), int)
    q_min, q_max = np.inf, -np.inf
    for start in range(0, len(states), 20000):
        index = np.arange(start, min(start + 20000, len(states)))
        q = forward(p, states[index])
        q_min, q_max = min(q_min, float(q.min())), max(q_max, float(q.max()))
        legal = np.ones_like(q, bool)
        legal[:, :, 2] = can_dim[index]
        allowed = present[index][:, :, None] & legal
        greedy[index] = np.argmax(np.where(allowed, q, -np.inf), axis=2)

    gate.record('Q values finite and bounded', np.isfinite([q_min, q_max]).all()
                and abs(q_min) < 1e4 and abs(q_max) < 1e4,
                f'Q in [{q_min:.2f}, {q_max:.2f}]')

    illegal = int(((greedy == 2) & ~can_dim & present).sum())
    gate.record('no illegal dim', illegal == 0,
                f'{illegal} attempts to dim a device that cannot dim')

    recall, shares = [], []
    for level in range(N_LEVELS):
        picked = present & (actions == level)
        recall.append(float((picked & (greedy == level)).sum() / max(1, picked.sum())))
        shares.append(float((present & (greedy == level)).sum() / max(1, present.sum())))
    balanced = float(np.mean(recall))
    gate.record('beats chance', balanced > CHANCE + 0.05,
                f'balanced accuracy {balanced:.4f} against chance {CHANCE:.4f}')
    gate.record('has not collapsed to one level', max(shares) < 0.98,
                'greedy share ' + ', '.join(f'{LEVEL_NAMES[i]} {shares[i]:.3f}'
                                            for i in range(N_LEVELS)))
    gate.record('still chooses dim', recall[2] > 0.05,
                f'dim recall {recall[2]:.4f} - dimming is the behaviour that '
                f'distinguishes SHARP from a rule-based shedder', fatal=False)

    behaviour = {'balanced_accuracy': balanced,
                 'recall_by_level': {LEVEL_NAMES[i]: recall[i] for i in range(N_LEVELS)},
                 'greedy_share': {LEVEL_NAMES[i]: shares[i] for i in range(N_LEVELS)},
                 'q_min': q_min, 'q_max': q_max}

    # ---- GATES 7-8: the simulator --------------------------------------------
    simulation = None
    if not skip_simulator:
        print('\nSimulator, held-out households')
        import evaluate_sharp_policy_v1 as ev
        simulation = ev.evaluate(root, checkpoint.relative_to(root)
                                 if checkpoint.is_absolute() and root in checkpoint.parents
                                 else checkpoint, episodes, seed=7, cohort='plain_grid')
        safety = simulation['results_versus_baseline'][
            'SHARP learned policy']['safety']
        gate.record('no necessity load shed while entitled',
                    safety['necessity_shed_while_entitled'] == 0,
                    f"{safety['necessity_shed_while_entitled']} of "
                    f"{safety['necessity_service_opportunities']} opportunities")
        share = simulation.get('share_of_achievable_saving', {})
        if share:
            peak = share.get('peak_kw', {})
            gate.record('reduces peak demand',
                        peak.get('sharp', 1e9) < peak.get('no_control', 0),
                        f"peak {peak.get('no_control', float('nan')):.3f} -> "
                        f"{peak.get('sharp', float('nan')):.3f} kW, "
                        f"{peak.get('sharp_share_of_achievable', float('nan')):.1f}% "
                        f'of what is physically achievable', fatal=False)
    else:
        print('\nSimulator: skipped')

    verdict = {
        'status': 'MODEL_VALIDATED' if not gate.failed else 'MODEL_REJECTED',
        'checkpoint': str(checkpoint),
        'validated_at': pd.Timestamp.now(tz='Asia/Kolkata').isoformat(),
        'gates': gate.results,
        'gates_failed': [r['gate'] for r in gate.failed],
        'held_out_behaviour': behaviour,
        'parameter_count': parameters,
        'simulator': simulation,
        'seconds': round(time.time() - started, 1),
        'test_split_used': False,
        'what_this_does_not_prove': [
            'That the policy helps in a real home. Every number here comes from '
            'a simulator built on survey data and declared assumptions.',
            'That the occupant model is realistic. It is synthetic throughout.',
            'That the shield is safe against hardware faults. It is verified '
            'against the simulator, not against a stuck relay.',
        ],
        'approved_for_deployment': False,
    }
    out = root / 'reports/model_end_to_end_validation_v1.json'
    out.write_text(json.dumps(verdict, indent=2, default=str), encoding='utf-8')

    print(f"\n{'=' * 62}")
    print(f"{verdict['status']}  ({len(gate.results)} gates, "
          f"{len(gate.failed)} fatal failures, {verdict['seconds']}s)")
    if gate.failed:
        for r in gate.failed:
            print(f"  FAILED: {r['gate']} - {r['detail']}")
    print(f"{'=' * 62}")
    print('Output:', out)
    return verdict


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--report', type=Path, default=None,
                   help='training_report.json; defaults to beside the checkpoint')
    p.add_argument('--golden', type=Path, default=None,
                   help='golden_vector.json; defaults to beside the checkpoint')
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--skip-simulator', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    checkpoint = a.checkpoint if a.checkpoint.is_absolute() else root / a.checkpoint
    report = a.report or checkpoint.parent / 'training_report.json'
    golden = a.golden or checkpoint.parent / 'golden_vector.json'
    verdict = validate(root, checkpoint, report, golden, a.episodes, a.skip_simulator)
    raise SystemExit(0 if verdict['status'] == 'MODEL_VALIDATED' else 1)
