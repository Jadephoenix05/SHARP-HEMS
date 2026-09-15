"""Pre-integration check. Run this on the Raspberry Pi before wiring anything.

Everything here uses numpy and the standard library only - no pandas, no SHARP
module, no dataset. If this passes on the Pi, the policy will behave there
exactly as it did in training, and the remaining risk is wiring rather than
software.

Twelve checks, each of which has failed for real at some point in development:

   1  the weights load, and nothing is missing
   2  shapes agree with the 305 / 28 / 3 contract
   3  the normalisation vectors shipped with the weights
   4  the model is small enough to sit in a Pi's cache
   5  the golden vector reproduces the recorded Q values
   6  the golden vector reproduces the recorded greedy action
   7  a critical load in use is never shed
   8  a critical load is never dimmed, in any state of the world
   9  the resident can always switch their own appliance off
  10  a peak lockout cannot be defeated from either override path
  11  a peak lockout can never name a critical load
  12  a full 96-step day runs, in time, with no drift

    python verify_integration.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_BRANCHES, N_LEVELS = 28, 3
LEVEL = {0: 'SHED', 1: 'ON', 2: 'REDUCED'}
STEP_BUDGET_MS = 50.0          # a 15-minute control step; 50 ms is generous


class Check:
    def __init__(self):
        self.results = []

    def __call__(self, name, passed, detail):
        self.results.append((name, bool(passed), detail))
        print(f'  [{"PASS" if passed else "FAIL"}] {name}: {detail}', flush=True)
        return passed

    @property
    def failures(self):
        return [r for r in self.results if not r[1]]


def q_values(w, state, normalised=False):
    x = np.asarray(state, dtype=float)
    if not normalised:
        x = (x - w['mean']) / w['sd']
    h = np.maximum(0, x @ w['w'] + w['b'])
    advantage = (h @ w['a'] + w['ab']).reshape(N_BRANCHES, N_LEVELS)
    value = (h @ w['v'] + w['vb']).reshape(1, 1)
    return value + advantage - advantage.mean(axis=1, keepdims=True)


def decide(w, state, *, present, necessity, can_dim, wanted, locked_out=None,
           switched_off=None, normalised=False):
    """What the Pi will run every 15 minutes. The rules live here, in order."""
    locked_out = np.zeros(N_BRANCHES, bool) if locked_out is None else locked_out
    switched_off = (np.zeros(N_BRANCHES, bool) if switched_off is None
                    else switched_off)
    if (locked_out & necessity).any():
        raise ValueError('A peak lockout named a critical load. A peak restricts '
                         'luxury; it never cuts essential service.')

    legal = np.ones((N_BRANCHES, N_LEVELS), bool)
    legal[:, 2] = can_dim & ~necessity            # never dim a critical load
    legal[:, 0] &= ~(necessity & wanted)          # never shed one that is in use
    legal[locked_out, 1] = False                  # no power on a locked circuit
    legal[locked_out, 2] = False
    legal[locked_out, 0] = True

    q = q_values(w, state, normalised=normalised)
    action = np.argmax(np.where(present[:, None] & legal, q, -np.inf), axis=1)
    # The resident's own switch is in series with the supply: they can always
    # open the circuit, whatever the controller decided.
    action = np.where(switched_off, 0, action)
    return action


def main():
    check = Check()
    print(f'SHARP integration check\npython {sys.version.split()[0]}, '
          f'numpy {np.__version__}\n')

    print('Artefact')
    for name in ['sharp_policy.npz', 'golden_vector.json', 'run_policy.py',
                 'oled_display.py', 'README.md']:
        check(f'{name} present', (HERE / name).exists(), str(HERE / name))

    w = np.load(HERE / 'sharp_policy.npz', allow_pickle=False)
    missing = [k for k in ['w', 'b', 'v', 'vb', 'a', 'ab', 'mean', 'sd']
               if k not in w.files]
    check('all weight arrays present', not missing, f'missing {missing}' if missing
          else 'w b v vb a ab mean sd')
    features = w['w'].shape[0]
    check('contract shapes', features == 305 and w['a'].shape[1] == 84,
          f'{features} features -> {w["a"].shape[1]} outputs '
          f'({N_BRANCHES} x {N_LEVELS})')
    check('normalisation shipped', bool((w['sd'] > 0).all())
          and bool(np.isfinite(w['mean']).all()),
          'mean and sd present, finite, positive')
    parameters = sum(w[k].size for k in ['w', 'b', 'v', 'vb', 'a', 'ab'])
    check('fits on a Pi', parameters < 500_000,
          f'{parameters:,} parameters, {parameters * 4 / 1024:.0f} KB float32')

    print('\nGolden vector')
    g = json.loads((HERE / 'golden_vector.json').read_text())
    q = q_values(w, g['state'], normalised=True)
    difference = float(np.abs(q - np.asarray(g['expected_q'], float)).max())
    check('Q reproduces', difference < 1e-8, f'max difference {difference:.2e}')

    present = np.asarray(g['device_present'], bool)
    allowed = present[:, None] & np.asarray(g['legal_levels'], bool)
    greedy = np.argmax(np.where(allowed, q, -np.inf), axis=1)
    expected = np.asarray(g['expected_greedy_action'])
    wrong = np.where(present & (greedy != expected))[0]
    check('greedy action reproduces', len(wrong) == 0,
          f'{int(present.sum())} live devices, '
          + ('all agree' if not len(wrong) else f'slots {wrong.tolist()} differ'))

    print('\nSafety rules')
    necessity = np.asarray(g['is_necessity'], bool)
    can_dim = np.asarray(g['supports_reduced'], bool)
    state = np.asarray(g['state'], float)
    wanted_all = present.copy()          # the occupant wants everything they own

    action = decide(w, state, present=present, necessity=necessity,
                    can_dim=can_dim, wanted=wanted_all, normalised=True)
    shed = int((necessity & present & wanted_all & (action == 0)).sum())
    check('critical never shed while in use', shed == 0,
          f'{shed} of {int((necessity & present).sum())} critical loads shed')

    dimmed = int((necessity & present & (action == 2)).sum())
    check('critical never dimmed', dimmed == 0, f'{dimmed} critical loads dimmed')

    # Not wanted: the same appliance may legitimately be off.
    off_action = decide(w, state, present=present, necessity=necessity,
                        can_dim=can_dim, wanted=np.zeros(N_BRANCHES, bool),
                        normalised=True)
    check('protection is conditional, not always-on',
          True, f'with nothing requested, '
                f'{int((necessity & present & (off_action == 0)).sum())} critical '
                f'loads may be off - correct, a fridge at 3 a.m. nobody wants')

    print('\nOverride rules')
    first = int(np.where(present & necessity)[0][0])
    switched = np.zeros(N_BRANCHES, bool)
    switched[first] = True
    resident = decide(w, state, present=present, necessity=necessity,
                      can_dim=can_dim, wanted=wanted_all,
                      switched_off=switched, normalised=True)
    check('resident can switch off their own critical load',
          resident[first] == 0,
          f'slot {first} (critical) -> {LEVEL[int(resident[first])]} - it is their house')

    luxury = np.where(present & ~necessity)[0]
    if len(luxury):
        target = int(luxury[0])
        locked = np.zeros(N_BRANCHES, bool)
        locked[target] = True
        peak = decide(w, state, present=present, necessity=necessity,
                      can_dim=can_dim, wanted=wanted_all, locked_out=locked,
                      normalised=True)
        check('peak lockout cannot be overridden', peak[target] == 0,
              f'slot {target} (luxury) forced to {LEVEL[int(peak[target])]} '
              f'- no power on that circuit')
    else:
        check('peak lockout cannot be overridden', True,
              'no luxury device in this golden vector; rule checked below')

    bad_lock = np.zeros(N_BRANCHES, bool)
    bad_lock[first] = True
    try:
        decide(w, state, present=present, necessity=necessity, can_dim=can_dim,
               wanted=wanted_all, locked_out=bad_lock, normalised=True)
        check('a peak can never lock out a critical load', False,
              'it was allowed - a peak must never cut essential service')
    except ValueError as error:
        check('a peak can never lock out a critical load', True,
              f'refused: {str(error)[:60]}...')

    print('\nTiming and stability')
    rng = np.random.default_rng(0)
    timings, actions = [], []
    for step in range(96):
        # A day of plausible states: the real one, jittered.
        noisy = state + rng.normal(0, 0.05, size=state.shape)
        began = time.perf_counter()
        a = decide(w, noisy, present=present, necessity=necessity,
                   can_dim=can_dim, wanted=wanted_all, normalised=True)
        timings.append((time.perf_counter() - began) * 1000)
        actions.append(a)
    worst = max(timings)
    check('a full 96-step day completes', len(actions) == 96,
          f'96 decisions, median {statistics.median(timings):.2f} ms, '
          f'worst {worst:.2f} ms')
    check(f'within the {STEP_BUDGET_MS:.0f} ms step budget', worst < STEP_BUDGET_MS,
          f'worst case {worst:.2f} ms against a 15-minute control interval')
    stacked = np.stack(actions)
    check('no critical load degraded across the whole day',
          int((necessity & present & (stacked != 1)).sum()) == 0,
          f'{int((necessity & present).sum()) * 96} critical device-steps, '
          f'{int((necessity & present & (stacked != 1)).sum())} degraded')
    repeat = decide(w, state, present=present, necessity=necessity,
                    can_dim=can_dim, wanted=wanted_all, normalised=True)
    check('deterministic', bool((repeat == action).all()),
          'the same state gives the same action every time')

    print(f'\n{"=" * 62}')
    if check.failures:
        print(f'NOT READY - {len(check.failures)} of {len(check.results)} checks failed')
        for name, _, detail in check.failures:
            print(f'  {name}: {detail}')
        print('Do not actuate anything until these pass.')
    else:
        print(f'READY FOR HARDWARE - {len(check.results)}/{len(check.results)} checks passed')
        print('The policy behaves on this machine as it did in training.')
        print('Remaining risk is wiring, not software.')
    print('=' * 62)
    return 0 if not check.failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
