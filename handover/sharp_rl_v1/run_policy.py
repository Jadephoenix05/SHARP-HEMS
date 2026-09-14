"""Reference inference for the SHARP policy on a Raspberry Pi. numpy only.

Copy this next to sharp_policy.npz and golden_vector.json. Run it once to check
the boot assertion, then import SharpPolicy in the controller.

    python run_policy.py          # runs the golden-vector check and exits
"""
from pathlib import Path
import json
import numpy as np

N_BRANCHES = 28
N_LEVELS = 3
LEVEL_NAMES = {0: 'SHED', 1: 'ON', 2: 'REDUCED'}


class SharpPolicy:
    def __init__(self, weights_path='sharp_policy.npz'):
        w = np.load(weights_path, allow_pickle=False)
        self.w, self.b = w['w'], w['b']
        self.v, self.vb = w['v'], w['vb']
        self.a, self.ab = w['a'], w['ab']
        # Ship these WITH the weights and use nothing else. Normalising with a
        # different mean or sd is silently wrong: no exception, no warning, just
        # confident bad commands.
        self.mean, self.sd = w['mean'], w['sd']
        if len(self.mean) != self.w.shape[0]:
            raise ValueError('Normalisation length does not match the trunk')

    def q_values(self, state, normalised=False):
        """state: 305 raw floats (or already-normalised, for the golden check)."""
        state = np.asarray(state, dtype=float)
        if state.shape != (self.w.shape[0],):
            raise ValueError(f'Expected {self.w.shape[0]} features, got {state.shape}')
        x = state if normalised else (state - self.mean) / self.sd
        h = np.maximum(0, x @ self.w + self.b)
        advantage = (h @ self.a + self.ab).reshape(N_BRANCHES, N_LEVELS)
        value = (h @ self.v + self.vb).reshape(1, 1)
        return value + advantage - advantage.mean(axis=1, keepdims=True)

    def decide(self, state, *, device_present, is_necessity, supports_reduced,
               occupant_wants, normalised=False):
        """One level per device. The shield still runs after this and may refuse.

        The two protection rules are applied here as well as in the shield, on
        purpose: the policy should not even propose an action the shield would
        have to reject, but the shield remains the authority.
        """
        device_present = np.asarray(device_present, bool)
        is_necessity = np.asarray(is_necessity, bool)
        supports_reduced = np.asarray(supports_reduced, bool)
        occupant_wants = np.asarray(occupant_wants, bool)

        legal = np.ones((N_BRANCHES, N_LEVELS), bool)
        # A critical load is never DIMMED.
        legal[:, 2] = supports_reduced & ~is_necessity
        # A critical load the occupant is using is never SHED. One they are not
        # asking for may be off - protecting essential service does not mean
        # running a fridge for an empty house.
        legal[:, 0] &= ~(is_necessity & occupant_wants)

        q = self.q_values(state, normalised=normalised)
        allowed = device_present[:, None] & legal
        if not allowed.any(axis=1).all():
            raise ValueError('A present device has no legal level at all')
        return np.argmax(np.where(allowed, q, -np.inf), axis=1)


def check_golden_vector(policy, path='golden_vector.json'):
    """Assert this at boot, BEFORE accepting any command.

    A state-builder mismatch between training and the Pi does not crash. The
    relays click, the dashboard looks right, and every decision is wrong. This
    is the only thing that catches it.
    """
    g = json.loads(Path(path).read_text())
    q = policy.q_values(g['state'], normalised=True)
    expected = np.asarray(g['expected_q'], float)
    difference = float(np.abs(q - expected).max())
    if difference > 1e-8:
        raise AssertionError(f'Q mismatch: max difference {difference:.2e}')

    present = np.asarray(g['device_present'], bool)
    allowed = present[:, None] & np.asarray(g['legal_levels'], bool)
    action = np.argmax(np.where(allowed, q, -np.inf), axis=1)
    expected_action = np.asarray(g['expected_greedy_action'])
    wrong = np.where(present & (action != expected_action))[0]
    if len(wrong):
        raise AssertionError(f'Action mismatch at slots {wrong.tolist()}')

    print(f'golden vector OK  (Q matches to {difference:.1e}, '
          f'{int(present.sum())} live devices agree)')
    for slot in np.where(present)[0]:
        print(f'  slot {slot}: {LEVEL_NAMES[int(action[slot])]}'
              + ('   [critical - protected]' if g['is_necessity'][slot] else ''))
    return True


if __name__ == '__main__':
    policy = SharpPolicy()
    print(f'loaded {policy.w.shape[0]} features -> '
          f'{N_BRANCHES} branches x {N_LEVELS} levels')
    check_golden_vector(policy)
    print('\nSafe to accept commands.')
