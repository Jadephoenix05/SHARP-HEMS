"""Prove the learned-policy hook changed nothing for the existing policies.

Why this exists. Adding a hook meant reordering when the observation is built
inside a validated 800-line generator. A reordering that quietly alters the
recorded data would invalidate the shipped dataset and every number derived from
it, and it would do so silently - the episodes would still generate, the audit
would still pass, and the values would just be different.

So this replays real episodes through the patched generator and compares every
column, cell by cell, against the rows the shipped dataset already holds for
those same episodes. Equality is the whole test. Anything else fails.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def verify(root, episodes):
    import generate_sharp_rl_transitions_v2 as gen

    shipped = pd.read_parquet(
        root / 'data/processed/sharp_rl_transitions_v2/rl_transitions.parquet')
    chosen = sorted(shipped.episode_id.unique())[:episodes]
    check(len(chosen) > 0, 'No episodes in the shipped dataset')

    # Rebuild exactly the inputs run() assembles, then replay only the chosen
    # episodes rather than regenerating the whole release.
    replayed, _, _ = gen.replay_episodes(root, chosen)
    check(len(replayed) > 0, 'Replay produced no rows')

    expected = shipped[shipped.episode_id.isin(chosen)].reset_index(drop=True)
    got = replayed.reset_index(drop=True)
    order = ['episode_id', 'step_id']
    expected = expected.sort_values(order).reset_index(drop=True)
    got = got.sort_values(order).reset_index(drop=True)

    check(len(expected) == len(got),
          f'Row count differs: shipped {len(expected)}, replayed {len(got)}')
    missing = set(expected.columns) - set(got.columns)
    check(not missing, f'Replay is missing columns: {sorted(missing)}')

    differences = []
    for column in expected.columns:
        a, b = expected[column], got[column]
        if a.map(lambda v: isinstance(v, (list, np.ndarray))).any():
            same = all(np.array_equal(np.asarray(x, float), np.asarray(y, float))
                       for x, y in zip(a, b))
        elif pd.api.types.is_float_dtype(a):
            same = np.allclose(a.to_numpy(float), b.to_numpy(float),
                               rtol=0, atol=0, equal_nan=True)
        else:
            same = a.equals(b)
        if not same:
            differences.append(column)

    print(f'episodes replayed : {len(chosen)}')
    print(f'rows compared     : {len(expected)}')
    print(f'columns compared  : {len(expected.columns)}')
    if differences:
        print(f'\nFAILED - these columns differ: {differences}')
        for column in differences[:3]:
            a, b = expected[column], got[column]
            for i in range(len(a)):
                if not np.array_equal(np.asarray(a[i]), np.asarray(b[i])):
                    print(f'  {column} row {i}: shipped {a[i]!r} vs replayed {b[i]!r}')
                    break
        return False

    print('\nPOLICY HOOK IS INERT')
    print('  every column of every replayed row is identical to the shipped data')
    print('  the reordered observation and the new policy_fn branch changed nothing')
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--episodes', type=int, default=6)
    a = p.parse_args()
    sys.exit(0 if verify(a.root.resolve(), a.episodes) else 1)
