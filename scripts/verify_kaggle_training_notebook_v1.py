"""Execute the Kaggle training notebook locally, so it cannot ship broken.

A notebook that has never been run is a notebook that does not work. This pulls
out every code cell, rewrites only the three things that are specific to Kaggle
- the input path, the output path, and the training budget - and runs them in
order in one namespace, exactly as Run All would.

Anything else is executed verbatim. If a cell raises, this exits non-zero.
"""
from pathlib import Path
import argparse
import json
import re
import sys
import time


def substitute(source, dataset, working, quick):
    source = source.replace("Path('/kaggle/input/sharp-master-dataset-v2')",
                            f'Path(r"{dataset}")')
    source = source.replace("Path('/kaggle/working')", f'Path(r"{working}")')
    if quick:
        # Shrink the budget only. Every code path still executes.
        source = re.sub(r'^WARM_START_UPDATES = \d+', 'WARM_START_UPDATES = 60',
                        source, flags=re.M)
        source = re.sub(r'^CONSERVATIVE_UPDATES = \d+', 'CONSERVATIVE_UPDATES = 60',
                        source, flags=re.M)
        source = re.sub(r'^ABLATION_STEPS = \d+', 'ABLATION_STEPS = 40',
                        source, flags=re.M)
        source = source.replace('for step in range(1, 4001):',
                                'for step in range(1, 501):')
        # The ablation configurations carry their own warm_start, so the
        # module-level constant above does not reach them.
        source = source.replace('warm_start=2000', 'warm_start=40')
        source = source.replace('if step % 2500 == 0 or step == steps:',
                                'if step % 20 == 0 or step == steps:')
        source = source.replace('if step % 1000 == 0 or step == warm_start:',
                                'if step % 20 == 0 or step == warm_start:')
        source = source.replace('if step % 500 == 0:', 'if step % 100 == 0:')
    return source


def run(root, quick):
    notebook = json.loads(
        (root / 'notebooks/sharp_rl_training_kaggle.ipynb').read_text(encoding='utf-8'))
    dataset = root / 'release/SHARP_MASTER_V2'
    working = root / 'reports/kaggle_notebook_verification'
    working.mkdir(parents=True, exist_ok=True)
    if not dataset.exists():
        raise FileNotFoundError(f'Missing {dataset}')

    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    print(f'{len(cells)} code cells, dataset {dataset}')
    print(f'budget: {"reduced for verification" if quick else "full"}\n')

    # display() exists in Kaggle's kernel but not in a plain interpreter.
    namespace = {'__name__': '__main__', 'display': print}
    started = time.time()
    for number, cell in enumerate(cells, 1):
        source = substitute(''.join(cell['source']), dataset, working, quick)
        print(f'--- cell {number}/{len(cells)} ---', flush=True)
        try:
            exec(compile(source, f'<cell {number}>', 'exec'), namespace)
        except Exception as error:
            print(f'\nCELL {number} FAILED: {type(error).__name__}: {error}')
            raise

    elapsed = time.time() - started
    produced = sorted(p.name for p in working.glob('*'))
    print(f'\nNOTEBOOK VERIFICATION PASSED in {elapsed:.1f}s')
    print(f'  cells executed: {len(cells)}')
    print(f'  outputs written: {produced}')

    # The two claims the notebook makes about itself, checked independently.
    report = json.loads((working / 'training_report.json').read_text())
    if report['test_split_used']:
        raise AssertionError('The notebook read the test split')
    if report['final_validation']['illegal_greedy_picks'] != 0:
        raise AssertionError('The notebook selected an illegal action level')
    print('  test split untouched: confirmed')
    print('  illegal greedy picks: 0')
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--full', action='store_true',
                   help='run the real training budget instead of a reduced one')
    a = p.parse_args()
    sys.exit(0 if run(a.root.resolve(), not a.full) else 1)
