"""Prepare the archived raw sources as a second PRIVATE Kaggle dataset.

Two datasets, deliberately separate:
  sharp-master-dataset-v1   ~26 MB   the RL training data you attach to notebooks
  sharp-raw-sources-v1     ~2.8 GB   the raw source archives, for rebuilding

Keeping them apart means a training notebook does not pull 2.8 GB it will never
read, and a rebuild does not force you to re-download the training data.

PRIVACY IS NOT OPTIONAL HERE. Several upstream licences (MoSPI TUS, eMARC,
IRES, BEE/CLASP) do not permit public redistribution. This script writes
metadata with no public flag and prints a reminder; making the dataset public
afterwards is a licence decision with real consequences.
"""
from pathlib import Path
import argparse
import json


def build(root, username, archive_dir):
    manifest_path = archive_dir / 'RAW_MANIFEST.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Run archive_sharp_raw_sources.py first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    slug = 'sharp-raw-sources-v1'
    metadata = {
        'title': 'SHARP Raw Sources V1',
        'id': f'{username}/{slug}',
        'licenses': [{'name': 'other'}],
    }
    (archive_dir / 'dataset-metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')

    restricted = [e for e in manifest['entries'] if e['licence_restricted']]
    lines = [
        '# SHARP Raw Sources V1',
        '',
        'PRIVATE archival copy of the raw datasets behind the SHARP Master Dataset.',
        'These are SOURCE datasets. They do NOT train the model. The reinforcement',
        'learning data is the separate `sharp-master-dataset-v1` dataset.',
        '',
        f"{manifest['raw_gb']} GB of raw data stored as {manifest['archive_gb']} GB "
        f"of verified archives ({manifest['overall_ratio']}x compression).",
        'Every archive was opened and its file count checked after writing, and',
        'every archive has a SHA-256 in `RAW_MANIFEST.json`.',
        '',
        '## Contents',
        '',
        '| Archive | Raw | Compressed | Files | Licence restricted |',
        '|---|---|---|---|---|',
    ]
    for entry in sorted(manifest['entries'], key=lambda e: -e['archive_bytes']):
        lines.append(
            f"| `{entry['archive']}` | {entry['raw_bytes'] / 1024 ** 3:.2f} GB "
            f"| {entry['archive_bytes'] / 1024 ** 2:.0f} MB | {entry['files']} "
            f"| {'yes' if entry['licence_restricted'] else 'no'} |")

    lines += [
        '',
        '## Restore an archive',
        '',
        '```python',
        'import tarfile',
        'tarfile.open("/kaggle/input/sharp-raw-sources-v1/refit.tar.gz"'
        ').extractall("data/raw")',
        '```',
        '',
        '## Licences',
        '',
        'Each upstream source keeps its own licence. This dataset must stay PRIVATE:',
        '',
    ]
    for entry in restricted:
        lines.append(f"- `{entry['archive']}` — {entry['licence_note']}")
    lines += [
        '',
        'REFIT CLEANED is CC BY 4.0 (University of Strathclyde). RESIDE-AC is CC0.',
        'NASA POWER is public domain. The rest carry use agreements that do not',
        'permit public redistribution.',
        '',
        '## Provenance note',
        '',
        'REFIT is 20 UK homes and is never Indian household data. iAWE is one New',
        'Delhi home and is not a population. RESIDE-AC is 11 Hyderabad houses over',
        '19 days in May 2019 and is not Andhra Pradesh.',
    ]
    (archive_dir / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('RAW KAGGLE DATASET PREPARED')
    print('  folder:  ', archive_dir)
    print('  dataset: ', metadata['id'])
    print(f"  size:     {manifest['archive_gb']} GB in {len(manifest['entries'])} archives")
    print(f'  licence-restricted archives included: {len(restricted)}')
    print()
    print('  Create it (PRIVATE by default, do not pass --public):')
    print(f'    kaggle datasets create -p {archive_dir} --dir-mode tar')
    print()
    print('  Upload is a few GB, so expect it to take a while.')
    print('  After upload, confirm Settings -> Visibility shows Private.')
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--username', required=True, help='your Kaggle username')
    p.add_argument('--archive-dir', type=Path, default=None)
    a = p.parse_args()
    root = a.root.resolve()
    build(root, a.username, (a.archive_dir or root / 'raw_archive').resolve())
