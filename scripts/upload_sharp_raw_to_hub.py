"""Upload archived raw sources to a PRIVATE Hugging Face dataset repository.

Why Hugging Face rather than GitHub or Google Drive:
  * GitHub rejects files over 100 MB and is built for code diffs, not static data.
  * Google Drive serves large files through a virus-scan interstitial, so plain
    URL downloads break in scripts.
  * The Hub is built for large data files, gives resumable hash-verified
    downloads, allows free private repositories, and reaches any Python code
    through one function call.

Authentication. This script NEVER takes a token as an argument and never prints
one. Set it in your environment first so it stays out of shell history and logs:

    Windows PowerShell:   $env:HF_TOKEN = "hf_..."
    bash:                 export HF_TOKEN=hf_...

Create the token at https://huggingface.co/settings/tokens with WRITE access.

Licence gate. Archives flagged licence_restricted in RAW_MANIFEST.json are
SKIPPED unless you pass --include-restricted. TUS microdata in particular came
through a MoSPI registration agreement; uploading it anywhere, even privately,
is your decision to make deliberately rather than by default.
"""
from pathlib import Path
import argparse
import json
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--archive-dir', type=Path, default=None)
    p.add_argument('--repo', required=True,
                   help='Target repo, for example your-username/sharp-raw-sources-v1')
    p.add_argument('--include-restricted', action='store_true',
                   help='Also upload archives flagged as licence restricted')
    p.add_argument('--public', action='store_true',
                   help='Create a PUBLIC repo. Do not use for this data.')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    root = a.root.resolve()
    archive_dir = (a.archive_dir or root / 'raw_archive').resolve()
    manifest_path = archive_dir / 'RAW_MANIFEST.json'
    if not manifest_path.exists():
        sys.exit(f'No RAW_MANIFEST.json in {archive_dir}. '
                 'Run archive_sharp_raw_sources.py first.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    selected, skipped = [], []
    for entry in manifest['entries']:
        if entry['licence_restricted'] and not a.include_restricted:
            skipped.append(entry)
        else:
            selected.append(entry)

    print(f'Repository: {a.repo}   private: {not a.public}')
    print(f'Archives to upload: {len(selected)}')
    for entry in selected:
        print(f"  {entry['archive']}  {entry['archive_bytes'] / 1024 / 1024:.0f} MB")
    if skipped:
        print(f'Skipped as licence restricted ({len(skipped)}):')
        for entry in skipped:
            print(f"  {entry['archive']}  -- {entry['licence_note']}")
        print('  Pass --include-restricted only if you have checked those terms.')

    total = sum(e['archive_bytes'] for e in selected)
    print(f'Total upload: {total / 1024 ** 3:.2f} GB')
    if a.public:
        print('\nREFUSING to create a public repository for raw source data.')
        print('Several upstream licences forbid public redistribution.')
        sys.exit(2)
    if a.dry_run:
        print('\nDry run only. Nothing uploaded.')
        return

    token = os.environ.get('HF_TOKEN')
    if not token:
        sys.exit('HF_TOKEN is not set in the environment. See this file\'s docstring.')

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=a.repo, repo_type='dataset', private=True,
                    exist_ok=True)
    print('\nRepository ready. Uploading...', flush=True)

    for entry in selected:
        path = archive_dir / entry['archive']
        print(f"  uploading {entry['archive']} "
              f"({entry['archive_bytes'] / 1024 / 1024:.0f} MB)", flush=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=entry['archive'],
                        repo_id=a.repo, repo_type='dataset')

    api.upload_file(path_or_fileobj=str(manifest_path),
                    path_in_repo='RAW_MANIFEST.json',
                    repo_id=a.repo, repo_type='dataset')

    readme = f"""---
license: other
---

# SHARP raw source archives

PRIVATE archival copy of the raw datasets behind the SHARP Master Dataset.
Not for redistribution. Each upstream source keeps its own licence.

Total {manifest['raw_gb']} GB of raw data stored as {manifest['archive_gb']} GB
of verified archives ({manifest['overall_ratio']}x). Every archive has a
SHA-256 in `RAW_MANIFEST.json`.

## Load from any Python code

```python
from huggingface_hub import hf_hub_download
import tarfile

path = hf_hub_download(
    repo_id="{a.repo}", filename="refit.tar.gz",
    repo_type="dataset", token="hf_...")
tarfile.open(path).extractall("data/raw")
```

These are SOURCE datasets. They do not train the model. The RL training data is
the separate SHARP Master Dataset release.
"""
    api.upload_file(path_or_fileobj=readme.encode('utf-8'), path_in_repo='README.md',
                    repo_id=a.repo, repo_type='dataset')

    print('\nUPLOAD COMPLETE')
    print(f'  https://huggingface.co/datasets/{a.repo}')
    print(f'  uploaded {len(selected)} archives, {total / 1024 ** 3:.2f} GB')
    if skipped:
        print(f'  {len(skipped)} archive(s) held back for licence review')


if __name__ == '__main__':
    main()
