"""Compress each raw source into a verified archive for cloud storage.

Non-destructive. Originals are never touched. Each archive is written, its
SHA-256 recorded, and its member list read back to confirm the archive opens
and contains the expected number of files before it is accepted.

Raw CSV compresses heavily (REFIT measured at about 7x), so roughly 13 GB of
raw sources becomes roughly 3 GB of archives that fit comfortably in a private
cloud repository.

Licence note. TUS microdata came through a MoSPI registration agreement and
eMARC has its own terms. This script archives them like anything else, but
whether you may upload a given archive is a licence question, not a technical
one. The manifest flags the restricted ones so you can exclude them.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import time

# Sources whose terms may restrict uploading anywhere, even privately.
LICENCE_RESTRICTED = {
    'tus_india': 'MoSPI registration agreement. Prefer physical media over cloud.',
    'emarc': 'Prayas / Harvard Dataverse terms. Check before uploading.',
    'ires': 'Harvard Dataverse terms. Check before uploading.',
    'bee_clasp': 'CLASP report content is copyrighted. Derived statistics only.',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def folder_stats(path):
    files = [p for p in path.rglob('*') if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def archive_one(source, out_dir, level):
    name = source.name
    count, raw_bytes = folder_stats(source)
    if count == 0:
        return None
    target = out_dir / f'{name}.tar.gz'
    print(f'  {name}: {count} files, {raw_bytes / 1024 / 1024:.0f} MB -> compressing',
          flush=True)
    started = time.time()
    with tarfile.open(target, 'w:gz', compresslevel=level) as tar:
        tar.add(source, arcname=name)

    # Read the archive back: it must open and hold every file.
    with tarfile.open(target, 'r:gz') as tar:
        members = sum(1 for m in tar.getmembers() if m.isfile())
    if members != count:
        target.unlink(missing_ok=True)
        raise ValueError(f'{name}: archive holds {members} files, expected {count}')

    size = target.stat().st_size
    ratio = raw_bytes / size if size else 0
    print(f'  {name}: -> {size / 1024 / 1024:.0f} MB ({ratio:.1f}x) '
          f'in {time.time() - started:.0f}s, {members} files verified', flush=True)
    return {
        'name': name, 'archive': target.name, 'files': members,
        'raw_bytes': raw_bytes, 'archive_bytes': size,
        'compression_ratio': round(ratio, 2), 'sha256': sha256(target),
        'licence_restricted': name in LICENCE_RESTRICTED,
        'licence_note': LICENCE_RESTRICTED.get(name),
    }


def build(root, out_dir, level, only):
    raw = root / 'data/raw'
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in raw.iterdir() if p.is_dir())
    if only:
        sources = [p for p in sources if p.name in only]
    if not sources:
        raise ValueError('No raw source folders found')

    print(f'Archiving {len(sources)} raw sources into {out_dir}\n')
    entries = []
    for source in sources:
        entry = archive_one(source, out_dir, level)
        if entry:
            entries.append(entry)

    # Loose files directly under data/raw are archived together.
    loose = [p for p in raw.iterdir() if p.is_file()]
    if loose:
        target = out_dir / 'raw_loose_files.tar.gz'
        with tarfile.open(target, 'w:gz', compresslevel=level) as tar:
            for p in loose:
                tar.add(p, arcname=p.name)
        entries.append({
            'name': 'raw_loose_files', 'archive': target.name, 'files': len(loose),
            'raw_bytes': sum(p.stat().st_size for p in loose),
            'archive_bytes': target.stat().st_size,
            'compression_ratio': None, 'sha256': sha256(target),
            'licence_restricted': False, 'licence_note': None})

    raw_total = sum(e['raw_bytes'] for e in entries)
    archive_total = sum(e['archive_bytes'] for e in entries)
    manifest = {
        'status': 'RAW_SOURCES_ARCHIVED',
        'archives': len(entries),
        'raw_bytes': raw_total,
        'archive_bytes': archive_total,
        'raw_gb': round(raw_total / 1024 ** 3, 2),
        'archive_gb': round(archive_total / 1024 ** 3, 2),
        'overall_ratio': round(raw_total / archive_total, 2) if archive_total else None,
        'originals_deleted': False,
        'restricted_archives': [e['name'] for e in entries if e['licence_restricted']],
        'entries': entries,
        'restore': 'python -c "import tarfile,sys; tarfile.open(sys.argv[1]).extractall(sys.argv[2])" <archive> data/raw',
    }
    (out_dir / 'RAW_MANIFEST.json').write_text(json.dumps(manifest, indent=2),
                                               encoding='utf-8')
    print(f"\nRAW SOURCES ARCHIVED")
    print(f"  {manifest['raw_gb']} GB -> {manifest['archive_gb']} GB "
          f"({manifest['overall_ratio']}x)")
    print(f"  archives: {len(entries)}   originals deleted: False")
    if manifest['restricted_archives']:
        print('  licence-restricted (review before uploading):',
              ', '.join(manifest['restricted_archives']))
    print('  manifest:', out_dir / 'RAW_MANIFEST.json')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--level', type=int, default=6, help='gzip level 1-9')
    p.add_argument('--only', nargs='*', default=None)
    a = p.parse_args()
    root = a.root.resolve()
    build(root, (a.out or root / 'raw_archive').resolve(), a.level, a.only)
