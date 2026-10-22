"""Verify tracked release files against their SHA-256 manifest (standard library only)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'reproducibility/file_manifest.json').read_text(encoding='utf-8'))
    failures = []
    for entry in manifest:
        p = ROOT / entry['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != entry['sha256']:
            failures.append(entry['path'])
    if failures:
        raise SystemExit('Missing or modified files: ' + ', '.join(failures))
    print('Verified %d release files.' % len(manifest))


if __name__ == '__main__':
    main()
