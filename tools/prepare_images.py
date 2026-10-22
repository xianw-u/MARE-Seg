"""Copy only the archived split images from an existing dataset directory."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True, help='Contains SUIM/ and DUT/')
    args = p.parse_args()
    copied = 0
    for ds in ['suim','dut']:
        for split in ['training','validation']:
            ids = (ROOT / 'data/splits' / (ds + '_' + split + '.txt')).read_text().splitlines()
            for name in ids:
                src = args.source_root / ds.upper() / 'images' / split / (name + '.jpg')
                if not src.is_file():
                    raise FileNotFoundError(src)
                dst = ROOT / 'data' / ds.upper() / 'images' / split / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.resolve() != dst.resolve(): shutil.copy2(src, dst)
                copied += 1
    print('Prepared %d images. Image directories are excluded from Git.' % copied)


if __name__ == '__main__':
    main()
