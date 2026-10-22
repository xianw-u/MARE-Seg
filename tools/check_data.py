"""Validate exact split IDs, indexed annotation ranges, and optional RGB image sizes."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-root', type=Path, help='Optional data root containing SUIM/ and DUT/')
    args = p.parse_args()
    report = {}
    for ds, classes in [('suim',6), ('dut',5)]:
        splits = {}
        for split in ['training','validation']:
            ids = (ROOT / 'data/splits' / (ds+'_'+split+'.txt')).read_text().splitlines()
            assert len(ids) == len(set(ids)), 'Duplicate IDs'
            splits[split] = ids
            for name in ids:
                path = ROOT / 'data' / ds.upper() / 'annotations' / split / (name+'.png')
                with Image.open(path) as mask:
                    a = np.asarray(mask)
                    assert a.ndim == 2, 'Mask is not indexed: '+str(path)
                    assert np.all((a < classes) | (a == 255)), 'Unexpected class ID: '+str(path)
                    if args.image_root:
                        with Image.open(args.image_root / ds.upper() / 'images' / split / (name+'.jpg')) as rgb:
                            assert rgb.size == mask.size, 'Image/mask sizes differ: '+name
        assert not set(splits['training']) & set(splits['validation']), 'Train/validation leakage'
        report[ds] = {k:len(v) for k,v in splits.items()}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
