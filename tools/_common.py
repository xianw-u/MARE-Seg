"""Portable loaders for the two released MARE-Seg checkpoints."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from mmcv import Config
from mmseg.apis import init_segmentor
from mmseg.datasets import SUIMDataset, DUTDataset

EMPTY_LEGACY_BUFFERS = {
    'decode_head.text_relation_target',
    'decode_head.text_relation_raw_target',
    'decode_head.text_relation_target_matrix',
}


def load_model(dataset, device='cpu', checkpoint=None, data_root=None):
    os.chdir(ROOT)
    cfg = Config.fromfile(str(ROOT / 'configs' / ('mare_seg_' + dataset + '.py')))
    cfg.model.pretrained = None
    cfg.model.decode_head.text_prototype_path = str(
        ROOT / 'data/class_embeddings' / (dataset + '_class_embeddings.pt'))
    for key in ('val', 'test'):
        cfg.data[key].data_root = str(Path(data_root or ROOT / 'data') / dataset.upper())
        # External paths select RGB images only; labels always come from this release.
        cfg.data[key].ann_dir = str(ROOT / 'data' / dataset.upper() / 'annotations/validation')
        cfg.data[key].split = str(ROOT / 'data/splits' / (dataset + '_validation.txt'))
    path = Path(checkpoint or ROOT / 'checkpoints' / ('mare_seg_' + dataset + '.pth'))
    # Checkpoints are supplied by this repository; verify hashes before loading.
    obj = torch.load(str(path), map_location='cpu', weights_only=False)
    model = init_segmentor(cfg, device=device)
    state = obj['state_dict']
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    expected = model.state_dict()
    extra = set(state) - set(expected)
    missing = set(expected) - set(state)
    mismatched = [k for k in set(state) & set(expected) if state[k].shape != expected[k].shape]
    if missing or mismatched or not extra <= EMPTY_LEGACY_BUFFERS or any(state[k].numel() for k in extra):
        raise RuntimeError('Checkpoint mismatch: missing=%s extra=%s shapes=%s' %
                           (sorted(missing), sorted(extra), mismatched))
    model.load_state_dict({k: v for k, v in state.items() if k not in extra}, strict=True)
    model.CLASSES = tuple(obj['meta']['CLASSES'])
    dataset_type = SUIMDataset if dataset == 'suim' else DUTDataset
    if model.CLASSES != tuple(dataset_type.CLASSES):
        raise RuntimeError('Checkpoint class order does not match the released dataset')
    model.PALETTE = obj['meta'].get('PALETTE', dataset_type.PALETTE)
    model.eval()
    return model, cfg, obj['meta']
