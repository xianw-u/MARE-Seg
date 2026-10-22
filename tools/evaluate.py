"""Reevaluate a released checkpoint on every image in its exact validation split."""
import argparse
import json
from pathlib import Path

from _common import ROOT, load_model
import numpy as np
import torch
from PIL import Image
from mmseg.apis import inference_segmentor
from mmseg.datasets import build_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['suim', 'dut'])
    p.add_argument('--data-root', type=Path, help='Directory containing SUIM/ and DUT/')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--limit', type=int, help='Smoke test only; not a full validation result')
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model, cfg, meta = load_model(args.dataset, args.device, args.checkpoint, args.data_root)
    dataset = build_dataset(cfg.data.test)
    if tuple(dataset.CLASSES) != tuple(model.CLASSES):
        raise RuntimeError('Dataset/checkpoint class order differs')
    n = len(dataset.CLASSES)
    confusion = np.zeros((n, n), dtype=np.int64)
    infos = dataset.img_infos[:args.limit] if args.limit else dataset.img_infos
    buffers_before = {k: v.clone() for k, v in model.decode_head.named_buffers()}
    for i, info in enumerate(infos):
        prediction = inference_segmentor(model, str(Path(dataset.img_dir) / info['filename']))[0]
        target = np.asarray(Image.open(Path(dataset.ann_dir) / info['ann']['seg_map']))
        if prediction.shape != target.shape:
            raise RuntimeError('Prediction/annotation size mismatch')
        valid = (target >= 0) & (target < n)
        confusion += np.bincount(n * target[valid].astype(np.int64) + prediction[valid],
                                 minlength=n*n).reshape(n, n)
        if (i + 1) % 10 == 0 or i + 1 == len(infos):
            print('%s %d/%d' % (args.dataset.upper(), i + 1, len(infos)), flush=True)
    for key, value in model.decode_head.named_buffers():
        if not torch.equal(value, buffers_before[key]):
            raise RuntimeError('Inference changed decode-head buffer: ' + key)
    union = confusion.sum(0) + confusion.sum(1) - np.diag(confusion)
    iou = np.divide(np.diag(confusion), union, out=np.full(n, np.nan), where=union != 0)
    result = dict(dataset=args.dataset.upper(), images=len(infos),
                  full_validation=len(infos) == len(dataset),
                  mIoU=float(np.nanmean(iou)),
                  class_iou=dict(zip(dataset.CLASSES, map(float, iou))),
                  parameters=sum(x.numel() for x in model.parameters()),
                  checkpoint_iteration=meta.get('iter'), device=args.device,
                  inference_buffers_unchanged=True, confusion=confusion.tolist())
    print(json.dumps(result, indent=2))
    output = args.output or ROOT / 'results' / (args.dataset + '_evaluation.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
