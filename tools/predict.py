"""Predict a single image; save indexed labels and a color visualization."""
import argparse
from pathlib import Path
from _common import ROOT, load_model
import numpy as np
import torch
from PIL import Image
from mmseg.apis import inference_segmentor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('--dataset', choices=['suim', 'dut'], required=True)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/predictions')
    args = p.parse_args()
    image = args.image.resolve()
    out = args.output_dir.resolve()
    torch.set_num_threads(8)
    model, _, _ = load_model(args.dataset, args.device)
    pred = inference_segmentor(model, str(image))[0].astype('uint8')
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(out / (image.stem + '_labels.png'))
    color = Image.fromarray(np.asarray(model.PALETTE, dtype='uint8')[pred])
    color.save(out / (image.stem + '_color.png'))
    Image.blend(Image.open(image).convert('RGB'), color, 0.45).save(out / (image.stem + '_overlay.png'))
    print(out)


if __name__ == '__main__':
    main()
