# MARE-Seg

**MARE-Seg: Memory-Augmented Reference Embeddings for Underwater Scene Understanding**

This repository contains the main MARE-Seg model, portable training configurations,
the SUIM and DUT main-model checkpoints, and reproducible evaluation/prediction tools.
MARE-Seg combines supervised learnable class embeddings and historical EMA visual
prototypes with SegFormer-B0. There is no language encoder, CLIP guidance, or
prototype update from target annotations at inference.

## Released models

| Dataset | Checkpoint | Classes | Validation images | Archived mIoU |
|---|---|---:|---:|---:|
| SUIM | `checkpoints/mare_seg_suim.pth` | 6 | 110 | 82.4757% |
| DUT | `checkpoints/mare_seg_dut.pth` | 5 | 107 | 72.3806% |

Only these two trained model checkpoints are included. Two small class-embedding
initialization matrices are also provided; they are training inputs, not additional
trained models. No baseline, ablation, final-iteration, or ImageNet-pretrained model
weights are included. The checkpoint/state-dict naming remains compatible with the
historical `PrAdaULPSegFormerHead`; the public registry alias is `MARESegHead`.

These scores are full-validation checkpoint reevaluations using the supplied indexed
labels and exact split IDs. They are not independent test results or a guarantee that
fresh retraining will produce identical peak scores.

## Installation

Use Python 3.10 for the historical evaluation environment. PyTorch 2.7.1,
torchvision 0.22.1, and pure-Python MMCV 1.7.1 were used for the archived reevaluation.
This model path does not require compiled MMCV CUDA operators. Do not substitute
MMCV 2.x or a current pip release of MMSegmentation for the vendored `mmseg/` package.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "setuptools<81" wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation -r requirements.txt
```

For CPU evaluation, replace `cu128` with `cpu` in the PyTorch command. On Windows,
create the environment at a short path (for example `C:\venvs\mare`) to avoid
Windows path-length errors while installing PyTorch. Run commands below from the
repository root. Retained upstream/modified sources and licenses are described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Dataset setup

The exact indexed training/validation masks and split IDs used by this project are
provided under `data/`. Original RGB images are excluded from Git and must be obtained
from the original data providers or an existing local copy. See
[DATA.md](DATA.md) for class orders and dataset sources.

```text
data/
  SUIM/
    images/training/*.jpg
    images/validation/*.jpg
    annotations/training/*.png
    annotations/validation/*.png
  DUT/
    images/training/*.jpg
    images/validation/*.jpg
    annotations/training/*.png
    annotations/validation/*.png
  splits/
  class_embeddings/
```

If you already have the same layout elsewhere, copy only the required RGB images:

```bash
python tools/prepare_images.py --source-root /path/to/existing/data
```

SUIM uses this project's **six-class indexed annotation variant**, not the original
eight-class protocol. Original RGB color masks cannot be substituted directly.
The historical eight-to-six conversion script was not recovered, so the exact
indexed masks are included instead of inventing a remapping rule.

## Verify the release and load the models

```bash
python tools/verify.py
python tools/check_data.py
python tools/preflight.py --device cpu
python tools/preflight.py --device cpu --training-step
```

The loader checks every trained tensor with `strict=True`. Only explicitly named,
zero-length legacy relation buffers may be discarded, and any other unexpected,
missing, or shape-mismatched tensor stops loading. Preflight also checks that
inference leaves the historical prototype state unchanged. The one-step training
check uses synthetic input and does not overwrite the released checkpoints.

## Reproduce validation results

```bash
python tools/evaluate.py --dataset suim --device cuda:0
python tools/evaluate.py --dataset dut --device cuda:0
```

Use `--device cpu` without a CUDA GPU. To read RGB images from another data root:

```bash
python tools/evaluate.py --dataset suim --data-root /path/to/data --device cpu
```

That external root must contain the same `SUIM/images/...` and `DUT/images/...`
layout. Evaluation always uses this release's exact split list and distributed
indexed annotations, even when RGB images are read from an external root. Results contain the confusion
matrix, per-class IoU, mIoU, parameter count, image count, and prototype-state check.
`--limit 2` is available for a quick smoke test and is explicitly marked as partial.
Numerical differences across hardware/software may slightly change predictions.

## Predict one image

```bash
python tools/predict.py /path/to/image.jpg --dataset suim --device cuda:0
```

Indexed labels, color segmentation, and an overlay are saved under `results/`.

## Train

Place the official MiT-B0 ImageNet initialization at `pretrained/mit_b0.pth`.
It is required only for training, not evaluation of the released checkpoints.
Obtain it through the [official SegFormer repository](https://github.com/NVlabs/SegFormer).

```bash
python tools/train.py configs/mare_seg_suim.py --seed 42 --deterministic
python tools/train.py configs/mare_seg_dut.py --seed 42 --deterministic
```

The SUIM archive records seed 42 and deterministic execution. The DUT checkpoint's
embedded configuration records no seed; the DUT command above selects an explicit
seed for a new reproducible run and is not a claim about its historical training seed.
Training configurations retain each run's model, optimizer, schedule, preprocessing,
and class embeddings while removing automatic resume and external cloud logging.

The DUT released checkpoint records learning rate **3e-4**. The current manuscript
text states 6e-4; the code follows checkpoint provenance. Resolve this discrepancy
against the historical training record before final paper submission. Other
provenance details are in [reproducibility/NOTES.md](reproducibility/NOTES.md).

## Repository scope

The private repository is a staging release for reproducibility. Manuscript drafts,
personal files, source RGB datasets, credentials, virtual environments, and unrelated
experiments are excluded. Dataset redistribution terms and upstream source licenses
continue to apply. Public release can be performed later by the owner.
