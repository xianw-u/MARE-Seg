# Validation of the prepared release

The prepared repository was checked locally on 2026-10-22 using Python 3.11,
PyTorch 2.7.1+cpu, torchvision 0.22.1+cpu, and MMCV 1.7.1. This is a fresh CPU
reevaluation of the supplied checkpoints, separate from the historical GPU environment.

| Dataset | Images evaluated | Fresh CPU mIoU | Historical GPU mIoU | Paper, rounded |
|---|---:|---:|---:|---:|
| SUIM | 110/110 | 82.476260% | 82.475655% | 82.48% |
| DUT | 107/107 | 72.380378% | 72.380562% | 72.38% |

Both CPU results round to the paper values. The differences from the historical GPU
results are below 0.001 percentage points. No retraining or new performance claim is
implied. Complete confusion matrices and per-class IoUs are saved in
`verified_suim_evaluation.json` and `verified_dut_evaluation.json`.

Both models passed strict state-dict loading and a finite forward pass. The
historical decode-head buffers remained unchanged throughout full validation.
Synthetic one-step training checks passed finite-loss/backpropagation/optimizer
updates, produced gradients for the class embeddings, and preserved the archived
detached-gate behavior. See `verified_preflight.json`. The CUDA training launcher
was syntax/import checked; a full training run was not performed.

The single-image prediction command was also exercised. RGB images were read from
the existing local dataset; they are not included in the repository. The exact
indexed annotations and split IDs are included, and the RGB image manifest records
the expected input files.
