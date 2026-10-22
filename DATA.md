# Data protocol

## Class orders

| Dataset | IDs in order | Meaning |
|---|---|---|
| SUIM | BW, HD, WR, RO, RI, FV | background; divers; wrecks/ruins; robots/equipment; reefs/invertebrates; fish/vertebrates |
| DUT | BW, SC, SU, SL, SF | background; sea cucumber; sea urchin; scallop; starfish |

Class IDs start at 0; 255 is ignored. Masks are single-channel indexed PNGs,
not color visualizations. Background participates in mean IoU.

| Dataset | Training pairs | Validation pairs |
|---|---:|---:|
| SUIM | 1525 | 110 |
| DUT | 1380 | 107 |

Exact file stems are recorded in `data/splits/`. No train/validation IDs overlap.
All annotation files in the release are the current indexed versions used for
checkpoint reevaluation. The SUIM corrected validation label `f_r_478_.png` is included.

Original dataset references:

- [SUIM dataset and benchmark](https://irvlab.cs.umn.edu/resources/suim-dataset)
- [DUT-USEG original paper](https://arxiv.org/abs/2108.11727)

This release uses the archived subset and annotation protocol documented above.
The original SUIM protocol has eight classes; this project uses six. DUT's original
collection also contains detection-only images; this release uses only its supplied
segmentation subset. Do not infer a dataset-wide split from the subset counts.

The `file_manifest.json` records every distributed mask hash. Original RGB images
are not redistributed here. File-level RGB hashes and image IDs are provided in
`reproducibility/image_manifest.json` for checking an existing copy. Dataset rights
remain with the original providers.
