# Third-party source notices

This repository includes modified source derived from:

- [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer), particularly the MiT backbone
  and SegFormer decoder. Its NVIDIA Source Code License is retained at
  `LICENSES/NVIDIA-SegFormer.txt` and applies to those sources.
- [OpenMMLab MMSegmentation v0.11.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.11.0),
  including dataset, pipeline, runner integration, and segmentation utilities.
  Its Apache-2.0 license is retained at `LICENSES/MMSegmentation-Apache-2.0.txt`.

Copyright and license headers in retained source files are preserved. Dependencies
such as PyTorch, MMCV, timm, NumPy, and Pillow retain their respective licenses.
Dataset and pretrained-model terms are separate from code licenses.

No new blanket license is applied to author modifications or trained weights by this
private staging package. The owner can choose their release terms before making it public.
