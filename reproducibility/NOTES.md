# Provenance and limits

## Main models

- SUIM: archived best checkpoint at training iteration 100000, not the final 160000
  checkpoint. Its metadata may store `iter=99999` because of the runner's counter.
- DUT: the main checkpoint underlying 72.3806% validation mIoU; its metadata stores
  `iter=31999`. This is not the older 28000-iteration checkpoint used for timing.
- Only these main-model weights are distributed. State-dict loading remains strict.

## Portability changes

The public model name is MARE-Seg. `MARESegHead` is a registry alias for the historical
`PrAdaULPSegFormerHead`, preserving all tensor names. Unrelated registered backbones,
heads, and losses have been removed from package imports. The used MLP, SegFormer
decoder, and cross-entropy definitions were extracted unchanged from the source;
the main MARE-Seg heads and MiT backbone retain their original implementations.

Machine-specific data/prototype paths were replaced with repository-relative paths.
Automatic resume and SwanLab cloud logging are disabled. Exact split IDs are enforced.
The historical SUIM IAAImageReplace step was a no-op because it preceded image loading
and queried a filename that did not exist yet; that step is omitted from the portable
configuration. The reported score is not evidence of IAA or language-model augmentation.

## Initialization and training

The archived SUIM 6x256 initialization matrix carries a dummy marker; its generating
algorithm/seed was not recorded. The DUT 5x256 initialization records random class
embeddings and seed 42, but its sampling distribution was not recorded. Exact numeric
initializers are provided. They are trainable class parameters, not CLIP text features.

The spatial gate uses uncertainty, boundary variation, and foreground probability.
The archived configuration detaches its inputs and output; the segmentation loss
does not train its convolution through this path. Historical prototypes update after
the current batch's prediction/loss computation and stay fixed at inference.

The DUT embedded configuration has lr=3e-4, whereas the current manuscript says 6e-4.
The released configuration follows the checkpoint. Its historical training seed is
not available. SUIM uses lr=6e-5, seed 42, 160k iterations, and best validation selection.
DUT uses 40k configured iterations and the released 32k checkpoint. Fresh training
is supported, but an exact historical trajectory or new multi-seed result is not claimed.

## Evaluation

Single-scale whole-image prediction uses each main model's embedded test pipeline,
including AlignedResize, and restores outputs to annotation size. The exact class
order, indexed masks, and full validation split must match. Reported checkpoint
reevaluations are separate from literature-reported values and independently trained
ablation groups. Small floating-point differences across devices are possible.

`historical_evaluation_environment.json` records the prior GPU evaluation environment.
`historical_training_requirements.txt` is retained for audit, not as a recommended
install command: it includes old wheel URLs and conflicting OpenCV packages.
Use the root README and requirements.txt for the prepared environment instead.
