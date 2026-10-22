import torch
import torch.nn as nn


class ClassPresenceBranch(nn.Module):
    """Image-level multi-label class presence branch.

    Input feature shape: [B, C, H, W]
    Output logits shape: [B, num_classes]
    """

    def __init__(self,
                 channels,
                 num_classes,
                 mlp_type='linear',
                 dropout=0.0):
        super(ClassPresenceBranch, self).__init__()
        if mlp_type not in ('linear', 'mlp'):
            raise ValueError(
                'presence_mlp_type must be "linear" or "mlp", but got {}.'.
                format(mlp_type))

        self.pool = nn.AdaptiveAvgPool2d(1)
        if mlp_type == 'linear':
            self.classifier = nn.Linear(channels, num_classes)
        else:
            hidden_channels = max(channels // 2, 1)
            self.classifier = nn.Sequential(
                nn.Linear(channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes))

    def forward(self, x):
        assert x.dim() == 4, (
            'presence source feature must be [B, C, H, W], got {}'.format(
                tuple(x.shape)))
        pooled = self.pool(x).flatten(1)
        assert pooled.dim() == 2
        return self.classifier(pooled)


def build_presence_labels(seg_label, num_classes, ignore_index=255,
                          min_pixels=10):
    """Build image-level class presence labels from segmentation masks.

    Args:
        seg_label (Tensor): [B, H, W] or [B, 1, H, W].
        num_classes (int): Number of semantic classes.
        ignore_index (int): Label ignored by segmentation loss.
        min_pixels (int): Minimum valid pixels for a class to be present.

    Returns:
        Tensor: Float labels with shape [B, num_classes].
    """
    if seg_label.dim() == 4:
        seg_label = seg_label.squeeze(1)
    assert seg_label.dim() == 3, (
        'gt_semantic_seg must be [B, H, W] or [B, 1, H, W], got {}'.format(
            tuple(seg_label.shape)))

    bsz = seg_label.size(0)
    presence = torch.zeros(
        (bsz, num_classes), device=seg_label.device, dtype=torch.float32)
    for bidx in range(bsz):
        label_one = seg_label[bidx]
        valid = (label_one != ignore_index) & (label_one >= 0)
        for cls_idx in range(num_classes):
            count = ((label_one == cls_idx) & valid).sum()
            if count >= min_pixels:
                presence[bidx, cls_idx] = 1.0
    return presence
