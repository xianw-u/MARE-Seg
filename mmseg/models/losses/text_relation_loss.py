"""Training-only relation distillation for class prototype embeddings."""

import math

import torch
import torch.nn.functional as F


def build_cosine_relation_matrix(prototypes, eps=1e-6):
    """Build a cosine-similarity matrix from a finite ``[K, D]`` tensor."""
    if not isinstance(prototypes, torch.Tensor):
        raise TypeError('prototypes must be a torch.Tensor.')
    if prototypes.dim() != 2:
        raise ValueError(
            'prototypes must be two-dimensional [K, D], got {}.'.format(
                tuple(prototypes.shape)))
    if prototypes.size(0) < 2:
        raise ValueError('Relation distillation requires at least two classes.')
    if prototypes.size(1) < 1:
        raise ValueError('Prototype feature dimension must be positive.')
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError('eps must be finite and greater than zero.')
    values = prototypes.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError('Prototypes contain NaN or Inf.')
    norms = values.norm(p=2, dim=-1)
    if bool(norms.le(float(eps)).any()):
        raise ValueError(
            'Every prototype norm must be greater than eps ({}).'.format(
                float(eps)))
    normalized = F.normalize(values, p=2, dim=-1, eps=float(eps))
    return normalized @ normalized.t()


def build_cosine_relation_vector(prototypes, eps=1e-6):
    """Return each unordered class-pair cosine exactly once."""
    matrix = build_cosine_relation_matrix(prototypes, eps=eps)
    indices = torch.triu_indices(
        matrix.size(0), matrix.size(1), offset=1, device=matrix.device)
    return matrix[indices[0], indices[1]]


def standardize_relation_vector(relation, eps=1e-6, name='relation'):
    """Standardize a relation vector using population standard deviation."""
    if not isinstance(relation, torch.Tensor):
        raise TypeError('{} must be a torch.Tensor.'.format(name))
    if relation.dim() != 1 or relation.numel() < 1:
        raise ValueError('{} must be a non-empty 1D tensor.'.format(name))
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError('eps must be finite and greater than zero.')
    values = relation.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError('{} contains NaN or Inf.'.format(name))
    std = values.std(unbiased=False)
    if not bool(torch.isfinite(std)) or float(std.detach().item()) <= float(eps):
        raise ValueError(
            '{} population standard deviation ({}) is not greater than eps '
            '({}); relation geometry is degenerate.'.format(
                name, float(std.detach().item()), float(eps)))
    return (values - values.mean()) / std.clamp_min(float(eps))


def compute_text_relation_loss(visual_embeddings,
                               text_relation_target,
                               eps=1e-6,
                               smooth_l1_beta=1.0):
    """Match standardized visual relations to a detached text target."""
    beta = float(smooth_l1_beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError('smooth_l1_beta must be finite and non-negative.')
    visual_relation = build_cosine_relation_vector(
        visual_embeddings, eps=eps)
    visual_standardized = standardize_relation_vector(
        visual_relation, eps=eps, name='visual relation')
    if not isinstance(text_relation_target, torch.Tensor):
        raise TypeError('text_relation_target must be a torch.Tensor.')
    if text_relation_target.dim() != 1:
        raise ValueError('text_relation_target must be one-dimensional.')
    if visual_standardized.shape != text_relation_target.shape:
        raise ValueError(
            'Visual relation target shape mismatch: visual {}, text {}.'.format(
                tuple(visual_standardized.shape),
                tuple(text_relation_target.shape)))
    target = text_relation_target.detach().to(
        device=visual_standardized.device, dtype=visual_standardized.dtype)
    if not bool(torch.isfinite(target).all()):
        raise ValueError('text_relation_target contains NaN or Inf.')
    loss = F.smooth_l1_loss(
        visual_standardized, target, beta=beta, reduction='mean')
    return loss, visual_relation, visual_standardized


def relation_pearson_correlation(first, second):
    """Return Pearson correlation for two standardized relation vectors."""
    if first.shape != second.shape or first.dim() != 1:
        raise ValueError('Pearson inputs must be equal-shaped 1D tensors.')
    if first.numel() < 1:
        raise ValueError('Pearson inputs must not be empty.')
    return (first.float() * second.float()).mean()
