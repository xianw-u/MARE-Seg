"""Online visual class prototypes for the ULP-SegFormer baseline."""

import math
import os
import hashlib
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, get_dist_info

from mmseg.models.builder import HEADS
from mmseg.ops import resize
from ..losses import (accuracy, build_cosine_relation_matrix,
                      build_cosine_relation_vector,
                      compute_text_relation_loss,
                      relation_pearson_correlation,
                      standardize_relation_vector)
from .ulp_segformer_head import ULPSegFormerHead


@HEADS.register_module()
class PrAdaULPSegFormerHead(ULPSegFormerHead):
    """ULP-SegFormer with one online EMA visual prototype per class.

    The original ULP output is computed by :class:`ULPSegFormerHead` without
    modification.  Once all prototypes are initialized and warm-up has
    finished, cosine-similarity logits from the historical prototype bank are
    added as a parallel residual.  During training the current batch updates
    the bank only *after* its logits and losses have been computed.
    """

    def __init__(self,
                 feature_strides,
                 use_visual_prototype=True,
                 use_online_visual_prototypes=None,
                 num_visual_prototypes=1,
                 prototype_momentum=0.99,
                 prototype_warmup_iters=1000,
                 prototype_min_pixels=10,
                 prototype_beta=0.05,
                 learnable_prototype_beta=True,
                 prototype_logit_scale=1.0,
                 prototype_temperature=None,
                 use_prototype_aux_loss=True,
                 prototype_loss_weight=0.1,
                 strict_prototype_inference=True,
                 use_text_relation_distillation=None,
                 text_relation_prototype_path=None,
                 text_relation_loss_weight=0.0,
                 text_relation_warmup_iters=5000,
                 text_relation_decay_start_iter=80000,
                 text_relation_end_iter=120000,
                 text_relation_eps=1e-6,
                 text_relation_smooth_l1_beta=1.0,
                 text_relation_expected_dataset_id=None,
                 text_relation_expected_class_names=None,
                 text_relation_expected_class_ids=None,
                 use_direct_real_text_logits=False,
                 use_agreement_gate=False,
                 use_confusion_loss=False,
                 use_class_specific_loss=False,
                 use_uncertainty_dual_prototypes=False,
                 use_stable_hard_dual_prototypes=False,
                 use_weak_sampling_for_stable=False,
                 use_anchored_hard_correction=False,
                 stable_uncertainty_quantile=0.30,
                 noise_uncertainty_quantile=0.90,
                 prototype_quantile_min_pixels=20,
                 gt_boundary_width=1,
                 hard_use_gt_boundary=True,
                 hard_use_moderate_uncertainty=True,
                 hard_gate_uncertainty_weight=1.0,
                 hard_gate_boundary_weight=0.5,
                 exclude_highest_uncertainty=True,
                 require_correct_for_stable=True,
                 normalize_prototypes=True,
                 stable_prototype_momentum=None,
                 hard_prototype_momentum=None,
                 hard_min_gt_probability=0.0,
                 hard_uncertainty_lower_quantile=None,
                 hard_uncertainty_upper_quantile=None,
                 exclude_top_uncertain_quantile=None,
                 hard_boundary_threshold=0.0,
                 detach_dual_gate_inputs=False,
                 prototype_start_iter=0,
                 hard_prototype_update_start_iter=0,
                 hard_correction_start_iter=0,
                 hard_correction_end_iter=1,
                 prototype_beta_final=None,
                 hard_correction_scale_final=1.0,
                 prototype_aux_loss_weight_final=None,
                 initialize_random_class_embeddings=False,
                 prototype_eps=1e-6,
                 **kwargs):
        if use_text_relation_distillation is None:
            # Preserve the one historical local relation config that enabled
            # the feature by positive weight before the explicit switch was
            # introduced. New configs must set the switch directly.
            use_text_relation_distillation = (
                float(text_relation_loss_weight) > 0.0)
            if use_text_relation_distillation:
                warnings.warn(
                    'Implicit text-relation enablement from a positive loss '
                    'weight is deprecated; set '
                    'use_text_relation_distillation=True explicitly.',
                    DeprecationWarning)
        if int(num_visual_prototypes) != 1:
            raise ValueError(
                'PrAdaULPSegFormerHead implements exactly one visual '
                'prototype per class; use TextAnchoredMultiPrototypeULPHead '
                'for multi-prototype ablations.')
        if not 0.0 <= float(prototype_momentum) < 1.0:
            raise ValueError('prototype_momentum must be in [0, 1).')
        if int(prototype_warmup_iters) < 0:
            raise ValueError('prototype_warmup_iters must be non-negative.')
        if int(prototype_min_pixels) < 1:
            raise ValueError('prototype_min_pixels must be at least 1.')
        if float(prototype_beta) < 0.0:
            raise ValueError('prototype_beta must be non-negative.')
        if bool(learnable_prototype_beta) and float(prototype_beta) == 0.0:
            raise ValueError(
                'A learnable prototype_beta must be positive for finite '
                'inverse-softplus initialization.')
        if prototype_temperature is not None:
            if (float(prototype_logit_scale) != 1.0 and float(
                    prototype_logit_scale) != float(prototype_temperature)):
                raise ValueError(
                    'Set only prototype_logit_scale; prototype_temperature is '
                    'a deprecated compatibility alias.')
            warnings.warn(
                'prototype_temperature is deprecated; use '
                'prototype_logit_scale instead.', DeprecationWarning)
            prototype_logit_scale = prototype_temperature
        if not math.isfinite(float(prototype_logit_scale)) or float(
                prototype_logit_scale) <= 0.0:
            raise ValueError('prototype_logit_scale must be finite and > 0.')
        if float(prototype_loss_weight) < 0.0:
            raise ValueError('prototype_loss_weight must be non-negative.')
        if float(text_relation_loss_weight) < 0.0:
            raise ValueError(
                'text_relation_loss_weight must be non-negative.')
        relation_schedule = (
            int(text_relation_warmup_iters),
            int(text_relation_decay_start_iter),
            int(text_relation_end_iter))
        if relation_schedule[0] < 0:
            raise ValueError(
                'text_relation_warmup_iters must be non-negative.')
        if not (relation_schedule[0] <= relation_schedule[1]
                < relation_schedule[2]):
            raise ValueError(
                'Text-relation schedule must satisfy warmup_iters <= '
                'decay_start_iter < end_iter, got {}.'.format(
                    relation_schedule))
        if (not math.isfinite(float(text_relation_eps))
                or float(text_relation_eps) <= 0.0):
            raise ValueError('text_relation_eps must be finite and > 0.')
        if (not math.isfinite(float(text_relation_smooth_l1_beta))
                or float(text_relation_smooth_l1_beta) < 0.0):
            raise ValueError(
                'text_relation_smooth_l1_beta must be finite and >= 0.')
        if (bool(use_text_relation_distillation)
                and not text_relation_prototype_path):
            raise ValueError(
                'text_relation_prototype_path is required when the text '
                'relation loss is enabled.')
        if bool(use_text_relation_distillation) and float(
                text_relation_loss_weight) <= 0.0:
            raise ValueError(
                'text_relation_loss_weight must be positive when relation '
                'distillation is enabled.')
        if bool(use_direct_real_text_logits):
            raise ValueError(
                'PrAdaULPSegFormerHead does not support direct real-text '
                'logits. The prediction prototype is the existing learnable '
                'visual class embedding.')
        if use_agreement_gate or use_confusion_loss or use_class_specific_loss:
            raise ValueError(
                'Agreement, confusion, and class-specific losses are not part '
                'of the single-prototype PrAda-ULP experiment.')

        stable_uncertainty_quantile = float(stable_uncertainty_quantile)
        noise_uncertainty_quantile = float(noise_uncertainty_quantile)
        if not (0.0 <= stable_uncertainty_quantile
                < noise_uncertainty_quantile <= 1.0):
            raise ValueError(
                'Dual-prototype quantiles must satisfy 0 <= stable < noise '
                '<= 1.')
        if int(prototype_quantile_min_pixels) < 1:
            raise ValueError(
                'prototype_quantile_min_pixels must be at least 1.')
        if int(gt_boundary_width) < 1:
            raise ValueError('gt_boundary_width must be at least 1.')
        if float(hard_gate_uncertainty_weight) < 0.0:
            raise ValueError(
                'hard_gate_uncertainty_weight must be non-negative.')
        if float(hard_gate_boundary_weight) < 0.0:
            raise ValueError(
                'hard_gate_boundary_weight must be non-negative.')

        anchored_dual = bool(use_stable_hard_dual_prototypes)
        if anchored_dual:
            if not bool(use_uncertainty_dual_prototypes):
                raise ValueError(
                    'use_stable_hard_dual_prototypes requires '
                    'use_uncertainty_dual_prototypes=True.')
            if not bool(use_weak_sampling_for_stable):
                raise ValueError(
                    'The anchored experiment requires '
                    'use_weak_sampling_for_stable=True.')
            if not bool(use_anchored_hard_correction):
                raise ValueError(
                    'The anchored experiment requires '
                    'use_anchored_hard_correction=True.')
        stable_prototype_momentum = (
            prototype_momentum if stable_prototype_momentum is None else
            stable_prototype_momentum)
        hard_prototype_momentum = (
            prototype_momentum if hard_prototype_momentum is None else
            hard_prototype_momentum)
        for name, value in (
                ('stable_prototype_momentum', stable_prototype_momentum),
                ('hard_prototype_momentum', hard_prototype_momentum)):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError('{} must be in [0, 1).'.format(name))
        hard_uncertainty_lower_quantile = (
            stable_uncertainty_quantile
            if hard_uncertainty_lower_quantile is None else
            float(hard_uncertainty_lower_quantile))
        hard_uncertainty_upper_quantile = (
            noise_uncertainty_quantile
            if hard_uncertainty_upper_quantile is None else
            float(hard_uncertainty_upper_quantile))
        exclude_top_uncertain_quantile = (
            1.0 - hard_uncertainty_upper_quantile
            if exclude_top_uncertain_quantile is None else
            float(exclude_top_uncertain_quantile))
        if not (0.0 <= hard_uncertainty_lower_quantile <
                hard_uncertainty_upper_quantile <= 1.0):
            raise ValueError(
                'Hard uncertainty quantiles must satisfy 0 <= lower < '
                'upper <= 1.')
        if not 0.0 <= exclude_top_uncertain_quantile < 1.0:
            raise ValueError(
                'exclude_top_uncertain_quantile must be in [0, 1).')
        expected_upper = 1.0 - exclude_top_uncertain_quantile
        if anchored_dual and not math.isclose(
                hard_uncertainty_upper_quantile, expected_upper,
                rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                'hard_uncertainty_upper_quantile must equal '
                '1 - exclude_top_uncertain_quantile.')
        if not 0.0 <= float(hard_min_gt_probability) <= 1.0:
            raise ValueError('hard_min_gt_probability must be in [0, 1].')
        if not math.isfinite(float(hard_boundary_threshold)) or float(
                hard_boundary_threshold) < 0.0:
            raise ValueError(
                'hard_boundary_threshold must be finite and non-negative.')
        schedule = (int(prototype_start_iter),
                    int(hard_prototype_update_start_iter),
                    int(hard_correction_start_iter),
                    int(hard_correction_end_iter))
        if schedule[0] < 0 or not (
                schedule[0] <= schedule[1] <= schedule[2] < schedule[3]):
            raise ValueError(
                'Anchored schedule must satisfy 0 <= prototype_start_iter '
                '<= hard_prototype_update_start_iter <= '
                'hard_correction_start_iter < hard_correction_end_iter.')
        prototype_beta_final = (
            prototype_beta if prototype_beta_final is None else
            prototype_beta_final)
        prototype_aux_loss_weight_final = (
            prototype_loss_weight if prototype_aux_loss_weight_final is None
            else prototype_aux_loss_weight_final)
        for name, value in (
                ('prototype_beta_final', prototype_beta_final),
                ('hard_correction_scale_final',
                 hard_correction_scale_final),
                ('prototype_aux_loss_weight_final',
                 prototype_aux_loss_weight_final)):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError('{} must be finite and non-negative.'.format(
                    name))

        if use_online_visual_prototypes is not None:
            if (use_visual_prototype is not True
                    and bool(use_visual_prototype) != bool(
                        use_online_visual_prototypes)):
                raise ValueError(
                    'Conflicting use_visual_prototype and '
                    'use_online_visual_prototypes values.')
            use_visual_prototype = use_online_visual_prototypes
        self.use_visual_prototype = bool(use_visual_prototype)
        self.use_online_visual_prototypes = self.use_visual_prototype
        self.num_visual_prototypes = 1
        self.prototype_momentum = float(prototype_momentum)
        self.prototype_warmup_iters = int(prototype_warmup_iters)
        self.prototype_min_pixels = int(prototype_min_pixels)
        self.learnable_prototype_beta = bool(learnable_prototype_beta)
        self.prototype_logit_scale = float(prototype_logit_scale)
        self.use_prototype_aux_loss = bool(use_prototype_aux_loss)
        self.prototype_loss_weight = float(prototype_loss_weight)
        self.strict_prototype_inference = bool(strict_prototype_inference)
        self.use_text_relation_distillation = bool(
            use_text_relation_distillation)
        self.use_direct_real_text_logits = False
        self.text_relation_applies_to = 'learnable_visual_class_embeddings'
        self.text_relation_prototype_path = text_relation_prototype_path
        self.text_relation_loss_weight = float(text_relation_loss_weight)
        self.text_relation_warmup_iters = relation_schedule[0]
        self.text_relation_decay_start_iter = relation_schedule[1]
        self.text_relation_end_iter = relation_schedule[2]
        self.text_relation_eps = float(text_relation_eps)
        self.text_relation_smooth_l1_beta = float(
            text_relation_smooth_l1_beta)
        self.text_relation_expected_dataset_id = (
            text_relation_expected_dataset_id)
        self.text_relation_expected_class_names = (
            list(text_relation_expected_class_names)
            if text_relation_expected_class_names is not None else None)
        self.text_relation_expected_class_ids = (
            list(text_relation_expected_class_ids)
            if text_relation_expected_class_ids is not None else None)
        self.use_agreement_gate = False
        self.use_confusion_loss = False
        self.use_class_specific_loss = False
        self.use_uncertainty_dual_prototypes = bool(
            use_uncertainty_dual_prototypes)
        self.use_stable_hard_dual_prototypes = anchored_dual
        self.use_weak_sampling_for_stable = bool(
            use_weak_sampling_for_stable)
        self.use_anchored_hard_correction = bool(
            use_anchored_hard_correction)
        self.stable_uncertainty_quantile = stable_uncertainty_quantile
        self.noise_uncertainty_quantile = noise_uncertainty_quantile
        self.prototype_quantile_min_pixels = int(
            prototype_quantile_min_pixels)
        self.gt_boundary_width = int(gt_boundary_width)
        self.hard_use_gt_boundary = bool(hard_use_gt_boundary)
        self.hard_use_moderate_uncertainty = bool(
            hard_use_moderate_uncertainty)
        self.hard_gate_uncertainty_weight = float(
            hard_gate_uncertainty_weight)
        self.hard_gate_boundary_weight = float(hard_gate_boundary_weight)
        self.exclude_highest_uncertainty = bool(
            exclude_highest_uncertainty)
        self.require_correct_for_stable = bool(require_correct_for_stable)
        self.normalize_prototypes = bool(normalize_prototypes)
        self.stable_prototype_momentum = float(stable_prototype_momentum)
        self.hard_prototype_momentum = float(hard_prototype_momentum)
        self.hard_min_gt_probability = float(hard_min_gt_probability)
        self.hard_uncertainty_lower_quantile = float(
            hard_uncertainty_lower_quantile)
        self.hard_uncertainty_upper_quantile = float(
            hard_uncertainty_upper_quantile)
        self.exclude_top_uncertain_quantile = float(
            exclude_top_uncertain_quantile)
        self.hard_boundary_threshold = float(hard_boundary_threshold)
        self.detach_dual_gate_inputs = bool(detach_dual_gate_inputs)
        self.prototype_start_iter = schedule[0]
        self.hard_prototype_update_start_iter = schedule[1]
        self.hard_correction_start_iter = schedule[2]
        self.hard_correction_end_iter = schedule[3]
        self.prototype_beta_final = float(prototype_beta_final)
        self.hard_correction_scale_final = float(
            hard_correction_scale_final)
        self.prototype_aux_loss_weight_final = float(
            prototype_aux_loss_weight_final)
        self.initialize_random_class_embeddings = bool(
            initialize_random_class_embeddings)
        self.prototype_eps = float(prototype_eps)

        super(PrAdaULPSegFormerHead, self).__init__(
            feature_strides=feature_strides, **kwargs)

        self.register_buffer(
            'visual_prototypes',
            torch.zeros(self.num_classes, self.decoder_channels))
        self.register_buffer(
            'prototype_initialized',
            torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer(
            'prototype_update_count',
            torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer(
            'prototype_step', torch.zeros((), dtype=torch.long))

        # Register dual-prototype state only for the opt-in experiment.  Thus
        # old configurations retain their exact state_dict contract and old
        # checkpoints continue to load strictly.
        if self.use_uncertainty_dual_prototypes:
            prototype_shape = (self.num_classes, self.decoder_channels)
            self.register_buffer(
                'stable_prototypes', torch.zeros(*prototype_shape))
            self.register_buffer(
                'hard_prototypes', torch.zeros(*prototype_shape))
            for name in ('stable_initialized', 'hard_initialized'):
                self.register_buffer(
                    name, torch.zeros(self.num_classes, dtype=torch.bool))
            for name in ('stable_update_count', 'hard_update_count',
                         'stable_pixel_count', 'hard_pixel_count'):
                self.register_buffer(
                    name, torch.zeros(self.num_classes, dtype=torch.long))
            for name in ('last_stable_logit_mean',
                         'last_stable_logit_std',
                         'last_hard_logit_mean',
                         'last_hard_logit_std'):
                self.register_buffer(
                    name, torch.zeros(self.num_classes, dtype=torch.float32))
            self.register_buffer(
                'dual_prototype_step', torch.zeros((), dtype=torch.long))
            if self.use_stable_hard_dual_prototypes:
                for name in ('last_stable_selected_pixels',
                             'last_hard_selected_pixels',
                             'last_discarded_top_uncertain_pixels'):
                    self.register_buffer(
                        name,
                        torch.zeros(self.num_classes, dtype=torch.long))
                for name in ('last_hard_gate_mean', 'last_hard_gate_p50',
                             'last_hard_gate_p90',
                             'last_hard_gate_saturation_ratio',
                             'last_prototype_residual_ratio',
                             'last_hard_correction_ratio'):
                    self.register_buffer(
                        name, torch.zeros((), dtype=torch.float32))

        self._text_relation_metadata = None
        if self.use_text_relation_distillation:
            teacher = self._load_text_relation_teacher(
                self.text_relation_prototype_path)
            teacher_relation_raw = build_cosine_relation_vector(
                teacher, eps=self.text_relation_eps)
            teacher_relation = standardize_relation_vector(
                teacher_relation_raw,
                eps=self.text_relation_eps,
                name='text relation').detach()
            teacher_relation_matrix = build_cosine_relation_matrix(
                teacher, eps=self.text_relation_eps).detach()
        else:
            teacher_relation_raw = torch.empty(0, dtype=torch.float32)
            teacher_relation = torch.empty(0, dtype=torch.float32)
            teacher_relation_matrix = torch.empty(0, dtype=torch.float32)
        # The offline teacher is training-only.  The anchored mode has no
        # teacher, but MMCV 1.x's runner checkpoint helper serializes even
        # PyTorch non-persistent buffers.  Persist its three empty placeholders
        # only in the new opt-in mode so runner checkpoints strict-load while
        # every historical state_dict contract remains unchanged.
        persist_empty_relation_placeholders = (
            self.use_stable_hard_dual_prototypes)
        self.register_buffer(
            'text_relation_target', teacher_relation,
            persistent=persist_empty_relation_placeholders)
        self.register_buffer(
            'text_relation_raw_target',
            teacher_relation_raw.detach(),
            persistent=persist_empty_relation_placeholders)
        self.register_buffer(
            'text_relation_target_matrix',
            teacher_relation_matrix,
            persistent=persist_empty_relation_placeholders)

        if self.learnable_prototype_beta:
            self.prototype_beta_raw = nn.Parameter(
                self._inverse_softplus_exact(prototype_beta))
            self.register_buffer('prototype_beta_fixed', None)
        else:
            self.register_parameter('prototype_beta_raw', None)
            self.register_buffer(
                'prototype_beta_fixed',
                torch.tensor(float(prototype_beta), dtype=torch.float32))
        self._latest_logit_stats = {}
        self._latest_gate_maps = {}
        self._latest_dual_diagnostics = {}
        self._latest_visual_logits = None
        self._latest_dual_diagnostics = {}
        self._predicted_boundary_warning_emitted = False
        self._prototype_warning_emitted = False
        self._visual_logit_stats_hook = self.linear_pred.register_forward_hook(
            self._capture_visual_logits)
        self._log_prada_initialization()

    @staticmethod
    def _inverse_softplus_exact(value):
        """Stable scalar inverse of softplus for a strictly positive value."""
        value = float(value)
        return torch.tensor(
            value + math.log(-math.expm1(-value)), dtype=torch.float32)

    @property
    def beta_proto(self):
        """Return the non-negative prototype residual coefficient."""
        if self.learnable_prototype_beta:
            return F.softplus(self.prototype_beta_raw)
        return self.prototype_beta_fixed

    @staticmethod
    def _distributed():
        return (dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1)

    def prototypes_ready(self):
        """Whether the historical bank is safe to use for this batch."""
        return (int(self.prototype_step.item()) >= self.prototype_warmup_iters
                and bool(self.prototype_initialized.all().item()))

    def _log_prada_initialization(self):
        rank, _ = get_dist_info()
        if rank != 0:
            return
        print('[PrAdaULP] online prototype shape: {}'.format(
            tuple(self.visual_prototypes.shape)))
        print('[PrAdaULP] momentum/warmup/min_pixels: {}/{}/{}'.format(
            self.prototype_momentum, self.prototype_warmup_iters,
            self.prototype_min_pixels))
        print('[PrAdaULP] learnable beta / initial beta: {} / {:.8f}'.format(
            self.learnable_prototype_beta, float(self.beta_proto.item())))
        print('[PrAdaULP] prototype logit scale: {}'.format(
            self.prototype_logit_scale))
        print('[PrAdaULP] use_text_relation_distillation: {}'.format(
            self.use_text_relation_distillation))
        print('[PrAdaULP] use_direct_real_text_logits: {}'.format(
            self.use_direct_real_text_logits))
        print('[PrAdaULP] text_relation_applies_to: {}'.format(
            self.text_relation_applies_to))
        print('[PrAdaULP] text relation prototype path: {}'.format(
            self.text_relation_prototype_path))
        print('[PrAdaULP] class embedding source: {}'.format(
            'random learnable E (no file)' if
            self.initialize_random_class_embeddings else
            'legacy configured tensor'))
        print('[PrAdaULP] learnable E shape: {}'.format(
            tuple(self.text_prototypes.shape)))
        if self.use_uncertainty_dual_prototypes:
            print('[PrAdaULP-Dual] num_classes/prototype_dim: {}/{}'.format(
                self.num_classes, self.decoder_channels))
            print('[PrAdaULP-Dual] stable/hard shapes: {}/{}'.format(
                tuple(self.stable_prototypes.shape),
                tuple(self.hard_prototypes.shape)))
            print('[PrAdaULP-Dual] momentum/quantiles/min_pixels/'
                  'quantile_min_pixels: {}/{}/{}/{}/{}'.format(
                      self.prototype_momentum,
                      self.stable_uncertainty_quantile,
                      self.noise_uncertainty_quantile,
                      self.prototype_min_pixels,
                      self.prototype_quantile_min_pixels))
            print('[PrAdaULP-Dual] warmup/boundary_width: {}/{}'.format(
                self.prototype_warmup_iters, self.gt_boundary_width))
            print('[PrAdaULP-Dual] hard gate uncertainty/boundary weights: '
                  '{}/{}'.format(self.hard_gate_uncertainty_weight,
                                 self.hard_gate_boundary_weight))
            print('[PrAdaULP-Dual] predicted boundary available: {}'.format(
                self.use_boundary_gate))
            print('[PrAdaULP-Dual] beta/logit_scale: {}/{}'.format(
                float(self.beta_proto.item()), self.prototype_logit_scale))
            print('[PrAdaULP-Dual] text relation/direct real-text: '
                  '{}/{}'.format(self.use_text_relation_distillation,
                                 self.use_direct_real_text_logits))
        if self.use_text_relation_distillation:
            metadata = self._text_relation_metadata
            raw = self.text_relation_raw_target
            target = self.text_relation_target
            print('[PrAdaULP] text teacher is_dummy/dataset/encoder: '
                  '{}/{}/{}'.format(metadata['is_dummy'],
                                    metadata['dataset_id'],
                                    metadata['encoder_name']))
            print('[PrAdaULP] text teacher class_names: {}'.format(
                metadata['class_names']))
            print('[PrAdaULP] relation K/D_text/D_visual/pairs: '
                  '{}/{}/{}/{}'.format(
                      self.num_classes, metadata['text_dim'],
                      self.decoder_channels, raw.numel()))
            print('[PrAdaULP] raw text relation mean/std/min/max: '
                  '{:.8f}/{:.8f}/{:.8f}/{:.8f}'.format(
                      float(raw.mean()), float(raw.std(unbiased=False)),
                      float(raw.min()), float(raw.max())))
            print('[PrAdaULP] standardized text relation mean/std: '
                  '{:.8f}/{:.8f}'.format(
                      float(target.mean()),
                      float(target.std(unbiased=False))))
            print('[PrAdaULP] text teacher SHA256: {}'.format(
                metadata['sha256']))
            print('[PrAdaULP] text relation weight/warmup/decay/end/eps/beta: '
                  '{}/{}/{}/{}/{}/{}'.format(
                      self.text_relation_loss_weight,
                      self.text_relation_warmup_iters,
                      self.text_relation_decay_start_iter,
                      self.text_relation_end_iter,
                      self.text_relation_eps,
                      self.text_relation_smooth_l1_beta))

    def _load_text_prototypes(self, text_prototype_path):
        """Build E directly for the text-free experiment, or use legacy IO.

        ``text_prototypes`` is retained as the state_dict name for backward
        compatibility.  In this mode it is a randomly initialized learnable
        visual class embedding E, not a language/text prototype.
        """
        if not self.initialize_random_class_embeddings:
            return super(PrAdaULPSegFormerHead, self)._load_text_prototypes(
                text_prototype_path)
        embeddings = torch.randn(self.num_classes, self.decoder_channels)
        embeddings = F.normalize(
            embeddings, p=2, dim=1, eps=self.prototype_eps)
        return None, embeddings

    def _load_text_relation_teacher(self, path):
        """Load and strictly validate a real offline text-prototype teacher."""
        if not os.path.isfile(path):
            raise FileNotFoundError(
                'Text-relation prototype file does not exist: {}'.format(
                    path))
        obj = torch.load(path, map_location='cpu')
        if not isinstance(obj, dict) or 'text_prototypes' not in obj:
            raise ValueError(
                'Text-relation teacher must be a metadata dict containing '
                'text_prototypes.')
        if obj.get('is_dummy') is not False:
            raise ValueError(
                'Text-relation teacher must be real (is_dummy=False).')
        if obj.get('source_type') != 'real_text_encoder':
            raise ValueError(
                'Text-relation teacher must have '
                'source_type="real_text_encoder".')
        prototypes = obj['text_prototypes']
        if (not isinstance(prototypes, torch.Tensor)
                or prototypes.dim() != 2):
            raise ValueError(
                'Text-relation prototypes must be a 2D Tensor [K, D].')
        if prototypes.size(0) != self.num_classes:
            raise ValueError(
                'Text-relation teacher has {} classes, expected {}.'.format(
                    prototypes.size(0), self.num_classes))
        if not bool(torch.isfinite(prototypes).all()):
            raise ValueError(
                'Text-relation prototypes contain NaN or Inf.')
        if bool(prototypes.norm(dim=1).le(self.text_relation_eps).any()):
            raise ValueError(
                'Every text-relation prototype must have non-zero norm.')
        expected_names = self.text_relation_expected_class_names
        if expected_names is None:
            expected_names = self.expected_class_names
        if expected_names is None:
            raise ValueError(
                'Text-relation formal mode requires configured expected '
                'class names so label order can be verified.')
        if expected_names is not None:
            if list(obj.get('class_names', [])) != expected_names:
                raise ValueError(
                    'Text-relation teacher class order mismatch. Expected '
                    '{}, got {}.'.format(
                        expected_names, obj.get('class_names')))
        expected_dataset = self.text_relation_expected_dataset_id
        if expected_dataset is None:
            expected_dataset = self.expected_dataset_id
        if expected_dataset is None:
            raise ValueError(
                'Text-relation formal mode requires an expected dataset_id.')
        if (expected_dataset is not None
                and obj.get('dataset_id') != expected_dataset):
            raise ValueError(
                'Text-relation teacher dataset_id mismatch. Expected {}, '
                'got {}.'.format(expected_dataset, obj.get('dataset_id')))
        expected_ids = self.text_relation_expected_class_ids
        if expected_ids is None:
            expected_ids = list(range(self.num_classes))
        if list(obj.get('class_ids', [])) != expected_ids:
            raise ValueError(
                'Text-relation teacher class_ids mismatch. Expected {}, got '
                '{}.'.format(expected_ids, obj.get('class_ids')))
        self._text_relation_metadata = dict(
            is_dummy=obj['is_dummy'],
            dataset_id=obj['dataset_id'],
            encoder_name=obj.get('encoder_name', '<missing>'),
            class_names=list(obj['class_names']),
            text_dim=int(prototypes.size(1)),
            sha256=self._sha256(path))
        return prototypes.float()

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _text_relation_weight(self):
        """Piecewise-linear training-only relation-distillation weight."""
        step = int(self.prototype_step.item())
        maximum = self.text_relation_loss_weight
        if maximum <= 0.0 or step >= self.text_relation_end_iter:
            return 0.0
        if step < self.text_relation_warmup_iters:
            if self.text_relation_warmup_iters == 0:
                return maximum
            return maximum * step / self.text_relation_warmup_iters
        if step < self.text_relation_decay_start_iter:
            return maximum
        remaining = self.text_relation_end_iter - step
        duration = (self.text_relation_end_iter -
                    self.text_relation_decay_start_iter)
        return maximum * remaining / duration

    def _text_relation_loss(self):
        """Distill only class relations into the existing learnable vectors."""
        weight = self._text_relation_weight()
        # ``text_prototypes`` is legacy naming inherited from ULP. In this
        # experiment it is E: the learnable visual class embedding used by the
        # existing prediction branch, not the offline real-text teacher T.
        visual_embeddings = self._get_text_prototypes()
        raw, visual_relation, visual_standardized = (
            compute_text_relation_loss(
                visual_embeddings,
                self.text_relation_target,
                eps=self.text_relation_eps,
                smooth_l1_beta=self.text_relation_smooth_l1_beta))
        target = self.text_relation_target.detach().to(
            device=visual_standardized.device,
            dtype=visual_standardized.dtype)
        correlation = relation_pearson_correlation(
            visual_standardized, target)
        norms = visual_embeddings.float().norm(dim=-1)
        diagnostics = dict(
            text_rel_unweighted=raw.detach(),
            text_rel_weight=raw.new_tensor(weight),
            text_rel_visual_mean=visual_relation.detach().mean(),
            text_rel_visual_std=visual_relation.detach().std(unbiased=False),
            text_rel_target_mean=self.text_relation_raw_target.mean().detach(),
            text_rel_target_std=self.text_relation_raw_target.std(
                unbiased=False).detach(),
            text_rel_corr=correlation.detach(),
            text_rel_embedding_norm_mean=norms.detach().mean(),
            text_rel_embedding_norm_min=norms.detach().min(),
            text_rel_embedding_norm_max=norms.detach().max(),
            text_rel_iter=raw.new_tensor(float(self.prototype_step.item())))
        for class_index, norm in enumerate(norms.detach()):
            diagnostics['text_rel_embedding_norm_{}'.format(
                class_index)] = norm
        return weight * raw, diagnostics

    def _record_abs_statistics(self, name, logits):
        """Store detached absolute-logit mean/std for the current forward."""
        absolute = logits.detach().float().abs()
        self._latest_logit_stats['{}_abs_mean'.format(name)] = absolute.mean()
        self._latest_logit_stats['{}_abs_std'.format(name)] = absolute.std(
            unbiased=False)

    def _capture_visual_logits(self, module, inputs, output):
        self._record_abs_statistics('z_visual', output)
        # Used only for detached GT mask construction after current-batch
        # logits/losses have already been computed.
        self._latest_visual_logits = output.detach()

    def _compute_gate_maps(self, z_visual):
        """Expose the exact maps already computed by the inherited UBM."""
        gate_maps = super(
            PrAdaULPSegFormerHead, self)._compute_gate_maps(z_visual)
        self._latest_gate_maps = gate_maps
        return gate_maps

    def _compute_language_logits(self, decoder_feat):
        z_text = super(
            PrAdaULPSegFormerHead, self)._compute_language_logits(decoder_feat)
        self._record_abs_statistics('z_text', z_text)
        return z_text

    def _compute_prototype_logits(self, decoder_feat, prototypes):
        """Compute [B, K, Hf, Wf] cosine logits from historical prototypes."""
        if decoder_feat.dim() != 4:
            raise ValueError('decoder_feat must be [B, D, Hf, Wf].')
        if decoder_feat.size(1) != self.decoder_channels:
            raise ValueError(
                'Decoder feature has {} channels, expected {}.'.format(
                    decoder_feat.size(1), self.decoder_channels))
        feature_norm = F.normalize(
            decoder_feat, p=2, dim=1, eps=self.prototype_eps)
        prototype_norm = F.normalize(
            prototypes, p=2, dim=-1, eps=self.prototype_eps)
        z_proto = torch.einsum(
            'bdhw,kd->bkhw', feature_norm, prototype_norm)
        z_proto = self.prototype_logit_scale * z_proto
        expected = (decoder_feat.size(0), self.num_classes,
                    decoder_feat.size(2), decoder_feat.size(3))
        if tuple(z_proto.shape) != expected:
            raise RuntimeError(
                'Prototype logits have shape {}, expected {}.'.format(
                    tuple(z_proto.shape), expected))
        if not bool(torch.isfinite(z_proto).all()):
            raise RuntimeError('Prototype logits contain NaN or Inf.')
        self._record_abs_statistics('z_proto', z_proto)
        return z_proto

    def _compute_visual_prototype_logits(self,
                                         decoder_feat,
                                         prototypes=None):
        """Backward-compatible wrapper for earlier local tests/scripts."""
        if prototypes is None:
            prototypes = self.visual_prototypes.detach().clone()
        return self._compute_prototype_logits(decoder_feat, prototypes)

    @staticmethod
    def generate_gt_boundary(seg_label, ignore_index, boundary_width=1):
        """Generate a label boundary without one-hot tensors or padded edges."""
        if seg_label.dim() == 4:
            if seg_label.size(1) != 1:
                raise ValueError('4-D seg_label must have one channel.')
            seg_label = seg_label.squeeze(1)
        if seg_label.dim() != 3:
            raise ValueError('seg_label must be [B,H,W] or [B,1,H,W].')
        if int(boundary_width) < 1:
            raise ValueError('boundary_width must be at least 1.')
        labels = seg_label.long()
        valid = labels.ne(ignore_index)
        boundary = torch.zeros_like(valid)

        horizontal = (valid[:, :, :-1] & valid[:, :, 1:]
                      & labels[:, :, :-1].ne(labels[:, :, 1:]))
        boundary[:, :, :-1] |= horizontal
        boundary[:, :, 1:] |= horizontal
        vertical = (valid[:, :-1, :] & valid[:, 1:, :]
                    & labels[:, :-1, :].ne(labels[:, 1:, :]))
        boundary[:, :-1, :] |= vertical
        boundary[:, 1:, :] |= vertical

        if int(boundary_width) > 1:
            kernel = 2 * int(boundary_width) - 1
            boundary = F.max_pool2d(
                boundary.float().unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2).squeeze(1).bool()
        return boundary & valid

    def dual_prototypes_ready(self):
        """Whether at least one historical stable prototype may contribute."""
        return (int(self.dual_prototype_step.item()) >=
                self.prototype_warmup_iters
                and bool(self.stable_initialized.any().item()))

    @staticmethod
    def _linear_schedule(step, start, end, final_value):
        """Return a boundary-safe piecewise-linear scalar schedule."""
        step = int(step)
        start = int(start)
        end = int(end)
        final_value = float(final_value)
        if step <= start:
            return 0.0
        if step >= end:
            return final_value
        duration = end - start
        if duration <= 0:
            return final_value
        return final_value * float(step - start) / float(duration)

    def get_anchored_prototype_schedule(self, step=None):
        """Return the recoverable stable/hard schedule as Python floats."""
        if step is None:
            step = int(self.prototype_step.item())
        return dict(
            prototype_beta=self._linear_schedule(
                step, self.prototype_start_iter,
                self.hard_correction_start_iter,
                self.prototype_beta_final),
            hard_correction_scale=self._linear_schedule(
                step, self.hard_correction_start_iter,
                self.hard_correction_end_iter,
                self.hard_correction_scale_final),
            prototype_aux_weight=self._linear_schedule(
                step, self.hard_correction_start_iter,
                self.hard_correction_end_iter,
                self.prototype_aux_loss_weight_final))

    def _compute_hard_gate(self, uncertainty, predicted_boundary=None):
        """Select stable/hard scores using only base-prediction signals."""
        if uncertainty.dim() != 4 or uncertainty.size(1) != 1:
            raise ValueError('uncertainty must have shape [B,1,H,W].')
        dual_uncertainty = uncertainty
        dual_boundary = predicted_boundary
        if (self.use_stable_hard_dual_prototypes
                and self.detach_dual_gate_inputs):
            dual_uncertainty = uncertainty.detach()
            if dual_boundary is not None:
                dual_boundary = dual_boundary.detach()
        hard_gate = self.hard_gate_uncertainty_weight * dual_uncertainty
        if dual_boundary is not None:
            if (dual_boundary.dim() != 4 or dual_boundary.size(1) != 1
                    or dual_boundary.shape != uncertainty.shape):
                raise ValueError(
                    'predicted_boundary must match uncertainty [B,1,H,W].')
            hard_gate = (hard_gate + self.hard_gate_boundary_weight
                         * dual_boundary)
        elif (self.hard_gate_boundary_weight > 0.0
              and not self._predicted_boundary_warning_emitted):
            warnings.warn(
                'No predicted UBM boundary map is available; dual-prototype '
                'selection uses uncertainty only.', RuntimeWarning)
            self._predicted_boundary_warning_emitted = True
        return hard_gate.clamp(0.0, 1.0)

    def _compute_dual_prototype_logits(self, decoder_feat,
                                       stable_prototypes,
                                       hard_prototypes,
                                       stable_initialized,
                                       hard_initialized,
                                       uncertainty,
                                       predicted_boundary=None,
                                       hard_correction_scale=None):
        """Compute per-pixel stable/hard fusion from historical snapshots."""
        z_stable = self._compute_prototype_logits(
            decoder_feat, stable_prototypes)
        z_hard_raw = self._compute_prototype_logits(
            decoder_feat, hard_prototypes)
        stable_valid = stable_initialized.view(1, -1, 1, 1)
        hard_valid = hard_initialized.view(1, -1, 1, 1)
        # An unavailable hard prototype degrades to its stable counterpart.
        z_hard = torch.where(hard_valid, z_hard_raw, z_stable)
        hard_gate = self._compute_hard_gate(
            uncertainty, predicted_boundary=predicted_boundary)
        if hard_gate.shape != (decoder_feat.size(0), 1,
                               decoder_feat.size(2),
                               decoder_feat.size(3)):
            raise RuntimeError(
                'Hard gate has shape {}, expected [B,1,H,W].'.format(
                    tuple(hard_gate.shape)))
        if self.use_stable_hard_dual_prototypes:
            if hard_correction_scale is None:
                raise ValueError(
                    'hard_correction_scale is required for anchored fusion.')
            # Stable logits remain the anchor.  A hard prototype is allowed to
            # alter a class only when both historical class buffers exist.
            correction_valid = stable_valid & hard_valid
            hard_delta = torch.where(
                correction_valid, z_hard_raw - z_stable,
                torch.zeros_like(z_stable))
            z_dual = (z_stable + float(hard_correction_scale)
                      * hard_gate * hard_delta)
        else:
            z_dual = (1.0 - hard_gate) * z_stable + hard_gate * z_hard
        # Never let cosine against an all-zero, uninitialized stable buffer
        # enter classification or the auxiliary loss.
        z_stable = torch.where(stable_valid, z_stable,
                               torch.zeros_like(z_stable))
        z_hard = torch.where(stable_valid, z_hard,
                             torch.zeros_like(z_hard))
        z_dual = torch.where(stable_valid, z_dual,
                             torch.zeros_like(z_dual))
        self._latest_logit_stats['stable_score_std'] = (
            z_stable.detach().float().std(unbiased=False))
        self._latest_logit_stats['hard_score_std'] = (
            z_hard.detach().float().std(unbiased=False))
        self._latest_logit_stats['dual_score_std'] = (
            z_dual.detach().float().std(unbiased=False))
        with torch.no_grad():
            reduce_dims = (0, 2, 3)
            self.last_stable_logit_mean.copy_(
                z_stable.detach().float().mean(dim=reduce_dims))
            self.last_stable_logit_std.copy_(
                z_stable.detach().float().std(
                    dim=reduce_dims, unbiased=False))
            self.last_hard_logit_mean.copy_(
                z_hard.detach().float().mean(dim=reduce_dims))
            self.last_hard_logit_std.copy_(
                z_hard.detach().float().std(
                    dim=reduce_dims, unbiased=False))
        self._record_abs_statistics('z_proto', z_dual)
        return z_dual, z_stable, z_hard, hard_gate

    def _forward_prada_from_features(self, decoder_feat, return_aux=False):
        """Fuse the historical prototype residual with the exact ULP output."""
        self._latest_logit_stats = {}
        self._latest_gate_maps = {}
        self._latest_visual_logits = None
        z_8190, baseline_aux = super(
            PrAdaULPSegFormerHead, self)._forward_logits_from_features(
                decoder_feat, return_aux=True)

        if self.use_uncertainty_dual_prototypes:
            if 'uncertainty' not in self._latest_gate_maps:
                raise RuntimeError(
                    'The dual-prototype branch requires the uncertainty map '
                    'from the inherited UBM base prediction.')
            uncertainty = self._latest_gate_maps['uncertainty']
            predicted_boundary = self._latest_gate_maps.get('boundary')
            if self.use_stable_hard_dual_prototypes:
                schedule = self.get_anchored_prototype_schedule()
                # These are real copies, not views.  The current GT is not
                # observed until after logits and every loss are complete.
                stable_snapshot = self.stable_prototypes.detach().clone()
                hard_snapshot = self.hard_prototypes.detach().clone()
                stable_initialized = self.stable_initialized.detach().clone()
                hard_initialized = self.hard_initialized.detach().clone()
                z_proto, z_stable, z_hard, hard_gate = (
                    self._compute_dual_prototype_logits(
                        decoder_feat,
                        stable_snapshot,
                        hard_snapshot,
                        stable_initialized,
                        hard_initialized,
                        uncertainty,
                        predicted_boundary=predicted_boundary,
                        hard_correction_scale=schedule[
                            'hard_correction_scale']))
                effective_beta = schedule['prototype_beta']
                z_final = z_8190 + effective_beta * z_proto
                ready = (self.use_visual_prototype
                         and effective_beta > 0.0
                         and bool(stable_initialized.any().item()))
                with torch.no_grad():
                    eps = self.prototype_eps
                    base_norm = z_8190.detach().float().norm().clamp_min(eps)
                    proto_ratio = (
                        effective_beta * z_proto.detach().float()).norm(
                        ) / base_norm
                    correction_ratio = (
                        effective_beta *
                        (z_proto - z_stable).detach().float()).norm(
                        ) / base_norm
                    gate_float = hard_gate.detach().float().reshape(-1)
                    gate_mean = gate_float.mean()
                    gate_p50 = torch.quantile(gate_float, 0.50)
                    gate_p90 = torch.quantile(gate_float, 0.90)
                    gate_saturation = gate_float.ge(0.99).float().mean()
                    if self.training:
                        self.last_hard_gate_mean.copy_(gate_mean)
                        self.last_hard_gate_p50.copy_(gate_p50)
                        self.last_hard_gate_p90.copy_(gate_p90)
                        self.last_hard_gate_saturation_ratio.copy_(
                            gate_saturation)
                        self.last_prototype_residual_ratio.copy_(proto_ratio)
                        self.last_hard_correction_ratio.copy_(
                            correction_ratio)
                    self._latest_dual_diagnostics.update(dict(
                        effective_prototype_beta=z_8190.new_tensor(
                            effective_beta),
                        effective_hard_correction_scale=z_8190.new_tensor(
                            schedule['hard_correction_scale']),
                        effective_prototype_aux_weight=z_8190.new_tensor(
                            schedule['prototype_aux_weight']),
                        hard_gate_mean=gate_mean,
                        hard_gate_p50=gate_p50,
                        hard_gate_p90=gate_p90,
                        hard_gate_saturation_ratio=gate_saturation,
                        prototype_residual_ratio=proto_ratio,
                        hard_correction_ratio=correction_ratio))
                if return_aux:
                    baseline_aux = dict(baseline_aux)
                    baseline_aux.update(
                        z_proto=z_proto,
                        z_stable=z_stable,
                        z_hard=z_hard,
                        hard_gate=hard_gate,
                        predicted_boundary=predicted_boundary,
                        uncertainty=uncertainty,
                        prototype_ready=ready,
                        prototype_aux_ready=False,
                        stable_initialized_snapshot=stable_initialized,
                        effective_prototype_beta=effective_beta,
                        effective_hard_correction_scale=schedule[
                            'hard_correction_scale'],
                        effective_prototype_aux_weight=schedule[
                            'prototype_aux_weight'],
                        z_8190=z_8190)
                    return z_final, baseline_aux
                return z_final
            ready = self.use_visual_prototype and self.dual_prototypes_ready()
            z_proto = None
            z_stable = None
            z_hard = None
            hard_gate = self._compute_hard_gate(
                uncertainty, predicted_boundary=predicted_boundary)
            if ready:
                # Clone every historical input before any current-batch EMA
                # update.  These tensors cannot be modified in place later.
                stable_snapshot = self.stable_prototypes.detach().clone()
                hard_snapshot = self.hard_prototypes.detach().clone()
                stable_initialized = self.stable_initialized.detach().clone()
                hard_initialized = self.hard_initialized.detach().clone()
                z_proto, z_stable, z_hard, hard_gate = (
                    self._compute_dual_prototype_logits(
                        decoder_feat,
                        stable_snapshot,
                        hard_snapshot,
                        stable_initialized,
                        hard_initialized,
                        uncertainty,
                        predicted_boundary=predicted_boundary))
                # Preserve the actual 82.14 fusion: its single prototype was
                # an ungated residual on Z_8190.  Only its score is replaced.
                z_final = z_8190 + self.beta_proto * z_proto
            else:
                z_final = z_8190
            if return_aux:
                baseline_aux = dict(baseline_aux)
                baseline_aux.update(
                    z_proto=z_proto,
                    z_stable=z_stable,
                    z_hard=z_hard,
                    hard_gate=hard_gate,
                    uncertainty=uncertainty,
                    prototype_ready=ready,
                    prototype_aux_ready=(
                        ready and bool(self.stable_initialized.all().item())),
                    z_8190=z_8190)
                return z_final, baseline_aux
            return z_final

        ready = self.use_visual_prototype and self.prototypes_ready()
        if (not self.training and self.use_visual_prototype and not ready):
            message = (
                'Visual prototypes are not ready: initialized {}/{} classes '
                'at step {} (warm-up {}). Falling back to the unchanged ULP '
                'output.'.format(
                    int(self.prototype_initialized.sum().item()),
                    self.num_classes, int(self.prototype_step.item()),
                    self.prototype_warmup_iters))
            warmup_finished = (
                int(self.prototype_step.item()) >= self.prototype_warmup_iters)
            if self.strict_prototype_inference and warmup_finished:
                raise RuntimeError(message.replace(
                    ' Falling back to the unchanged ULP output.', ''))
            if not self._prototype_warning_emitted:
                warnings.warn(message, RuntimeWarning)
                self._prototype_warning_emitted = True

        z_proto = None
        if ready:
            # A real copy, not a view/reference: the following EMA update must
            # never change the historical prototypes used by this batch.
            historical_prototypes = (
                self.visual_prototypes.detach().clone())
            z_proto = self._compute_prototype_logits(
                decoder_feat, historical_prototypes)
            z_final = z_8190 + self.beta_proto * z_proto
        else:
            # Exact baseline behavior during warm-up or when disabled.
            z_final = z_8190

        if return_aux:
            baseline_aux = dict(baseline_aux)
            baseline_aux.update(
                z_proto=z_proto, prototype_ready=ready, z_8190=z_8190)
            return z_final, baseline_aux
        return z_final

    def forward(self, inputs):
        """Inference consumes only image features and checkpointed buffers."""
        return self._forward_prada_from_features(self._decode_features(inputs))

    @torch.no_grad()
    def _collect_batch_prototype_statistics(self, decoder_feat,
                                            gt_semantic_seg):
        """Return globally synchronized sums/counts of image-level vectors."""
        if gt_semantic_seg.dim() == 3:
            gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
        if gt_semantic_seg.dim() != 4 or gt_semantic_seg.size(1) != 1:
            raise ValueError(
                'gt_semantic_seg must be [B, 1, H, W] or [B, H, W].')
        if decoder_feat.size(0) != gt_semantic_seg.size(0):
            raise ValueError('Feature and GT batch sizes must match.')

        self._validate_segmentation_labels(gt_semantic_seg)

        gt_small = F.interpolate(
            gt_semantic_seg.float(),
            size=decoder_feat.shape[-2:],
            mode='nearest').long().squeeze(1)
        class_feature_sum = decoder_feat.new_zeros(
            self.num_classes, self.decoder_channels, dtype=torch.float32)
        class_sample_count = decoder_feat.new_zeros(
            self.num_classes, dtype=torch.float32)
        feature = decoder_feat.detach().float()

        for batch_id in range(feature.size(0)):
            for class_id in range(self.num_classes):
                mask = gt_small[batch_id].eq(class_id)
                if int(mask.sum().item()) < self.prototype_min_pixels:
                    continue
                vector = feature[batch_id, :, mask].mean(dim=1)
                norm = vector.norm(p=2)
                if (not bool(torch.isfinite(vector).all())
                        or float(norm.item()) <= self.prototype_eps):
                    continue
                vector = vector / norm
                class_feature_sum[class_id].add_(vector)
                class_sample_count[class_id].add_(1.0)

        if self._distributed():
            # Every image-class vector has equal weight, including across GPUs.
            dist.all_reduce(class_feature_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_sample_count, op=dist.ReduceOp.SUM)
        return class_feature_sum, class_sample_count

    @torch.no_grad()
    def _select_stable_pixels(self, class_mask, uncertainty, threshold,
                              correct, gt_boundary):
        """Shared reliable-interior selector for the anchored stable bank."""
        stable = (class_mask & correct & ~gt_boundary
                  & uncertainty.le(threshold))
        return stable

    @torch.no_grad()
    def _build_anchored_prototype_masks(self, uncertainty, z_visual,
                                        predicted_boundary,
                                        gt_semantic_seg):
        """Build conservative q30/q80 stable and hard pixel sets."""
        if predicted_boundary is None:
            raise RuntimeError(
                'Anchored hard selection requires the prediction-derived '
                'UBM boundary response.')
        if gt_semantic_seg.dim() == 3:
            gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
        if gt_semantic_seg.dim() != 4 or gt_semantic_seg.size(1) != 1:
            raise ValueError(
                'gt_semantic_seg must be [B,1,H,W] or [B,H,W].')
        self._validate_segmentation_labels(gt_semantic_seg)
        gt_small = F.interpolate(
            gt_semantic_seg.float(),
            size=uncertainty.shape[-2:],
            mode='nearest').long().squeeze(1)
        self._validate_segmentation_labels(gt_small)
        valid = gt_small.ne(self.ignore_index)
        gt_boundary = self.generate_gt_boundary(
            gt_small, self.ignore_index, self.gt_boundary_width)
        uncertainty_map = uncertainty.detach().float().squeeze(1)
        boundary_map = predicted_boundary.detach().float().squeeze(1)
        visual_probability = F.softmax(z_visual.detach().float(), dim=1)
        prediction = visual_probability.argmax(dim=1)
        correct = prediction.eq(gt_small)
        safe_gt = gt_small.clone()
        safe_gt[~valid] = 0
        p_gt = visual_probability.gather(
            1, safe_gt.unsqueeze(1)).squeeze(1)

        stable_masks = []
        hard_masks = []
        stable_counts = torch.zeros(
            self.num_classes, device=gt_small.device, dtype=torch.long)
        hard_counts = torch.zeros_like(stable_counts)
        discarded_counts = torch.zeros_like(stable_counts)
        predicted_hard_boundary = boundary_map.ge(
            self.hard_boundary_threshold)
        for class_id in range(self.num_classes):
            class_mask = valid & gt_small.eq(class_id)
            class_pixels = int(class_mask.sum().item())
            if class_pixels < self.prototype_quantile_min_pixels:
                empty = torch.zeros_like(class_mask)
                stable_masks.append(empty)
                hard_masks.append(empty)
                continue
            class_uncertainty = uncertainty_map[class_mask].float()
            q_lower = torch.quantile(
                class_uncertainty,
                self.hard_uncertainty_lower_quantile)
            q_upper = torch.quantile(
                class_uncertainty,
                self.hard_uncertainty_upper_quantile)
            stable = self._select_stable_pixels(
                class_mask, uncertainty_map, q_lower, correct, gt_boundary)
            # q_upper is the explicit exclusion boundary for both banks.
            allowed_uncertainty = uncertainty_map.le(q_upper)
            moderate = (uncertainty_map.ge(q_lower)
                        & allowed_uncertainty)
            hard_source = predicted_hard_boundary | moderate
            hard = (class_mask & allowed_uncertainty & hard_source
                    & p_gt.ge(self.hard_min_gt_probability) & ~stable)
            discarded = class_mask & uncertainty_map.gt(q_upper)
            # Assertions document and continuously enforce the set contract.
            if bool((stable & hard).any().item()):
                raise RuntimeError('Stable and hard pixel sets overlap.')
            if bool(((stable | hard) & discarded).any().item()):
                raise RuntimeError(
                    'Top-uncertainty pixels entered a prototype set.')
            stable_masks.append(stable)
            hard_masks.append(hard)
            stable_counts[class_id] = stable.sum()
            hard_counts[class_id] = hard.sum()
            discarded_counts[class_id] = discarded.sum()
        return dict(
            gt=gt_small,
            gt_boundary=gt_boundary,
            stable_masks=stable_masks,
            hard_masks=hard_masks,
            stable_counts=stable_counts,
            hard_counts=hard_counts,
            discarded_counts=discarded_counts)

    @staticmethod
    @torch.no_grad()
    def _compute_class_sufficient_statistics(feature, masks, num_classes,
                                               feature_dim):
        """Compute unweighted per-class feature sums and pixel counts."""
        feature = feature.detach().float().permute(0, 2, 3, 1)
        feature_sum = feature.new_zeros((num_classes, feature_dim))
        pixel_count = torch.zeros(
            num_classes, device=feature.device, dtype=torch.long)
        for class_id in range(num_classes):
            mask = masks[class_id]
            count = mask.sum()
            pixel_count[class_id] = count
            if int(count.item()) > 0:
                feature_sum[class_id] = feature[mask].sum(dim=0)
        return feature_sum, pixel_count

    @staticmethod
    def _all_reduce_class_statistics(feature_sum, pixel_count):
        """Synchronize sums/counts, never rank-local means."""
        if (dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            dist.all_reduce(feature_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(pixel_count, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def _collect_anchored_prototype_statistics(
            self, decoder_feat, uncertainty, z_visual, predicted_boundary,
            gt_semantic_seg):
        masks = self._build_anchored_prototype_masks(
            uncertainty, z_visual, predicted_boundary, gt_semantic_seg)
        stable_sum, stable_count = self._compute_class_sufficient_statistics(
            decoder_feat, masks['stable_masks'], self.num_classes,
            self.decoder_channels)
        hard_sum, hard_count = self._compute_class_sufficient_statistics(
            decoder_feat, masks['hard_masks'], self.num_classes,
            self.decoder_channels)
        self._all_reduce_class_statistics(stable_sum, stable_count)
        self._all_reduce_class_statistics(hard_sum, hard_count)
        discarded_count = masks['discarded_counts'].clone()
        if self._distributed():
            dist.all_reduce(discarded_count, op=dist.ReduceOp.SUM)
        return stable_sum, stable_count, hard_sum, hard_count, discarded_count

    @torch.no_grad()
    def _update_ema_prototypes(self, kind, feature_sum, pixel_count,
                               momentum):
        """Apply identical per-class EMA updates on every distributed rank."""
        prototypes = getattr(self, '{}_prototypes'.format(kind))
        initialized = getattr(self, '{}_initialized'.format(kind))
        update_count = getattr(self, '{}_update_count'.format(kind))
        cumulative_pixels = getattr(self, '{}_pixel_count'.format(kind))
        for class_id in range(self.num_classes):
            count = int(pixel_count[class_id].item())
            if count < self.prototype_min_pixels:
                continue
            batch_prototype = feature_sum[class_id] / float(count)
            batch_prototype = self._normalize_prototype(batch_prototype)
            if (not bool(torch.isfinite(batch_prototype).all())
                    or float(batch_prototype.norm().item()) <=
                    self.prototype_eps):
                continue
            if not bool(initialized[class_id].item()):
                updated = batch_prototype
                initialized[class_id] = True
            else:
                updated = (float(momentum) * prototypes[class_id].float()
                           + (1.0 - float(momentum)) * batch_prototype)
                updated = self._normalize_prototype(updated)
            prototypes[class_id].copy_(updated.to(prototypes.dtype))
            update_count[class_id].add_(1)
            cumulative_pixels[class_id].add_(pixel_count[class_id])

    @torch.no_grad()
    def _update_anchored_dual_prototypes(
            self, decoder_feat, uncertainty, z_visual, predicted_boundary,
            gt_semantic_seg):
        """Update anchored banks only after current-batch losses exist."""
        if not self.training or not self.use_stable_hard_dual_prototypes:
            return False
        (stable_sum, stable_count, hard_sum, hard_count,
         discarded_count) = self._collect_anchored_prototype_statistics(
             decoder_feat, uncertainty, z_visual, predicted_boundary,
             gt_semantic_seg)
        self.last_stable_selected_pixels.copy_(stable_count)
        self.last_hard_selected_pixels.copy_(hard_count)
        self.last_discarded_top_uncertain_pixels.copy_(discarded_count)
        self._update_ema_prototypes(
            'stable', stable_sum, stable_count,
            self.stable_prototype_momentum)
        if int(self.prototype_step.item()) >= (
                self.hard_prototype_update_start_iter):
            self._update_ema_prototypes(
                'hard', hard_sum, hard_count,
                self.hard_prototype_momentum)
        self._latest_dual_diagnostics.update(dict(
            stable_selected_pixels_per_class=stable_count.detach().clone(),
            hard_selected_pixels_per_class=hard_count.detach().clone(),
            discarded_top_uncertain_pixels_per_class=(
                discarded_count.detach().clone())))
        return True

    @torch.no_grad()
    def _build_dual_prototype_masks(self, uncertainty, z_visual,
                                    gt_semantic_seg):
        """Build per-class stable/hard masks at decoder resolution."""
        if gt_semantic_seg.dim() == 3:
            gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
        gt_small = F.interpolate(
            gt_semantic_seg.float(),
            size=uncertainty.shape[-2:],
            mode='nearest').long().squeeze(1)
        self._validate_segmentation_labels(gt_small)
        valid = gt_small.ne(self.ignore_index)
        boundary = self.generate_gt_boundary(
            gt_small, self.ignore_index, self.gt_boundary_width)
        uncertainty = uncertainty.detach().float().squeeze(1)
        prediction = z_visual.detach().argmax(dim=1)
        correct = prediction.eq(gt_small)
        stable_masks = []
        hard_masks = []
        stable_weights = []
        hard_weights = []
        excluded_noise_pixels = 0

        for class_id in range(self.num_classes):
            class_mask = valid & gt_small.eq(class_id)
            class_pixels = int(class_mask.sum().item())
            if class_pixels < self.prototype_quantile_min_pixels:
                empty = torch.zeros_like(class_mask)
                stable_masks.append(empty)
                hard_masks.append(empty)
                stable_weights.append(uncertainty.new_zeros(
                    uncertainty.shape))
                hard_weights.append(uncertainty.new_zeros(
                    uncertainty.shape))
                continue

            class_uncertainty = uncertainty[class_mask]
            tau_stable = torch.quantile(
                class_uncertainty, self.stable_uncertainty_quantile)
            tau_noise = torch.quantile(
                class_uncertainty, self.noise_uncertainty_quantile)
            below_noise = uncertainty.lt(tau_noise)
            excluded_noise_pixels += int(
                (class_mask & ~below_noise).sum().item())

            if bool(self.stable_initialized[class_id].item()):
                stable = class_mask & uncertainty.le(tau_stable) & ~boundary
                if self.require_correct_for_stable:
                    stable &= correct
            else:
                # Robust cold start: do not trust early predictions, but still
                # exclude boundaries and the class's highest-uncertainty tail.
                stable = class_mask & ~boundary
            if self.exclude_highest_uncertainty:
                stable &= below_noise

            moderate = (uncertainty.gt(tau_stable)
                        & uncertainty.lt(tau_noise))
            hard_source = torch.zeros_like(class_mask)
            if self.hard_use_gt_boundary:
                hard_source |= boundary
            if self.hard_use_moderate_uncertainty:
                hard_source |= moderate
            hard = class_mask & hard_source
            if self.exclude_highest_uncertainty:
                hard &= below_noise
            # A hard class cannot initialize before its stable anchor or
            # before the inherited warm-up budget has elapsed.
            if (not bool(self.stable_initialized[class_id].item())
                    or int(self.dual_prototype_step.item()) <
                    self.prototype_warmup_iters):
                hard.zero_()

            reliability = (1.0 - uncertainty).clamp(min=0.05)
            stable_weight = reliability * stable.float()
            hard_weight = reliability * (
                1.0 + self.hard_gate_boundary_weight * boundary.float())
            hard_weight *= hard.float()
            stable_masks.append(stable)
            hard_masks.append(hard)
            stable_weights.append(stable_weight)
            hard_weights.append(hard_weight)

        valid_pixels = max(int(valid.sum().item()), 1)
        stable_pixels = sum(int(mask.sum().item()) for mask in stable_masks)
        hard_pixels = sum(int(mask.sum().item()) for mask in hard_masks)
        diagnostics = dict(
            stable_pixel_ratio=uncertainty.new_tensor(
                stable_pixels / valid_pixels),
            hard_pixel_ratio=uncertainty.new_tensor(
                hard_pixels / valid_pixels),
            excluded_noise_ratio=uncertainty.new_tensor(
                excluded_noise_pixels / valid_pixels))
        return dict(
            gt=gt_small,
            boundary=boundary,
            stable_masks=stable_masks,
            hard_masks=hard_masks,
            stable_weights=stable_weights,
            hard_weights=hard_weights,
            diagnostics=diagnostics)

    @staticmethod
    def _all_reduce_dual_statistics(feature_sum, weight_sum, pixel_count):
        if (dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            dist.all_reduce(feature_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(weight_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(pixel_count, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def _collect_dual_prototype_statistics(self, decoder_feat, uncertainty,
                                           z_visual, gt_semantic_seg):
        masks = self._build_dual_prototype_masks(
            uncertainty, z_visual, gt_semantic_seg)
        feature = decoder_feat.detach().float().permute(0, 2, 3, 1)
        shape = (self.num_classes, self.decoder_channels)
        stable_sum = feature.new_zeros(shape)
        hard_sum = feature.new_zeros(shape)
        stable_weight_sum = feature.new_zeros(self.num_classes)
        hard_weight_sum = feature.new_zeros(self.num_classes)
        stable_count = torch.zeros(
            self.num_classes, device=feature.device, dtype=torch.long)
        hard_count = torch.zeros_like(stable_count)
        for class_id in range(self.num_classes):
            for kind, vector_sum, weight_sum, pixel_count in (
                    ('stable', stable_sum, stable_weight_sum, stable_count),
                    ('hard', hard_sum, hard_weight_sum, hard_count)):
                mask = masks['{}_masks'.format(kind)][class_id]
                weights = masks['{}_weights'.format(kind)][class_id][mask]
                count = int(mask.sum().item())
                if count == 0:
                    continue
                selected = feature[mask]
                vector_sum[class_id] = (selected * weights.unsqueeze(1)).sum(0)
                weight_sum[class_id] = weights.sum()
                pixel_count[class_id] = count

        self._all_reduce_dual_statistics(
            stable_sum, stable_weight_sum, stable_count)
        self._all_reduce_dual_statistics(
            hard_sum, hard_weight_sum, hard_count)
        return (stable_sum, stable_weight_sum, stable_count,
                hard_sum, hard_weight_sum, hard_count,
                masks['diagnostics'])

    def _normalize_prototype(self, vector):
        if not self.normalize_prototypes:
            return vector
        return F.normalize(vector, p=2, dim=0, eps=self.prototype_eps)

    @torch.no_grad()
    def get_dual_prototype_diagnostics(self, decoder_feat=None):
        """Return checkpoint-safe state diagnostics, optionally score stats."""
        if not self.use_uncertainty_dual_prototypes:
            raise RuntimeError('Uncertainty dual prototypes are disabled.')
        stable = self.stable_prototypes.detach().float()
        hard = self.hard_prototypes.detach().float()
        both = self.stable_initialized & self.hard_initialized
        cosine = stable.new_full((self.num_classes, ), float('nan'))
        if bool(both.any().item()):
            cosine[both] = F.cosine_similarity(
                stable[both], hard[both], dim=1, eps=self.prototype_eps)
        diagnostics = dict(
            stable_prototypes=stable.clone(),
            hard_prototypes=hard.clone(),
            stable_initialized=self.stable_initialized.detach().clone(),
            hard_initialized=self.hard_initialized.detach().clone(),
            stable_update_count=self.stable_update_count.detach().clone(),
            hard_update_count=self.hard_update_count.detach().clone(),
            stable_pixel_count=self.stable_pixel_count.detach().clone(),
            hard_pixel_count=self.hard_pixel_count.detach().clone(),
            stable_hard_cosine=cosine,
            stable_norm=stable.norm(dim=1),
            hard_norm=hard.norm(dim=1),
            stable_logit_mean=self.last_stable_logit_mean.detach().clone(),
            stable_logit_std=self.last_stable_logit_std.detach().clone(),
            hard_logit_mean=self.last_hard_logit_mean.detach().clone(),
            hard_logit_std=self.last_hard_logit_std.detach().clone(),
            dual_prototype_step=self.dual_prototype_step.detach().clone())
        if self.use_stable_hard_dual_prototypes:
            schedule = self.get_anchored_prototype_schedule()
            diagnostics.update(
                prototype_step=self.prototype_step.detach().clone(),
                effective_prototype_beta=stable.new_tensor(
                    schedule['prototype_beta']),
                effective_hard_correction_scale=stable.new_tensor(
                    schedule['hard_correction_scale']),
                effective_prototype_aux_weight=stable.new_tensor(
                    schedule['prototype_aux_weight']),
                hard_gate_mean=self.last_hard_gate_mean.detach().clone(),
                hard_gate_p50=self.last_hard_gate_p50.detach().clone(),
                hard_gate_p90=self.last_hard_gate_p90.detach().clone(),
                hard_gate_saturation_ratio=(
                    self.last_hard_gate_saturation_ratio.detach().clone()),
                stable_selected_pixels_per_class=(
                    self.last_stable_selected_pixels.detach().clone()),
                hard_selected_pixels_per_class=(
                    self.last_hard_selected_pixels.detach().clone()),
                discarded_top_uncertain_pixels_per_class=(
                    self.last_discarded_top_uncertain_pixels.detach().clone()),
                stable_update_count_per_class=(
                    self.stable_update_count.detach().clone()),
                hard_update_count_per_class=(
                    self.hard_update_count.detach().clone()),
                stable_hard_cosine_per_class=cosine.clone(),
                stable_prototype_norm_per_class=stable.norm(dim=1),
                hard_prototype_norm_per_class=hard.norm(dim=1),
                prototype_residual_ratio=(
                    self.last_prototype_residual_ratio.detach().clone()),
                hard_correction_ratio=(
                    self.last_hard_correction_ratio.detach().clone()))
        if decoder_feat is not None:
            z_stable = self._compute_prototype_logits(decoder_feat, stable)
            z_hard = self._compute_prototype_logits(decoder_feat, hard)
            reduce_dims = (0, 2, 3)
            diagnostics.update(
                stable_logit_mean=z_stable.mean(dim=reduce_dims),
                stable_logit_std=z_stable.std(
                    dim=reduce_dims, unbiased=False),
                hard_logit_mean=z_hard.mean(dim=reduce_dims),
                hard_logit_std=z_hard.std(
                    dim=reduce_dims, unbiased=False))
        return diagnostics

    @torch.no_grad()
    def _update_uncertainty_dual_prototypes(self, decoder_feat, uncertainty,
                                            z_visual, gt_semantic_seg):
        """Update synchronized EMA buffers for use by subsequent batches."""
        if not self.training or not self.use_uncertainty_dual_prototypes:
            return False
        statistics = self._collect_dual_prototype_statistics(
            decoder_feat, uncertainty, z_visual, gt_semantic_seg)
        (stable_sum, stable_weight_sum, stable_count,
         hard_sum, hard_weight_sum, hard_count, diagnostics) = statistics
        self._latest_dual_diagnostics = diagnostics

        for kind, feature_sum, weight_sum, pixel_count in (
                ('stable', stable_sum, stable_weight_sum, stable_count),
                ('hard', hard_sum, hard_weight_sum, hard_count)):
            prototypes = getattr(self, '{}_prototypes'.format(kind))
            initialized = getattr(self, '{}_initialized'.format(kind))
            update_count = getattr(self, '{}_update_count'.format(kind))
            cumulative_pixels = getattr(self, '{}_pixel_count'.format(kind))
            for class_id in range(self.num_classes):
                count = int(pixel_count[class_id].item())
                if (count < self.prototype_min_pixels
                        or float(weight_sum[class_id].item()) <=
                        self.prototype_eps):
                    continue
                batch_prototype = self._normalize_prototype(
                    feature_sum[class_id] / weight_sum[class_id])
                if not bool(torch.isfinite(batch_prototype).all()):
                    continue
                if not bool(initialized[class_id].item()):
                    updated = batch_prototype
                    initialized[class_id] = True
                else:
                    updated = (self.prototype_momentum
                               * prototypes[class_id].float()
                               + (1.0 - self.prototype_momentum)
                               * batch_prototype)
                    updated = self._normalize_prototype(updated)
                prototypes[class_id].copy_(updated.to(prototypes.dtype))
                update_count[class_id].add_(1)
                cumulative_pixels[class_id].add_(pixel_count[class_id])
        return True

    def _validate_segmentation_labels(self, seg_label):
        """Reject labels outside the configured closed-set taxonomy."""
        labels = seg_label.detach().long()
        valid = labels.ne(self.ignore_index)
        if not bool(valid.any()):
            return
        valid_labels = labels[valid]
        minimum = int(valid_labels.min().item())
        maximum = int(valid_labels.max().item())
        if minimum < 0 or maximum >= self.num_classes:
            raise ValueError(
                'GT contains class ids outside [0, {}): min={}, max={}; '
                'ignore_index={} is the only ignored value.'.format(
                    self.num_classes, minimum, maximum, self.ignore_index))

    @torch.no_grad()
    def _update_visual_prototypes(self, decoder_feat, gt_semantic_seg):
        """Update EMA buffers from current GT for use by later batches only."""
        if not self.training or not self.use_visual_prototype:
            return False
        class_sum, class_count = self._collect_batch_prototype_statistics(
            decoder_feat, gt_semantic_seg)
        for class_id in range(self.num_classes):
            if float(class_count[class_id].item()) <= 0.0:
                continue
            batch_prototype = F.normalize(
                class_sum[class_id] / class_count[class_id],
                p=2,
                dim=0,
                eps=self.prototype_eps)
            if not bool(self.prototype_initialized[class_id].item()):
                updated = batch_prototype
                self.prototype_initialized[class_id] = True
            else:
                updated = (
                    self.prototype_momentum *
                    self.visual_prototypes[class_id].float() +
                    (1.0 - self.prototype_momentum) * batch_prototype)
                updated = F.normalize(
                    updated, p=2, dim=0, eps=self.prototype_eps)
            self.visual_prototypes[class_id].copy_(
                updated.to(self.visual_prototypes.dtype))
            self.prototype_update_count[class_id].add_(1)
        return True

    @force_fp32(apply_to=('z_proto', ))
    def _prototype_loss(self, z_proto, seg_label, loss_weight=None):
        target_size = seg_label.shape[-2:]
        z_proto = resize(
            z_proto,
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        valid = seg_label.ne(self.ignore_index)
        if bool(valid.any()):
            loss_proto = F.cross_entropy(
                z_proto, seg_label, ignore_index=self.ignore_index)
            acc_proto = accuracy(z_proto, seg_label)
        else:
            loss_proto = z_proto.sum() * 0.0
            acc_proto = z_proto.new_zeros(())
        if loss_weight is None:
            loss_weight = self.prototype_loss_weight
        return float(loss_weight) * loss_proto, acc_proto

    def _batch_classes_have_stable_prototypes(self, seg_label,
                                               initialized_snapshot):
        """Check only valid classes occurring in this batch."""
        labels = seg_label.detach().long()
        valid = labels.ne(self.ignore_index)
        if not bool(valid.any().item()):
            return False
        present = torch.unique(labels[valid])
        return bool(initialized_snapshot[present].all().item())

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        self._validate_segmentation_labels(gt_semantic_seg)
        decoder_feat = self._decode_features(inputs)

        # z_proto is built from a cloned historical bank before current GT is
        # observed.  This ordering is the central no-label-leakage guarantee.
        seg_logits, aux = self._forward_prada_from_features(
            decoder_feat, return_aux=True)
        losses = self._losses_with_aux(
            seg_logits,
            gt_semantic_seg,
            s_pos=aux.get('s_pos'),
            s_contrast=aux.get('s_contrast'),
            presence_logits=aux.get('presence_logits'),
            img_metas=img_metas)

        prototype_aux_ready = aux['prototype_ready']
        if self.use_uncertainty_dual_prototypes:
            prototype_aux_ready = aux['prototype_aux_ready']
        prototype_loss_weight = None
        if self.use_stable_hard_dual_prototypes:
            prototype_loss_weight = aux['effective_prototype_aux_weight']
            prototype_aux_ready = (
                int(self.prototype_step.item()) >=
                self.hard_correction_start_iter
                and prototype_loss_weight > 0.0
                and self._batch_classes_have_stable_prototypes(
                    gt_semantic_seg,
                    aux['stable_initialized_snapshot']))
        if (self.use_visual_prototype and self.use_prototype_aux_loss
                and prototype_aux_ready):
            losses['loss_proto'], losses['acc_proto'] = self._prototype_loss(
                aux['z_proto'], gt_semantic_seg,
                loss_weight=prototype_loss_weight)
        elif self.use_stable_hard_dual_prototypes:
            # Do not run CE against unavailable class anchors.  Keep a graph-
            # connected zero so distributed loss parsing remains consistent.
            losses['loss_proto'] = aux['z_proto'].sum() * 0.0
            losses['acc_proto'] = seg_logits.new_zeros(())

        if self.use_text_relation_distillation:
            relation_loss, relation_diagnostics = self._text_relation_loss()
            losses['loss_text_relation'] = relation_loss
            losses.update(relation_diagnostics)

        # Current GT may affect only subsequent batches.
        with torch.no_grad():
            # Subclasses may transform the decoder feature before the final
            # prediction.  Updating from that same feature keeps the EMA bank
            # aligned with the representation it corrects while preserving
            # historical behavior for every existing PrAda configuration.
            prototype_update_feat = aux.get(
                'prototype_update_feat', decoder_feat.detach()).detach()
            if self.use_uncertainty_dual_prototypes:
                if self.use_stable_hard_dual_prototypes:
                    self._update_anchored_dual_prototypes(
                        prototype_update_feat, aux['uncertainty'].detach(),
                        self._latest_visual_logits.detach(),
                        aux['predicted_boundary'].detach(),
                        gt_semantic_seg)
                else:
                    self._update_uncertainty_dual_prototypes(
                        prototype_update_feat, aux['uncertainty'].detach(),
                        self._latest_visual_logits.detach(), gt_semantic_seg)
                self.dual_prototype_step.add_(1)
                # Keep the inherited step meaningful for common tooling while
                # the dedicated dual step remains the source of truth.
                self.prototype_step.copy_(self.dual_prototype_step)
            else:
                self._update_visual_prototypes(
                    prototype_update_feat, gt_semantic_seg)
                self.prototype_step.add_(1)

        # These detached scalars are logged by MMSeg but are not summed into
        # the objective because their names do not contain "loss".
        losses['beta_proto'] = self.beta_proto.detach()
        losses['prototype_ready'] = seg_logits.new_tensor(
            float(aux['prototype_ready']))
        if self.use_uncertainty_dual_prototypes:
            uncertainty = aux['uncertainty'].detach().float()
            hard_gate = aux['hard_gate'].detach().float()
            losses['prototype_initialized_count'] = seg_logits.new_tensor(
                float(self.stable_initialized.sum().item()))
            losses['uncertainty_mean'] = uncertainty.mean()
            losses['uncertainty_std'] = uncertainty.std(unbiased=False)
            losses['hard_gate_mean'] = hard_gate.mean()
            losses['hard_gate_max'] = hard_gate.max()
            losses['prototype_step'] = seg_logits.new_tensor(
                float(self.dual_prototype_step.item()))
            for name in ('stable_pixel_ratio', 'hard_pixel_ratio',
                         'excluded_noise_ratio'):
                losses[name] = self._latest_dual_diagnostics.get(
                    name, seg_logits.new_zeros(())).detach()
            initialized_pair = self.stable_initialized & self.hard_initialized
            if bool(initialized_pair.any().item()):
                cosine = F.cosine_similarity(
                    self.stable_prototypes[initialized_pair].float(),
                    self.hard_prototypes[initialized_pair].float(), dim=1)
                losses['stable_hard_cosine_mean'] = cosine.mean().detach()
            else:
                losses['stable_hard_cosine_mean'] = seg_logits.new_zeros(())
            if self.use_stable_hard_dual_prototypes:
                diagnostic_names = (
                    'effective_prototype_beta',
                    'effective_hard_correction_scale',
                    'effective_prototype_aux_weight',
                    'hard_gate_p50', 'hard_gate_p90',
                    'hard_gate_saturation_ratio',
                    'prototype_residual_ratio',
                    'hard_correction_ratio')
                for name in diagnostic_names:
                    losses[name] = self._latest_dual_diagnostics.get(
                        name, seg_logits.new_zeros(())).detach()
                for class_id in range(self.num_classes):
                    losses['stable_selected_pixels_{}'.format(class_id)] = (
                        self.last_stable_selected_pixels[class_id].detach(
                        ).float())
                    losses['hard_selected_pixels_{}'.format(class_id)] = (
                        self.last_hard_selected_pixels[class_id].detach(
                        ).float())
                    losses['discarded_top_uncertain_pixels_{}'.format(
                        class_id)] = (
                            self.last_discarded_top_uncertain_pixels[
                                class_id].detach().float())
                    losses['stable_update_count_{}'.format(class_id)] = (
                        self.stable_update_count[class_id].detach().float())
                    losses['hard_update_count_{}'.format(class_id)] = (
                        self.hard_update_count[class_id].detach().float())
        else:
            losses['prototype_initialized_count'] = seg_logits.new_tensor(
                float(self.prototype_initialized.sum().item()))
        for name in ('z_visual_abs_mean', 'z_visual_abs_std',
                     'z_text_abs_mean', 'z_text_abs_std',
                     'z_proto_abs_mean', 'z_proto_abs_std',
                     'stable_score_std', 'hard_score_std',
                     'dual_score_std'):
            if (not self.use_uncertainty_dual_prototypes
                    and name.startswith(('stable_', 'hard_', 'dual_'))):
                continue
            losses[name] = self._latest_logit_stats.get(
                name, seg_logits.new_zeros(())).detach()
        return losses
