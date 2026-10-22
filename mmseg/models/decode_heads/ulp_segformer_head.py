import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32, get_dist_info

from mmseg.models.builder import HEADS
from mmseg.ops import resize
from ..losses import accuracy
from .class_presence import ClassPresenceBranch, build_presence_labels
from .segformer_head import MLP, SegFormerHead


@HEADS.register_module()
class ULPSegFormerHead(SegFormerHead):
    """Learnable class-reference matching on shared SegFormer decoder features.

    Legacy class/parameter names are retained for checkpoint compatibility.
    The released configuration uses trainable numeric class embeddings, not
    language-model text features. The original visual classifier is preserved.
    """

    def __init__(self,
                 feature_strides,
                 text_prototype_path,
                 beta=0.1,
                 fusion_type='ubm_gate',
                 background_index=None,
                 background_class_indices=None,
                 expected_class_names=None,
                 expected_dataset_id=None,
                 require_real_text_prototypes=False,
                 allow_dummy_text_prototypes=True,
                 require_exact_text_prototype_dim=True,
                 use_uncertainty_gate=True,
                 use_boundary_gate=True,
                 use_foreground_gate=True,
                 normalize_feature=True,
                 learnable_prototype=False,
                 logit_scale=1.0,
                 detach_gate=True,
                 gate_kernel_size=3,
                 use_ccpa=False,
                 pos_text_embed_path=None,
                 neg_text_embed_path=None,
                 text_dim=None,
                 ccpa_neg_reduce='max',
                 ccpa_eta_type='scalar',
                 ccpa_eta_init=0.5,
                 use_ccpa_logits=False,
                 ccpa_beta_type='scalar',
                 ccpa_beta_init=0.0,
                 lambda_ccpa_pos=0.05,
                 lambda_ccpa_contrast=0.05,
                 learnable_text_embeddings=False,
                 ccpa_align_feature='f_lang',
                 use_presence_loss=False,
                 lambda_presence=0.05,
                 presence_min_pixels=10,
                 presence_feature='f_lang',
                 presence_mlp_type='linear',
                 presence_dropout=0.0,
                 presence_label_source='gt',
                 presence_label_file=None,
                 use_presence_inference=False,
                 presence_suppress_gamma=0.0,
                 presence_threshold=0.2,
                 **kwargs):
        self.text_prototype_path = text_prototype_path
        self.beta = beta
        self.fusion_type = fusion_type
        if background_index is not None:
            if background_class_indices is not None:
                raise ValueError(
                    'Set background_class_indices or deprecated '
                    'background_index, not both.')
            warnings.warn(
                'background_index is deprecated; use '
                'background_class_indices=[index].', DeprecationWarning)
            background_class_indices = [background_index]
        self.background_class_indices = list(
            background_class_indices or [])
        self.expected_class_names = (
            list(expected_class_names)
            if expected_class_names is not None else None)
        self.expected_dataset_id = expected_dataset_id
        self.require_real_text_prototypes = bool(
            require_real_text_prototypes)
        self.allow_dummy_text_prototypes = bool(
            allow_dummy_text_prototypes)
        self.require_exact_text_prototype_dim = bool(
            require_exact_text_prototype_dim)
        self.use_uncertainty_gate = use_uncertainty_gate
        self.use_boundary_gate = use_boundary_gate
        self.use_foreground_gate = use_foreground_gate
        self.normalize_feature = normalize_feature
        self.learnable_prototype = learnable_prototype
        self.logit_scale = logit_scale
        self.detach_gate = detach_gate
        self.gate_kernel_size = gate_kernel_size
        self.use_ccpa = use_ccpa
        self.pos_text_embed_path = pos_text_embed_path
        self.neg_text_embed_path = neg_text_embed_path
        self.text_dim = text_dim
        self.ccpa_neg_reduce = ccpa_neg_reduce
        self.ccpa_eta_type = ccpa_eta_type
        self.ccpa_eta_init = ccpa_eta_init
        self.use_ccpa_logits = use_ccpa_logits
        self.ccpa_beta_type = ccpa_beta_type
        self.ccpa_beta_init = ccpa_beta_init
        self.lambda_ccpa_pos = lambda_ccpa_pos
        self.lambda_ccpa_contrast = lambda_ccpa_contrast
        self.learnable_text_embeddings = learnable_text_embeddings
        self.ccpa_align_feature = ccpa_align_feature
        self.use_presence_loss = use_presence_loss
        self.lambda_presence = lambda_presence
        self.presence_min_pixels = presence_min_pixels
        self.presence_feature = presence_feature
        self.presence_mlp_type = presence_mlp_type
        self.presence_dropout = presence_dropout
        self.presence_label_source = presence_label_source
        self.presence_label_file = presence_label_file
        self.use_presence_inference = use_presence_inference
        self.presence_suppress_gamma = presence_suppress_gamma
        self.presence_threshold = presence_threshold

        if fusion_type not in ('direct', 'u_gate', 'ub_gate', 'ubm_gate'):
            raise ValueError(
                'fusion_type must be one of direct, u_gate, ub_gate, ubm_gate, '
                'but got {}.'.format(fusion_type))
        if ccpa_neg_reduce not in ('max', 'mean'):
            raise ValueError(
                'ccpa_neg_reduce must be "max" or "mean", but got {}.'.
                format(ccpa_neg_reduce))
        if ccpa_eta_type not in ('scalar', 'per_class'):
            raise ValueError(
                'ccpa_eta_type must be "scalar" or "per_class", but got {}.'.
                format(ccpa_eta_type))
        if ccpa_beta_type not in ('scalar', 'per_class'):
            raise ValueError(
                'ccpa_beta_type must be "scalar" or "per_class", but got {}.'.
                format(ccpa_beta_type))
        if ccpa_align_feature not in ('f_dec', 'f_lang'):
            raise ValueError(
                'ccpa_align_feature must be "f_dec" or "f_lang", but got {}.'.
                format(ccpa_align_feature))
        if presence_feature not in ('f_dec', 'f_lang'):
            raise ValueError(
                'presence_feature must be "f_dec" or "f_lang", but got {}.'.
                format(presence_feature))
        if presence_label_source not in ('gt', 'external'):
            raise ValueError(
                'presence_label_source must be "gt" or "external", but got {}.'
                .format(presence_label_source))

        super(ULPSegFormerHead, self).__init__(feature_strides, **kwargs)

        decoder_params = kwargs.get('decoder_params', {})
        self.decoder_channels = decoder_params.get('embed_dim', self.channels)

        class_names, text_prototypes = self._load_text_prototypes(
            text_prototype_path)
        text_dim = text_prototypes.size(1)
        if text_dim == self.decoder_channels:
            self.text_proj = None
        else:
            if self.require_exact_text_prototype_dim:
                raise ValueError(
                    'Text prototype feature dimension {} does not match the '
                    'actual decoder feature dimension {}. Generate a '
                    'dataset-specific prototype file with the correct '
                    'offline projection.'.format(
                        text_dim, self.decoder_channels))
            self.text_proj = nn.Linear(
                text_dim, self.decoder_channels, bias=False)

        if learnable_prototype:
            self.text_prototypes = nn.Parameter(text_prototypes)
        else:
            self.register_buffer('text_prototypes', text_prototypes)

        self.effective_fusion_type, gate_channels = self._resolve_gate_channels()
        if gate_channels > 0:
            padding = gate_kernel_size // 2
            self.gate_conv = nn.Conv2d(
                gate_channels, 1, kernel_size=gate_kernel_size, padding=padding)
        else:
            self.gate_conv = None

        if self.use_ccpa:
            self._init_ccpa_modules()
        else:
            self.ccpa_text_proj = None
            self.ccpa_eta_raw = None
            self.ccpa_beta_raw = None

        if self.use_presence_loss or self.use_presence_inference:
            self.presence_branch = ClassPresenceBranch(
                self.decoder_channels,
                self.num_classes,
                mlp_type=presence_mlp_type,
                dropout=presence_dropout)
        else:
            self.presence_branch = None

        self._log_initialization(class_names, text_prototypes.shape,
                                 gate_channels)

    def _load_text_prototypes(self, text_prototype_path):
        if not text_prototype_path:
            raise ValueError('text_prototype_path must be a non-empty path.')
        if not os.path.isfile(text_prototype_path):
            raise FileNotFoundError(
                'Text prototype file does not exist: {}'.format(
                    text_prototype_path))

        obj = torch.load(text_prototype_path, map_location='cpu')
        class_names = None
        if isinstance(obj, torch.Tensor):
            text_prototypes = obj
        elif isinstance(obj, dict):
            class_names = obj.get('class_names', None)
            if 'text_prototypes' in obj:
                text_prototypes = obj['text_prototypes']
            elif 'prototypes' in obj:
                text_prototypes = obj['prototypes']
            else:
                raise KeyError(
                    'Text prototype dict must contain "text_prototypes" or '
                    '"prototypes".')
        else:
            raise TypeError(
                'Text prototype file must contain a Tensor or a dict, but got '
                '{}.'.format(type(obj).__name__))

        if not isinstance(text_prototypes, torch.Tensor):
            raise TypeError(
                'Loaded text prototypes must be a torch.Tensor, but got {}.'.
                format(type(text_prototypes).__name__))
        if text_prototypes.dim() != 2:
            raise ValueError(
                'Loaded text prototypes must be a 2D tensor [num_classes, dim], '
                'but got shape {}.'.format(tuple(text_prototypes.shape)))
        if text_prototypes.size(0) != self.num_classes:
            raise ValueError(
                'Loaded text prototypes have {} classes, but decode head '
                'expects {} classes.'.format(text_prototypes.size(0),
                                             self.num_classes))
        if not bool(torch.isfinite(text_prototypes).all()):
            raise ValueError('Loaded text prototypes contain NaN or Inf.')
        if bool(text_prototypes.norm(dim=1).le(0).any()):
            raise ValueError('Every text prototype must have non-zero norm.')

        if isinstance(obj, dict):
            self._validate_text_prototype_metadata(
                obj, class_names, text_prototypes)
        elif (self.require_real_text_prototypes
              or self.expected_class_names is not None):
            raise ValueError(
                'Formal text prototypes must be stored as a metadata dict; '
                'a bare Tensor cannot prove encoder source or class order.')

        return class_names, text_prototypes.float()

    def _validate_text_prototype_metadata(self, obj, class_names,
                                          text_prototypes):
        is_dummy = bool(obj.get('is_dummy', False))
        if is_dummy and (self.require_real_text_prototypes
                         or not self.allow_dummy_text_prototypes):
            raise ValueError(
                'Dummy text prototypes are forbidden by this configuration: '
                '{}'.format(self.text_prototype_path))
        if self.require_real_text_prototypes:
            if self.expected_class_names is None:
                raise ValueError(
                    'Formal mode requires expected_class_names so class order '
                    'is verified against the active dataset configuration.')
            required = (
                'is_dummy', 'source_type', 'dataset_id', 'domain',
                'class_names', 'class_ids', 'background_class_indices',
                'prompts', 'encoder_name', 'encoder_version',
                'aggregation_method', 'projection_method', 'feature_dim',
                'num_classes', 'class_spec_sha256')
            missing = [key for key in required if key not in obj]
            if missing:
                raise ValueError(
                    'Real text prototype metadata is incomplete; missing: '
                    '{}'.format(', '.join(missing)))
            if obj['source_type'] != 'real_text_encoder' or is_dummy:
                raise ValueError(
                    'Formal mode requires source_type="real_text_encoder" '
                    'and is_dummy=False.')
            if obj['domain'] != 'underwater_semantic_segmentation':
                raise ValueError('Unexpected prototype domain: {}'.format(
                    obj['domain']))
            prompts = obj['prompts']
            if not isinstance(prompts, dict) or set(prompts) != set(
                    class_names or []):
                raise ValueError(
                    'Prompt metadata keys must exactly match class_names.')
            if any(not isinstance(items, list) or not items
                   for items in prompts.values()):
                raise ValueError('Every class must contain at least one prompt.')

        if class_names is None and self.expected_class_names is not None:
            raise ValueError(
                'Prototype metadata has no class_names, so class order '
                'cannot be verified.')
        if (self.expected_class_names is not None
                and list(class_names) != self.expected_class_names):
            raise ValueError(
                'Text prototype class order mismatch. Expected {}, got {}.'
                .format(self.expected_class_names, list(class_names)))
        if class_names is not None and len(class_names) != self.num_classes:
            raise ValueError(
                'Prototype class_names contains {} entries, expected {}.'
                .format(len(class_names), self.num_classes))
        if self.expected_dataset_id is not None:
            if obj.get('dataset_id') != self.expected_dataset_id:
                raise ValueError(
                    'Prototype dataset_id mismatch. Expected {}, got {}.'
                    .format(self.expected_dataset_id, obj.get('dataset_id')))

        if 'class_ids' in obj:
            expected_ids = list(range(self.num_classes))
            if list(obj['class_ids']) != expected_ids:
                raise ValueError(
                    'Prototype class_ids must be contiguous and ordered as '
                    '{}, got {}.'.format(expected_ids, obj['class_ids']))
        if ('num_classes' in obj
                and int(obj['num_classes']) != self.num_classes):
            raise ValueError('Prototype num_classes metadata mismatch.')
        if ('feature_dim' in obj
                and int(obj['feature_dim']) != text_prototypes.size(1)):
            raise ValueError('Prototype feature_dim metadata mismatch.')
        if 'background_class_indices' in obj:
            metadata_background = list(obj['background_class_indices'])
            if metadata_background != self.background_class_indices:
                raise ValueError(
                    'Prototype background indices {} do not match configured '
                    'indices {}.'.format(metadata_background,
                                        self.background_class_indices))

    def _load_ccpa_embeddings(self, path, name, valid_dims):
        if not path:
            raise ValueError('{} must be set when use_ccpa=True.'.format(name))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                '{} file does not exist: {}'.format(name, path))

        obj = torch.load(path, map_location='cpu')
        if isinstance(obj, torch.Tensor):
            embeddings = obj
        elif isinstance(obj, dict):
            for key in (name, 'text_embeddings', 'embeddings',
                        'text_prototypes', 'prototypes'):
                if key in obj:
                    embeddings = obj[key]
                    break
            else:
                raise KeyError(
                    '{} dict must contain one of {}, text_embeddings, '
                    'embeddings, text_prototypes, or prototypes.'.format(
                        name, name))
        else:
            raise TypeError(
                '{} file must contain a Tensor or dict, but got {}.'.format(
                    name, type(obj).__name__))

        if not isinstance(embeddings, torch.Tensor):
            raise TypeError('{} must be a torch.Tensor.'.format(name))
        if embeddings.dim() not in valid_dims:
            raise ValueError(
                '{} must have dim in {}, but got shape {}.'.format(
                    name, valid_dims, tuple(embeddings.shape)))
        if embeddings.size(0) != self.num_classes:
            raise ValueError(
                '{} has {} classes, but decode head expects {}.'.format(
                    name, embeddings.size(0), self.num_classes))
        return embeddings.float()

    @staticmethod
    def _inverse_softplus(value):
        value = max(float(value), 1e-6)
        return torch.log(torch.expm1(torch.tensor(value, dtype=torch.float32)))

    def _init_ccpa_modules(self):
        pos_text_embeddings = self._load_ccpa_embeddings(
            self.pos_text_embed_path, 'pos_text_embeddings', (2, 3))
        neg_text_embeddings = self._load_ccpa_embeddings(
            self.neg_text_embed_path, 'neg_text_embeddings', (3, ))

        pos_dim = pos_text_embeddings.size(-1)
        neg_dim = neg_text_embeddings.size(-1)
        if pos_dim != neg_dim:
            raise ValueError(
                'CCPA positive text dim {} must match negative text dim {}.'.
                format(pos_dim, neg_dim))
        if self.text_dim is not None and int(self.text_dim) != pos_dim:
            raise ValueError(
                'Configured text_dim={} but loaded CCPA embeddings have dim {}.'
                .format(self.text_dim, pos_dim))

        if pos_dim == self.decoder_channels:
            self.ccpa_text_proj = nn.Identity()
        else:
            self.ccpa_text_proj = nn.Linear(
                pos_dim, self.decoder_channels, bias=False)

        if self.learnable_text_embeddings:
            self.pos_text_embeddings = nn.Parameter(pos_text_embeddings)
            self.neg_text_embeddings = nn.Parameter(neg_text_embeddings)
        else:
            self.register_buffer('pos_text_embeddings', pos_text_embeddings)
            self.register_buffer('neg_text_embeddings', neg_text_embeddings)

        eta_shape = (1, ) if self.ccpa_eta_type == 'scalar' else (
            self.num_classes, )
        beta_shape = (1, ) if self.ccpa_beta_type == 'scalar' else (
            self.num_classes, )
        self.ccpa_eta_raw = nn.Parameter(
            self._inverse_softplus(self.ccpa_eta_init).repeat(*eta_shape))
        self.ccpa_beta_raw = nn.Parameter(
            self._inverse_softplus(self.ccpa_beta_init).repeat(*beta_shape))

    def _resolve_gate_channels(self):
        if len(set(self.background_class_indices)) != len(
                self.background_class_indices):
            raise ValueError('background_class_indices contains duplicates.')
        for index in self.background_class_indices:
            if not isinstance(index, int):
                raise TypeError('Background class indices must be integers.')
            if index < 0 or index >= self.num_classes:
                raise ValueError(
                    'Background class index {} is out of range for '
                    'num_classes={}.'.format(index, self.num_classes))

        if self.fusion_type == 'direct':
            return self.fusion_type, 0
        if self.fusion_type == 'u_gate':
            if not self.use_uncertainty_gate:
                raise ValueError('u_gate requires use_uncertainty_gate=True.')
            return self.fusion_type, 0

        use_foreground = (
            self.fusion_type == 'ubm_gate' and self.use_foreground_gate
            and bool(self.background_class_indices))
        effective_fusion_type = self.fusion_type
        if self.fusion_type == 'ubm_gate' and not use_foreground:
            if self.use_foreground_gate and not self.background_class_indices:
                raise ValueError(
                    'use_foreground_gate=True requires at least one explicit '
                    'background_class_indices entry. No class is implicitly '
                    'treated as background.')
            effective_fusion_type = 'ub_gate'

        gate_channels = 0
        if self.use_uncertainty_gate:
            gate_channels += 1
        if self.use_boundary_gate:
            gate_channels += 1
        if use_foreground:
            gate_channels += 1

        if gate_channels == 0:
            raise ValueError(
                '{} requires at least one enabled gate input.'.format(
                    self.fusion_type))
        if self.fusion_type == 'ub_gate' and gate_channels < 2:
            raise ValueError(
                'ub_gate expects uncertainty and boundary inputs. Please keep '
                'use_uncertainty_gate=True and use_boundary_gate=True.')
        if self.fusion_type == 'ubm_gate' and gate_channels < 2:
            raise ValueError(
                'ubm_gate degraded to ub_gate but still needs uncertainty and '
                'boundary inputs.')

        return effective_fusion_type, gate_channels

    def _log_initialization(self, class_names, prototype_shape, gate_channels):
        rank, _ = get_dist_info()
        if rank != 0:
            return
        print('[ULPSegFormerHead] loaded text prototype path: {}'.format(
            self.text_prototype_path))
        print('[ULPSegFormerHead] prototype shape: {}'.format(
            tuple(prototype_shape)))
        if class_names is not None:
            print('[ULPSegFormerHead] class_names: {}'.format(class_names))
        print('[ULPSegFormerHead] text projection used: {}'.format(
            self.text_proj is not None))
        print('[ULPSegFormerHead] beta: {}'.format(self.beta))
        print('[ULPSegFormerHead] fusion_type: {} (effective: {})'.format(
            self.fusion_type, self.effective_fusion_type))
        print('[ULPSegFormerHead] background_class_indices: {}'.format(
            self.background_class_indices))
        print('[ULPSegFormerHead] gate input channels: {}'.format(
            gate_channels))
        print('[ULPSegFormerHead] learnable_prototype: {}'.format(
            self.learnable_prototype))
        print('[ULPSegFormerHead] normalize_feature: {}'.format(
            self.normalize_feature))
        print('[ULPSegFormerHead] use_ccpa: {}'.format(self.use_ccpa))
        if self.use_ccpa:
            print('[ULPSegFormerHead] CCPA pos path: {}'.format(
                self.pos_text_embed_path))
            print('[ULPSegFormerHead] CCPA neg path: {}'.format(
                self.neg_text_embed_path))
            print('[ULPSegFormerHead] CCPA align feature: {}'.format(
                self.ccpa_align_feature))
            print('[ULPSegFormerHead] CCPA neg reduce: {}'.format(
                self.ccpa_neg_reduce))
            print('[ULPSegFormerHead] CCPA logits enabled: {}'.format(
                self.use_ccpa_logits))
        print('[ULPSegFormerHead] use_presence_loss: {}'.format(
            self.use_presence_loss))
        if self.use_presence_loss or self.use_presence_inference:
            print('[ULPSegFormerHead] presence feature: {}'.format(
                self.presence_feature))
            print('[ULPSegFormerHead] presence mlp type: {}'.format(
                self.presence_mlp_type))
            print('[ULPSegFormerHead] lambda_presence: {}'.format(
                self.lambda_presence))
            print('[ULPSegFormerHead] presence_min_pixels: {}'.format(
                self.presence_min_pixels))

    def _get_text_prototypes(self):
        text_proto = self.text_prototypes
        if self.text_proj is not None:
            text_proto = self.text_proj(text_proto)
        return text_proto

    def _compute_language_logits(self, decoder_feat):
        text_proto = self._get_text_prototypes()
        if text_proto.size(1) != decoder_feat.size(1):
            raise ValueError(
                'Projected text prototype dim {} does not match decoder '
                'feature channels {}.'.format(text_proto.size(1),
                                              decoder_feat.size(1)))

        # [B, C, H, W] -> [B, H, W, C] for pixel-to-prototype similarity.
        feat_flat = decoder_feat.permute(0, 2, 3, 1)
        if self.normalize_feature:
            feat_flat = F.normalize(feat_flat, dim=-1)
            text_proto = F.normalize(text_proto, dim=-1)

        z_lang = torch.einsum('bhwc,nc->bnhw', feat_flat, text_proto)
        return self.logit_scale * z_lang

    def _get_ccpa_text_features(self):
        pos = self.pos_text_embeddings
        neg = self.neg_text_embeddings

        # T_pos supports [K, D] or [K, M_pos, D].
        pos = self.ccpa_text_proj(pos)
        pos = F.normalize(pos, dim=-1)
        if pos.dim() == 3:
            pos = pos.mean(dim=1)
            pos = F.normalize(pos, dim=-1)
        assert pos.shape == (self.num_classes, self.decoder_channels)

        # T_neg keeps every negative prompt: [K, M_neg, C].
        neg = self.ccpa_text_proj(neg)
        neg = F.normalize(neg, dim=-1)
        assert neg.dim() == 3
        assert neg.size(0) == self.num_classes
        assert neg.size(2) == self.decoder_channels
        return pos, neg

    def _ccpa_eta(self):
        eta = F.softplus(self.ccpa_eta_raw)
        if self.ccpa_eta_type == 'per_class':
            eta = eta.view(1, self.num_classes, 1, 1)
        return eta

    def _ccpa_beta(self):
        beta = F.softplus(self.ccpa_beta_raw)
        if self.ccpa_beta_type == 'per_class':
            beta = beta.view(1, self.num_classes, 1, 1)
        return beta

    def _compute_ccpa_scores(self, align_feat):
        assert align_feat.dim() == 4, (
            'F_align must be [B, C, H, W], got {}'.format(
                tuple(align_feat.shape)))
        assert align_feat.size(1) == self.decoder_channels, (
            'F_align channels {} must equal decoder_channels {}.'.format(
                align_feat.size(1), self.decoder_channels))

        f_norm = F.normalize(align_feat, dim=1)
        t_pos, t_neg = self._get_ccpa_text_features()

        # S_pos: [B, num_classes, H, W]
        s_pos = torch.einsum('bchw,kc->bkhw', f_norm, t_pos)
        # S_neg_all: [B, num_classes, num_neg_prompts, H, W]
        s_neg_all = torch.einsum('bchw,kmc->bkmhw', f_norm, t_neg)
        if self.ccpa_neg_reduce == 'max':
            s_neg = s_neg_all.max(dim=2).values
        else:
            s_neg = s_neg_all.mean(dim=2)
        assert s_neg.shape == s_pos.shape

        # S_contrast: [B, num_classes, H, W]
        s_contrast = s_pos - self._ccpa_eta() * s_neg
        assert s_contrast.shape == s_pos.shape
        return s_pos, s_contrast

    def _compute_presence_logits(self, presence_feat):
        assert self.presence_branch is not None
        assert presence_feat.dim() == 4, (
            'presence source feature must be [B, C, H, W], got {}'.format(
                tuple(presence_feat.shape)))
        presence_logits = self.presence_branch(presence_feat)
        assert presence_logits.shape == (
            presence_feat.size(0), self.num_classes)
        return presence_logits

    def _compute_gate_maps(self, z_visual):
        p_visual = torch.softmax(z_visual, dim=1)
        gate_maps = {}

        if self.use_uncertainty_gate or self.fusion_type == 'u_gate':
            max_prob = p_visual.max(dim=1, keepdim=True)[0]
            uncertainty = 1.0 - max_prob
            gate_maps['uncertainty'] = uncertainty.detach(
            ) if self.detach_gate else uncertainty

        if self.use_boundary_gate:
            dx = torch.abs(p_visual[:, :, :, 1:] - p_visual[:, :, :, :-1])
            dy = torch.abs(p_visual[:, :, 1:, :] - p_visual[:, :, :-1, :])
            dx = F.pad(dx, (0, 1, 0, 0), mode='constant', value=0)
            dy = F.pad(dy, (0, 0, 0, 1), mode='constant', value=0)
            boundary = (dx + dy).mean(dim=1, keepdim=True)
            gate_maps['boundary'] = boundary.detach(
            ) if self.detach_gate else boundary

        if (self.use_foreground_gate and self.background_class_indices):
            background_prob = p_visual[:, self.background_class_indices].sum(
                dim=1, keepdim=True)
            if not bool(torch.isfinite(background_prob).all()):
                raise RuntimeError('Background probability contains NaN/Inf.')
            foreground = 1.0 - background_prob.clamp(0.0, 1.0)
            gate_maps['foreground'] = foreground.detach(
            ) if self.detach_gate else foreground

        return gate_maps

    def _fuse_logits(self, z_visual, z_lang):
        if self.fusion_type == 'direct':
            return z_visual + self.beta * z_lang

        gate_maps = self._compute_gate_maps(z_visual)
        if self.fusion_type == 'u_gate':
            uncertainty = gate_maps['uncertainty']
            return z_visual + self.beta * uncertainty * z_lang

        gate_inputs = []
        if self.use_uncertainty_gate:
            gate_inputs.append(gate_maps['uncertainty'])
        if self.use_boundary_gate:
            gate_inputs.append(gate_maps['boundary'])
        if (self.effective_fusion_type == 'ubm_gate'
                and self.use_foreground_gate
                and self.background_class_indices):
            gate_inputs.append(gate_maps['foreground'])

        gate = torch.sigmoid(self.gate_conv(torch.cat(gate_inputs, dim=1)))
        if self.detach_gate:
            gate = gate.detach()
        return z_visual + self.beta * gate * z_lang

    def _decode_features(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4, 1/8, 1/16, 1/32
        c1, c2, c3, c4 = x

        n, _, _, _ = c4.shape

        # Original SegFormer multi-scale MLP decoder path.
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(
            n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(
            _c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(
            n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(
            _c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(
            n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(
            _c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(
            n, -1, c1.shape[2], c1.shape[3])

        # decoder_feat: [B, decoder_channels, H/4, W/4]
        decoder_feat = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1],
                                                  dim=1))
        return decoder_feat

    def _forward_logits(self, inputs, return_aux=False, return_ccpa=False):
        return_aux = return_aux or return_ccpa
        decoder_feat = self._decode_features(inputs)
        return self._forward_logits_from_features(
            decoder_feat, return_aux=return_aux, return_ccpa=return_ccpa)

    def forward_features(self, inputs):
        return self._decode_features(inputs)

    def _forward_logits_from_features(self,
                                      decoder_feat,
                                      return_aux=False,
                                      return_ccpa=False):
        # Current ULP has no separate cross-attended F_lang tensor, so F_lang
        # falls back to the language-guided decoder feature used for logits.
        f_dec = decoder_feat
        f_lang = decoder_feat
        dropped_feat = self.dropout(decoder_feat)
        z_visual = self.linear_pred(dropped_feat)

        z_lang = self._compute_language_logits(decoder_feat)
        z_final = self._fuse_logits(z_visual, z_lang)
        aux_outputs = dict()
        if self.use_ccpa:
            align_feat = f_dec if self.ccpa_align_feature == 'f_dec' else f_lang
            s_pos, s_contrast = self._compute_ccpa_scores(align_feat)
            aux_outputs.update(s_pos=s_pos, s_contrast=s_contrast)
            if self.use_ccpa_logits:
                z_final = z_final + self._ccpa_beta() * s_contrast

        if self.presence_branch is not None:
            presence_feat = (
                f_dec if self.presence_feature == 'f_dec' else f_lang)
            presence_logits = self._compute_presence_logits(presence_feat)
            aux_outputs['presence_logits'] = presence_logits
            if self.use_presence_inference:
                presence_prob = torch.sigmoid(presence_logits)
                suppress = torch.log(
                    torch.clamp(presence_prob, min=1e-4)).view(
                        z_final.size(0), self.num_classes, 1, 1)
                z_final = z_final + self.presence_suppress_gamma * suppress

        if return_aux:
            return z_final, aux_outputs
        return z_final

    def forward(self, inputs):
        return self._forward_logits(inputs, return_ccpa=False)

    @force_fp32(
        apply_to=('seg_logit', 's_pos', 's_contrast', 'presence_logits'))
    def _losses_with_aux(self,
                         seg_logit,
                         seg_label,
                         s_pos=None,
                         s_contrast=None,
                         presence_logits=None,
                         img_metas=None):
        loss = dict()
        target_size = seg_label.shape[-2:]
        seg_logit = resize(
            input=seg_logit,
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        loss_decode = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        if isinstance(loss_decode, dict):
            loss.update(loss_decode)
        else:
            # Preserve this historical auxiliary-path key for existing PrAda
            # checkpoints/configs. CombinedLoss is the only new dict case.
            loss['loss_seg'] = loss_decode

        if self.use_ccpa and s_pos is not None and s_contrast is not None:
            s_pos = resize(
                input=s_pos,
                size=seg_label.shape[1:],
                mode='bilinear',
                align_corners=self.align_corners)
            s_contrast = resize(
                input=s_contrast,
                size=seg_label.shape[1:],
                mode='bilinear',
                align_corners=self.align_corners)
            loss['loss_ccpa_pos'] = self.lambda_ccpa_pos * F.cross_entropy(
                s_pos, seg_label, ignore_index=self.ignore_index)
            loss['loss_ccpa_contrast'] = (
                self.lambda_ccpa_contrast * F.cross_entropy(
                    s_contrast, seg_label, ignore_index=self.ignore_index))

        if self.use_presence_loss and presence_logits is not None:
            if self.presence_label_source == 'external':
                raise NotImplementedError(
                    'presence_label_source="external" needs dataset meta '
                    'filename support and is intentionally not enabled in '
                    'this first CPR version.')
            presence_labels = build_presence_labels(
                seg_label,
                self.num_classes,
                ignore_index=self.ignore_index,
                min_pixels=self.presence_min_pixels)
            assert presence_logits.shape == presence_labels.shape
            loss['loss_presence'] = (
                self.lambda_presence * F.binary_cross_entropy_with_logits(
                    presence_logits, presence_labels))
            presence_pred = (torch.sigmoid(presence_logits) >=
                             self.presence_threshold).float()
            loss['presence_acc'] = (
                presence_pred.eq(presence_labels).float().mean())

        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        if not self.use_ccpa and not self.use_presence_loss:
            return super(ULPSegFormerHead, self).forward_train(
                inputs, img_metas, gt_semantic_seg, train_cfg)

        seg_logits, aux_outputs = self._forward_logits(
            inputs, return_aux=True)
        return self._losses_with_aux(
            seg_logits,
            gt_semantic_seg,
            s_pos=aux_outputs.get('s_pos'),
            s_contrast=aux_outputs.get('s_contrast'),
            presence_logits=aux_outputs.get('presence_logits'),
            img_metas=img_metas)
