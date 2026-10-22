norm_cfg = {'type': 'BN', 'requires_grad': True}

find_unused_parameters = True

model = {'type': 'EncoderDecoder',
 'pretrained': 'pretrained/mit_b0.pth',
 'backbone': {'type': 'mit_b0', 'style': 'pytorch'},
 'decode_head': {'type': 'MARESegHead',
                 'in_channels': [32, 64, 160, 256],
                 'in_index': [0, 1, 2, 3],
                 'feature_strides': [4, 8, 16, 32],
                 'channels': 128,
                 'dropout_ratio': 0.1,
                 'num_classes': 5,
                 'norm_cfg': {'type': 'BN', 'requires_grad': True},
                 'align_corners': False,
                 'decoder_params': {'embed_dim': 256},
                 'loss_decode': {'type': 'CrossEntropyLoss',
                                 'use_sigmoid': False,
                                 'loss_weight': 1.0},
                 'text_prototype_path': 'data/class_embeddings/dut_class_embeddings.pt',
                 'beta': 0.1,
                 'fusion_type': 'ubm_gate',
                 'background_index': 0,
                 'use_uncertainty_gate': True,
                 'use_boundary_gate': True,
                 'use_foreground_gate': True,
                 'normalize_feature': True,
                 'learnable_prototype': True,
                 'logit_scale': 1.0,
                 'detach_gate': True,
                 'use_visual_prototype': True,
                 'num_visual_prototypes': 1,
                 'prototype_momentum': 0.99,
                 'prototype_warmup_iters': 1000,
                 'prototype_min_pixels': 10,
                 'prototype_beta': 0.05,
                 'learnable_prototype_beta': True,
                 'prototype_logit_scale': 1.0,
                 'use_prototype_aux_loss': True,
                 'prototype_loss_weight': 0.1,
                 'strict_prototype_inference': True,
                 'use_agreement_gate': False,
                 'use_confusion_loss': False,
                 'use_class_specific_loss': False},
 'train_cfg': {},
 'test_cfg': {'mode': 'whole'}}

dataset_type = 'DUTDataset'

data_root = 'data/DUT'

img_norm_cfg = {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}

crop_size = (480, 640)

train_pipeline = [{'type': 'LoadImageFromFile'},
 {'type': 'LoadAnnotations', 'reduce_zero_label': False},
 {'type': 'Resize', 'img_scale': (2048, 640), 'ratio_range': (0.5, 2.0)},
 {'type': 'RandomCrop', 'crop_size': (480, 640), 'cat_max_ratio': 0.75},
 {'type': 'RandomFlip', 'prob': 0.5},
 {'type': 'PhotoMetricDistortion'},
 {'type': 'Normalize',
  'mean': [123.675, 116.28, 103.53],
  'std': [58.395, 57.12, 57.375],
  'to_rgb': True},
 {'type': 'Pad', 'size': (480, 640), 'pad_val': 0, 'seg_pad_val': 255},
 {'type': 'DefaultFormatBundle'},
 {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}]

test_pipeline = [{'type': 'LoadImageFromFile'},
 {'type': 'MultiScaleFlipAug',
  'img_scale': (480, 640),
  'flip': False,
  'transforms': [{'type': 'AlignedResize', 'keep_ratio': True, 'size_divisor': 32},
                 {'type': 'RandomFlip'},
                 {'type': 'Normalize',
                  'mean': [123.675, 116.28, 103.53],
                  'std': [58.395, 57.12, 57.375],
                  'to_rgb': True},
                 {'type': 'ImageToTensor', 'keys': ['img']},
                 {'type': 'Collect', 'keys': ['img']}]}]

data = {'samples_per_gpu': 8,
 'workers_per_gpu': 4,
 'train': {'type': 'RepeatDataset',
           'times': 6000,
           'dataset': {'type': 'DUTDataset',
                       'data_root': 'data/DUT',
                       'img_dir': 'images/training',
                       'ann_dir': 'annotations/training',
                       'pipeline': [{'type': 'LoadImageFromFile'},
                                    {'type': 'LoadAnnotations', 'reduce_zero_label': False},
                                    {'type': 'Resize',
                                     'img_scale': (2048, 640),
                                     'ratio_range': (0.5, 2.0)},
                                    {'type': 'RandomCrop',
                                     'crop_size': (480, 640),
                                     'cat_max_ratio': 0.75},
                                    {'type': 'RandomFlip', 'prob': 0.5},
                                    {'type': 'PhotoMetricDistortion'},
                                    {'type': 'Normalize',
                                     'mean': [123.675, 116.28, 103.53],
                                     'std': [58.395, 57.12, 57.375],
                                     'to_rgb': True},
                                    {'type': 'Pad',
                                     'size': (480, 640),
                                     'pad_val': 0,
                                     'seg_pad_val': 255},
                                    {'type': 'DefaultFormatBundle'},
                                    {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}],
                       'split': '../splits/dut_training.txt'}},
 'val': {'type': 'DUTDataset',
         'data_root': 'data/DUT',
         'img_dir': 'images/validation',
         'ann_dir': 'annotations/validation',
         'pipeline': [{'type': 'LoadImageFromFile'},
                      {'type': 'MultiScaleFlipAug',
                       'img_scale': (480, 640),
                       'flip': False,
                       'transforms': [{'type': 'AlignedResize',
                                       'keep_ratio': True,
                                       'size_divisor': 32},
                                      {'type': 'RandomFlip'},
                                      {'type': 'Normalize',
                                       'mean': [123.675, 116.28, 103.53],
                                       'std': [58.395, 57.12, 57.375],
                                       'to_rgb': True},
                                      {'type': 'ImageToTensor', 'keys': ['img']},
                                      {'type': 'Collect', 'keys': ['img']}]}],
         'split': '../splits/dut_validation.txt'},
 'test': {'type': 'DUTDataset',
          'data_root': 'data/DUT',
          'img_dir': 'images/validation',
          'ann_dir': 'annotations/validation',
          'pipeline': [{'type': 'LoadImageFromFile'},
                       {'type': 'MultiScaleFlipAug',
                        'img_scale': (480, 640),
                        'flip': False,
                        'transforms': [{'type': 'AlignedResize',
                                        'keep_ratio': True,
                                        'size_divisor': 32},
                                       {'type': 'RandomFlip'},
                                       {'type': 'Normalize',
                                        'mean': [123.675, 116.28, 103.53],
                                        'std': [58.395, 57.12, 57.375],
                                        'to_rgb': True},
                                       {'type': 'ImageToTensor', 'keys': ['img']},
                                       {'type': 'Collect', 'keys': ['img']}]}],
          'split': '../splits/dut_validation.txt'}}

log_config = {'interval': 2, 'hooks': [{'type': 'TextLoggerHook', 'by_epoch': False}]}

dist_params = {'backend': 'nccl'}

log_level = 'INFO'

load_from = None

resume_from = None

workflow = [('train', 1)]

cudnn_benchmark = True

optimizer = {'type': 'AdamW',
 'lr': 0.0003,
 'betas': (0.9, 0.999),
 'weight_decay': 0.01,
 'paramwise_cfg': {'custom_keys': {'pos_block': {'decay_mult': 0.0},
                                   'norm': {'decay_mult': 0.0},
                                   'head': {'lr_mult': 10.0}}}}

optimizer_config = {}

lr_config = {'policy': 'poly',
 'warmup': 'linear',
 'warmup_iters': 1500,
 'warmup_ratio': 1e-06,
 'power': 1.0,
 'min_lr': 0.0,
 'by_epoch': False}

runner = {'type': 'IterBasedRunner', 'max_iters': 40000}

checkpoint_config = {'by_epoch': False, 'interval': 4000}

evaluation = {'interval': 4000, 'metric': 'mIoU', 'save_best': 'mIoU', 'rule': 'greater'}

swanlab = None

work_dir = 'work_dirs/mare_seg_dut'

gpu_ids = [0]

seed = None
