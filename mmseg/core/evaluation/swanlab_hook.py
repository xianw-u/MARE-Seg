import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook


def _unwrap_batch_item(x):
    if hasattr(x, 'data'):
        x = x.data
    while isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    return x


class SwanLabHook(Hook):
    """Minimal SwanLab logger hook for iter-based training."""

    def __init__(self, interval=50, init_kwargs=None):
        self.interval = interval
        self.init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
        self.log_eval_images = self.init_kwargs.pop('log_eval_images', True)
        self.num_eval_images = int(self.init_kwargs.pop('num_eval_images', 4))
        self.image_interval = self.init_kwargs.pop('image_interval', None)
        self._swanlab = None
        self._enabled = True
        self._eval_hook = None
        self._logged_eval_iters = set()

    def before_run(self, runner):
        if hasattr(runner, 'rank') and runner.rank != 0:
            self._enabled = False
            return
        try:
            import swanlab
        except ImportError:
            runner.logger.warning(
                'SwanLab is not installed. Logging is disabled. '
                'Install it with `pip install swanlab` to enable logging.')
            self._enabled = False
            return

        self._swanlab = swanlab
        init_kwargs = dict(self.init_kwargs)
        init_kwargs.setdefault(
            'experiment_name', runner.meta.get('exp_name', 'mmseg-run'))
        init_kwargs.setdefault('config', {'work_dir': runner.work_dir})
        self._swanlab.init(**init_kwargs)
        self._eval_hook = self._find_eval_hook(runner)

    def _collect_scalars(self, runner):
        payload = {}
        output = dict(getattr(runner.log_buffer, 'output', {}) or {})
        train_log_vars = {}
        runner_outputs = getattr(runner, 'outputs', None)
        if isinstance(runner_outputs, dict):
            train_log_vars = runner_outputs.get('log_vars', {}) or {}
        for key, value in train_log_vars.items():
            if isinstance(value, (int, float)):
                output.setdefault(key, value)
        for key, value in output.items():
            if not isinstance(value, (int, float)):
                continue
            if key in ('mIoU', 'mAcc', 'aAcc'):
                payload[f'val/{key}'] = float(value)
            else:
                payload[f'train/{key}'] = float(value)
        payload['meta/iter'] = runner.iter + 1
        return payload

    def _find_eval_hook(self, runner):
        for hook in getattr(runner, 'hooks', []):
            if hasattr(hook, 'dataloader'):
                return hook
        return None

    def _should_log_eval_images(self, runner):
        if not self.log_eval_images or self._eval_hook is None:
            return False
        current_iter = runner.iter + 1
        if current_iter in self._logged_eval_iters:
            return False
        if self.image_interval is None:
            return True
        return current_iter % int(self.image_interval) == 0

    def _colorize_mask(self, mask, palette):
        palette = np.array(palette, dtype=np.uint8)
        color_mask = np.zeros(mask.shape + (3, ), dtype=np.uint8)
        valid_mask = mask != 255
        for class_id, color in enumerate(palette):
            color_mask[mask == class_id] = color
        color_mask[~valid_mask] = 0
        return color_mask

    def _blend_mask(self, image_rgb, mask, palette, alpha=0.5):
        color_mask = self._colorize_mask(mask, palette)
        blended = (image_rgb.astype(np.float32) * (1 - alpha) +
                   color_mask.astype(np.float32) * alpha)
        return blended.astype(np.uint8)

    def _make_swanlab_image(self, image, caption):
        if hasattr(self._swanlab, 'Image'):
            return self._swanlab.Image(image, caption=caption)
        return image

    def _collect_eval_visuals(self, runner):
        eval_dataset = self._eval_hook.dataloader.dataset
        palette = getattr(eval_dataset, 'PALETTE', None)
        if palette is None:
            return {}

        model = runner.model
        module = model.module if hasattr(model, 'module') else model
        was_training = module.training
        module.eval()

        visuals = []
        max_images = min(self.num_eval_images, len(eval_dataset))
        gt_masks = None
        if getattr(eval_dataset, 'ann_dir', None) is not None:
            gt_masks = eval_dataset.get_gt_seg_maps()
        device = next(module.parameters()).device

        for idx in range(max_images):
            sample = eval_dataset[idx]
            data = collate([sample], samples_per_gpu=1)
            if device.type == 'cuda':
                data = scatter(
                    data, [device.index if device.index is not None else 0])[0]

            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            img_meta = _unwrap_batch_item(data['img_metas'])
            img_path = img_meta.get('filename')
            if img_path is None:
                img_path = osp.join(eval_dataset.img_dir,
                                    eval_dataset.img_infos[idx]['filename'])

            image_bgr = mmcv.imread(img_path)
            image_rgb = mmcv.bgr2rgb(image_bgr)
            pred_mask = result[0].astype(np.uint8)
            pred_overlay = self._blend_mask(image_rgb, pred_mask, palette)

            gt_mask = gt_masks[idx] if gt_masks is not None else None
            gt_overlay = (self._blend_mask(image_rgb, gt_mask, palette)
                          if gt_mask is not None else None)

            base_name = osp.basename(img_meta.get('ori_filename', img_path))
            visuals.append(
                self._make_swanlab_image(image_rgb, f'{base_name} | image'))
            visuals.append(
                self._make_swanlab_image(pred_overlay,
                                         f'{base_name} | prediction'))
            if gt_overlay is not None:
                visuals.append(
                    self._make_swanlab_image(gt_overlay,
                                             f'{base_name} | gt'))

        if was_training:
            module.train()

        if not visuals:
            return {}
        return {'val/segmentation_examples': visuals}

    def after_train_iter(self, runner):
        if not self._enabled or self._swanlab is None:
            return
        runner_outputs = getattr(runner, 'outputs', None)
        train_log_vars = {}
        if isinstance(runner_outputs, dict):
            train_log_vars = runner_outputs.get('log_vars', {}) or {}
        log_output = getattr(runner.log_buffer, 'output', {}) or {}
        has_eval_metrics = any(
            key in log_output for key in ('mIoU', 'mAcc', 'aAcc'))
        if not self.every_n_iters(runner, self.interval) and not has_eval_metrics:
            return
        if not has_eval_metrics and not any(
                isinstance(v, (int, float)) for v in train_log_vars.values()):
            return
        payload = self._collect_scalars(runner)
        if has_eval_metrics and self._should_log_eval_images(runner):
            payload.update(self._collect_eval_visuals(runner))
            self._logged_eval_iters.add(runner.iter + 1)
        if payload:
            self._swanlab.log(payload, step=runner.iter + 1)

    def after_run(self, runner):
        if self._enabled and self._swanlab is not None:
            self._swanlab.finish()
