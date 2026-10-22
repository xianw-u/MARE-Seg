import json
import os.path as osp

import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler


def _norm_key(path):
    return osp.normpath(path).replace('\\', '/').lower()


def _unwrap_repeat_dataset(dataset):
    if hasattr(dataset, 'dataset') and hasattr(dataset, '_ori_len'):
        return dataset.dataset, dataset._ori_len
    return dataset, len(dataset)


class WeakClassBalancedSampler(Sampler):
    """Weighted sampler for images containing weak classes."""

    def __init__(self,
                 dataset,
                 sampling_cache,
                 num_samples=None,
                 replacement=True,
                 seed=0,
                 strict=False):
        self.dataset = dataset
        self.base_dataset, self.base_len = _unwrap_repeat_dataset(dataset)
        self.rank, self.world_size = get_dist_info()
        self.replacement = replacement
        self.seed = int(seed or 0)
        self.epoch = 0
        self.strict = strict
        self.num_samples = int(num_samples) if num_samples is not None else int(
            (len(dataset) + self.world_size - 1) // self.world_size)
        self.total_size = self.num_samples * self.world_size
        self.weights = self._load_weights(sampling_cache)

    def _load_weights(self, sampling_cache):
        if not osp.isfile(sampling_cache):
            raise FileNotFoundError(
                'Sampling cache does not exist: {}'.format(sampling_cache))
        with open(sampling_cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        cache_weights = {}
        for sample in cache.get('samples', []):
            for key_name in ('seg_map', 'seg_map_path', 'filename'):
                if key_name in sample:
                    cache_weights[_norm_key(sample[key_name])] = float(
                        sample.get('weight', 1.0))

        weights = []
        missing = []
        ann_dir = getattr(self.base_dataset, 'ann_dir', None)
        for info in self.base_dataset.img_infos:
            seg_map = info.get('ann', {}).get('seg_map', info.get('filename'))
            candidates = [_norm_key(seg_map)]
            if ann_dir is not None:
                candidates.append(_norm_key(osp.join(ann_dir, seg_map)))
            weight = None
            for key in candidates:
                if key in cache_weights:
                    weight = cache_weights[key]
                    break
            if weight is None:
                missing.append(seg_map)
                weight = 1.0
            weights.append(weight)
        if self.strict and missing:
            raise KeyError(
                '{} dataset samples are missing in sampling cache, first: {}'.
                format(len(missing), missing[0]))
        base_weights = torch.tensor(weights, dtype=torch.double)
        repeated_indices = torch.arange(len(self.dataset)) % self.base_len
        return base_weights[repeated_indices]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.replacement:
            indices = torch.multinomial(
                self.weights,
                self.total_size,
                replacement=True,
                generator=generator).tolist()
        else:
            perm = torch.randperm(
                len(self.weights), generator=generator).tolist()
            if len(perm) < self.total_size:
                repeats = (self.total_size + len(perm) - 1) // len(perm)
                perm = (perm * repeats)[:self.total_size]
            indices = perm[:self.total_size]
        return iter(indices[self.rank:self.total_size:self.world_size])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def build_sampler(cfg, dataset):
    cfg = dict(cfg)
    sampler_type = cfg.pop('type')
    if sampler_type != 'WeakClassBalancedSampler':
        raise KeyError('Unsupported sampler type: {}'.format(sampler_type))
    return WeakClassBalancedSampler(dataset=dataset, **cfg)
