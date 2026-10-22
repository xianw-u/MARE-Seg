import copy

import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ExpMomentumEMAHook(Hook):
    """Exponential moving average hook for MMCV 1.x runners."""

    def __init__(self, momentum=0.0001, update_buffers=True):
        self.momentum = float(momentum)
        self.update_buffers = update_buffers
        self.ema_state = None
        self.model_state_backup = None

    @staticmethod
    def _model(runner):
        return runner.model.module if hasattr(runner.model,
                                             'module') else runner.model

    def before_run(self, runner):
        model = self._model(runner)
        self.ema_state = {}
        for name, tensor in model.state_dict().items():
            if tensor.is_floating_point() or self.update_buffers:
                self.ema_state[name] = tensor.detach().clone()

    def after_train_iter(self, runner):
        model = self._model(runner)
        model_state = model.state_dict()
        for name, ema_tensor in self.ema_state.items():
            tensor = model_state[name]
            if not tensor.is_floating_point():
                if self.update_buffers:
                    ema_tensor.copy_(tensor)
                continue
            ema_tensor.mul_(1.0 - self.momentum).add_(
                tensor.detach(), alpha=self.momentum)

    def before_val_epoch(self, runner):
        self._swap_to_ema(runner)

    def after_val_epoch(self, runner):
        self._restore_model(runner)

    def before_test_epoch(self, runner):
        self._swap_to_ema(runner)

    def after_test_epoch(self, runner):
        self._restore_model(runner)

    def _swap_to_ema(self, runner):
        if self.ema_state is None:
            return
        model = self._model(runner)
        self.model_state_backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.ema_state, strict=False)

    def _restore_model(self, runner):
        if self.model_state_backup is None:
            return
        model = self._model(runner)
        model.load_state_dict(self.model_state_backup, strict=False)
        self.model_state_backup = None
