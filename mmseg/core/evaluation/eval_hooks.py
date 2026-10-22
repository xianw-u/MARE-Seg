import os
import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 by_epoch=False,
                 save_best=None,
                 rule='greater',
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_best = save_best
        self.rule = rule
        self.best_score = None
        self.best_ckpt_path = None
        self.eval_kwargs = eval_kwargs
        if rule not in ['greater', 'less']:
            raise ValueError('rule must be either "greater" or "less"')

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        runner.call_hook('before_val_epoch')
        try:
            results = single_gpu_test(runner.model, self.dataloader, show=False)
        finally:
            runner.call_hook('after_val_epoch')
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        runner.call_hook('before_val_epoch')
        try:
            results = single_gpu_test(runner.model, self.dataloader, show=False)
        finally:
            runner.call_hook('after_val_epoch')
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        self._save_best_checkpoint(runner, eval_res)

    def _save_best_checkpoint(self, runner, eval_res):
        if self.save_best is None:
            return
        if self.save_best not in eval_res:
            raise KeyError(f'{self.save_best} is not in evaluation results: '
                           f'{list(eval_res.keys())}')

        score = eval_res[self.save_best]
        if self.best_score is not None:
            is_better = score > self.best_score if self.rule == 'greater' \
                else score < self.best_score
            if not is_better:
                return

        self.best_score = score
        if self.best_ckpt_path is not None and osp.exists(self.best_ckpt_path):
            os.remove(self.best_ckpt_path)

        filename_tmpl = f'best_{self.save_best}_iter_{{}}.pth'
        filename = filename_tmpl.format(runner.iter + 1)
        runner.save_checkpoint(
            runner.work_dir,
            filename_tmpl=filename_tmpl,
            save_optimizer=False,
            create_symlink=False)
        self.best_ckpt_path = osp.join(runner.work_dir, filename)
        runner.logger.info(
            f'Now best checkpoint is saved as {filename}. '
            f'Best {self.save_best}: {self.best_score:.4f}')


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 save_best=None,
                 rule='greater',
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.save_best = save_best
        self.rule = rule
        self.best_score = None
        self.best_ckpt_path = None
        self.eval_kwargs = eval_kwargs
        if rule not in ['greater', 'less']:
            raise ValueError('rule must be either "greater" or "less"')

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        runner.call_hook('before_val_epoch')
        try:
            results = multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=osp.join(runner.work_dir, '.eval_hook'),
                gpu_collect=self.gpu_collect)
        finally:
            runner.call_hook('after_val_epoch')
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        runner.call_hook('before_val_epoch')
        try:
            results = multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=osp.join(runner.work_dir, '.eval_hook'),
                gpu_collect=self.gpu_collect)
        finally:
            runner.call_hook('after_val_epoch')
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
