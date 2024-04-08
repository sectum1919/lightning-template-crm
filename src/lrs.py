import torch.optim.lr_scheduler as lrs
import numpy as np      
            
class WarmupCosineScheduler(lrs.LRScheduler):
    def __init__(self, optimizer, warmup_epochs, num_epochs, iter_per_epoch, last_epoch=-1, verbose=False):
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch
        super().__init__(optimizer, last_epoch, verbose)
        print(f'self.warmup_iter={self.warmup_iter}\nself.total_iter={self.total_iter}')
        # self.init_lr()  # so that at first step we have the correct step size

    def get_lr(self):
        if self._step_count < self.warmup_iter:
            return [base_lr * self._step_count / self.warmup_iter for base_lr in self.base_lrs]
        else:
            decay_iter = self.total_iter - self.warmup_iter
            return [
                0.5
                * base_lr
                * (1 + np.cos(np.pi * (self._step_count - self.warmup_iter) / decay_iter))
                for base_lr in self.base_lrs
            ]
            
class WarmupConstantScheduler(lrs.LRScheduler):
    def __init__(self, optimizer, warmup_epochs, num_epochs, iter_per_epoch, last_epoch=-1, verbose=False):
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch
        super().__init__(optimizer, last_epoch, verbose)
        print(f'self.warmup_iter={self.warmup_iter}\nself.total_iter={self.total_iter}')
        # self.init_lr()  # so that at first step we have the correct step size

    def get_lr(self):
        if self._step_count < self.warmup_iter:
            return [base_lr * self._step_count / self.warmup_iter for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]
