import math
from typing import Any

import os
from pathlib import Path

import torch
import torchmetrics
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.callbacks import BasePredictionWriter

from lrs import WarmupCosineScheduler, WarmupConstantScheduler
from resnet_espnet import BasicBlock, ResNet

class ResNet18(torch.nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.model = torch.nn.Sequential()
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=args.relu_type)
        
    def forward(self, input):
        B, T, C = input.shape
        feature = self.model(input)
        return feature
    
class ResNetModule(pl.LightningModule):
    def __init__(self, **cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.model = ResNet(self.hparams)
        
    def forward(self, input):
        result = self.model(input)
        return result
    
    def _step(self, batch, batch_idx):
        input = batch['input']
        pred = self(input) # this will call self.forward(input)
        targ = batch['label']
        ce_loss = torch.nn.functional.cross_entropy(input=pred, target=targ, reduction='mean')
        loss = ce_loss
        loss_dict = {
            'ce_loss': ce_loss,
        }
        if torch.isinf(loss):
            import pdb
            pdb.set_trace()
        return pred, loss, loss_dict

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred, loss, loss_dict = self._step(batch, batch_idx)
        # logging train losses
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f'train_{k}', v, on_step=True, on_epoch=True, prog_bar=False)
        self.log('monitor_step', self.global_step)
        # update train metrics
        self.train_acc.update(pred, batch['label'])
        self.train_auc.update(pred, batch['label'])
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred, loss, loss_dict = self._step(batch, batch_idx)
        # logging valid losses
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('monitor_step', self.global_step)
        for k, v in loss_dict.items():
            self.log(f'valid_{k}', v, on_step=False, on_epoch=True, prog_bar=False)
        # update valid metrics
        self.valid_acc.update(pred, batch['label'])
        self.valid_auc.update(pred, batch['label'])
        return loss
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        pred, loss, loss_dict = self._step(batch, batch_idx)
        return None

    def predict_step(self, batch, batch_idx):
        pred, loss, loss_dict = self._step(batch, batch_idx)
        return {
            'uid': batch['uid'],
            'pred': pred,
            'targ': batch['label'],
        }
    
    def on_train_epoch_start(self) -> None:
        self.train_acc.reset()
        self.train_auc.reset()
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.train_acc._update_count > 0:
            self.log('train_acc', self.train_acc.compute(), on_step=False, on_epoch=True, prog_bar=False)
        if self.train_auc._update_count > 0:
            self.log('train_auc', self.train_auc.compute(), on_step=False, on_epoch=True, prog_bar=False)
        return super().on_train_epoch_end()
    
    def on_validation_epoch_start(self) -> None:
        self.valid_acc.reset()
        self.valid_auc.reset()
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self) -> None:
        if self.valid_acc._update_count > 0:
            self.log('valid_acc', self.valid_acc.compute(), on_step=False, on_epoch=True, prog_bar=False)
        if self.valid_auc._update_count > 0:
            self.log('valid_auc', self.valid_auc.compute(), on_step=False, on_epoch=True, prog_bar=False)
        return super().on_validation_epoch_end()
    
    def on_test_epoch_start(self) -> None:
        return super().on_test_epoch_start()
    
    def on_test_epoch_end(self) -> None:
        return super().on_test_epoch_end()
    
    def configure_evaluation(self):
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.train_auc = torchmetrics.AUROC(task='binary')
        self.valid_acc = torchmetrics.Accuracy(task='binary')
        self.valid_auc = torchmetrics.AUROC(task='binary')
        self.metrics = {
            'train_acc': self.train_acc,
            'train_auc': self.train_auc,
            'valid_acc': self.valid_acc,
            'valid_auc': self.valid_auc,
        }
    
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 1e-2
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=weight_decay, betas=(0.9, 0.98))
        # lrs
        devices_count = int(self.hparams.devices)
        samples_per_iter = self.hparams.accumulate_grad_batches * devices_count
        samples_per_batch = len(self.trainer.datamodule.train_dataloader())
        iter_per_epoch = math.ceil(samples_per_batch / samples_per_iter)
        if self.hparams.lrs == 'warmup_cosine':
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=int(self.hparams.max_epochs * self.hparams.warm_up_ratio),
                num_epochs=self.hparams.max_epochs,
                iter_per_epoch=iter_per_epoch,
            )
        elif self.hparams.lrs == 'warmup_constant':
            scheduler = WarmupConstantScheduler(
                optimizer,
                warmup_epochs=int(self.hparams.max_epochs * self.hparams.warm_up_ratio),
                num_epochs=self.hparams.max_epochs,
                iter_per_epoch=iter_per_epoch,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
      
class ResultWriter(BasePredictionWriter):
    def __init__(self, output_dir, output_name, write_interval, data_config):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.data_config = data_config
        self.output_name = output_name

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        save_feature = os.path.join(self.output_dir, f"predictions_{self.output_name}_{trainer.global_rank}.npy")
        Path(os.path.dirname(os.path.abspath(save_feature))).mkdir(exist_ok=True, parents=True)
        infos = {
            'uid': [],
            'pred': [],
            'target': [],
        }
        for batch in predictions: # this batch structure is according to your ResNetModule.predict_step()
            uids = batch['uid']
            for i, uid in enumerate(uids):
                infos['uid'].append(batch['uid'][i])
                infos['pred'].append(batch['pred'][i])
                infos['targ'].append(batch['targ'][i])
        np.save(save_feature, infos)
        print(f'predicted files saved in {os.path.abspath(save_feature)}')