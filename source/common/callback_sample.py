import torch
import dill
import random
from typing import Any, Optional
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from source.common.pytorch_util import dict_apply
import logging


class SampleCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, policy):
        sample_batch = next(iter(trainer.train_dataloader))

        if sample_batch is None:
            if trainer.is_global_zero:
                logging.getLogger(__name__).warning("No batch found for sampling.")
            return

        policy.eval()
        policy.reset()

        with torch.no_grad():
            device = policy.device
            batch = dict_apply(sample_batch, lambda x: x.to(device, non_blocking=True))
            obs_dict = batch['obs']
            gt_action = batch['action']
            result = policy.predict_action(obs_dict)
            pred_action = result['action_pred']   
            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        
        policy.train()
        self.log(f"train_action_mse_error", mse, on_epoch=True, rank_zero_only=True)

    def on_validation_epoch_end(self, trainer, policy):
        # 注意： trainer只有val_dataloaders属性，没有val_dataloader属性。当使用多个验证集时，需要遍历val_dataloaders列表
        sample_batch = next(iter(trainer.val_dataloaders))
        
        if sample_batch is None:
            if trainer.is_global_zero:
                logging.getLogger(__name__).warning("No batch found for sampling.")
            return
        policy.eval()
        policy.reset()
        
        with torch.no_grad():
            device = policy.device
            batch = dict_apply(sample_batch, lambda x: x.to(device, non_blocking=True))
            obs_dict = batch['obs']
            gt_action = batch['action']
            result = policy.predict_action(obs_dict)
            pred_action = result['action_pred']   
            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        
        policy.train()
        self.log(f"val_action_mse_error", mse, on_epoch=True, rank_zero_only=True)