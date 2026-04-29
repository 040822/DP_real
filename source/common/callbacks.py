import torch
import itertools
from copy import deepcopy
from collections import OrderedDict
from lightning.pytorch.callbacks.callback import Callback 
from lightning.pytorch.callbacks import EMAWeightAveraging
from torch.optim.swa_utils import AveragedModel


class ModelAveragingCallback(EMAWeightAveraging):

    def setup(self, trainer, pl_module, stage):
        # 覆写setup方法，避免在此处创建平均模型，因为此时pl_module可能还在CPU上，导致设备不匹配问题
        # 延迟到 on_fit_start，确保 pl_module 已在正确设备上
        pass  

    def on_fit_start(self, trainer, pl_module):
        # 此时 pl_module 已被移到 GPU，device 与训练模型一致，指针交换安全
        device = self._device or pl_module.device
        self._average_model = AveragedModel(model=pl_module, device=device, use_buffers=self._use_buffers, **self._kwargs)

    @torch.no_grad()
    def _swap_models(self, pl_module):
        first_avg_param = next(itertools.chain(self._average_model.module.parameters(), self._average_model.module.buffers()), None)
        first_cur_param = next(itertools.chain(pl_module.parameters(), pl_module.buffers()), None)
        # print(f"[_swap_models] _average_model device={first_avg_param.device if first_avg_param is not None else 'N/A'}, pl_module device={first_cur_param.device if first_cur_param is not None else 'N/A'}")
        for avg_param, cur_param in zip(
            itertools.chain(self._average_model.module.parameters(), self._average_model.module.buffers()),
            itertools.chain(pl_module.parameters(), pl_module.buffers())
        ):
            avg_param.data, cur_param.data = cur_param.data, avg_param.data


class SaveConfigCallback(Callback):
    def __init__(self, cfg):
        self.cfg = cfg

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 保存 cfg 到 checkpoint 字典中
        checkpoint['cfg'] = self.cfg
