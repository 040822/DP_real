import dill
import torch
import random
import pathlib
from typing import Dict
from torch.optim import AdamW
from omegaconf import DictConfig
from lightning.pytorch import LightningModule
from source.policy.ddp3_fine import Fine_DP3
from source.policy.ddp3_coarse import Coarse_DP3
from source.common.pytorch_util import dict_apply
from source.model.common.lr_scheduler import get_scheduler
from source.model.common.normalizer import LinearNormalizer

class DDP3(LightningModule):
    """
    Double_DP3 是一个分层策略类，结合了粗粒度扩散策略（coarse_dp）和细粒度扩散策略（fine_dp），用于三维环境中的序列决策。
    参数:
        coarse_dp (Coarse_DP3): 粗粒度扩散策略实例。
        fine_dp (Fine_DP3): 细粒度扩散策略实例。
        debug (bool, 可选): 如果为 True，则启用调试打印。默认值为 False。
        **kwargs: 传递给基类策略的其他关键字参数。
    属性:
        coarse_dp (Coarse_DP3): 粗粒度策略。
        fine_dp (Fine_DP3): 细粒度策略。
        coarse_cache (dict 或 None): 粗粒度 DP 推理结果的缓存。
        idx (list 或 None): 用于批次分割的索引列表。
        coarse_cache_idx (int): 粗粒度缓存的当前索引。
        coarse_ratio (float): 粗粒度和细粒度损失的加权比例。
        debug (bool): 调试标志。
    方法:
        set_normalizer(coarse_normalizer, fine_normalizer=None):
            设置粗粒度和细粒度 DP 的归一化器。如果未提供 fine_normalizer，则两者均使用 coarse_normalizer。
        reset():
            清除粗粒度 DP 的缓存并重置缓存索引。
        predict_action(obs_dict, sample_idx=None):
            使用粗粒度和细粒度 DP 进行分层动作预测。
        predict_action_coarse(obs_dict):
            仅返回粗粒度 DP 的动作预测（用于测试）。
        predict_action_fine(obs_dict):
            使用粗粒度 DP 的输出作为输入，仅返回细粒度 DP 的动作预测（用于测试）。
        compute_loss(batch, only_coarse=True):
            计算粗粒度和细粒度 DP 的训练损失。如果 only_coarse 为 True，则仅计算粗粒度损失。
        coarse_compute_loss(batch):
            仅计算粗粒度 DP 的损失。
        fine_compute_loss(batch):
            仅计算细粒度 DP 的损失。
        set_idx(idx):
            设置用于批次分割的索引列表。
        split_batch(batch):
            根据 self.idx 将输入批次分割为粗粒度和细粒度批次，并生成样本/历史索引。
        load_coarse_dp_model():
            从检查点加载粗粒度 DP 模型。
        load_fine_dp_model():
            从检查点加载细粒度 DP 模型。
        load_payload(payload, exclude_keys=None, include_keys=None, model_name=None, **kwargs):
            从 payload 字典加载模型 state_dicts 和 pickles。
            Loads model state_dicts and pickles from a payload dictionary.
        load_checkpoint(path=None, tag='latest', exclude_keys=None, include_keys=None, model_name=None, **kwargs):
            Loads a checkpoint file and applies its payload to the policy.
    """
    def __init__(self, 
            optimizer_cfg: DictConfig,
            scheduler_cfg: DictConfig,
            num_epochs_coarse: int,

            coarse_dp: Coarse_DP3,
            fine_dp: Fine_DP3,
            debug: bool = False,
            **kwargs):
        super().__init__()

        self.num_epochs_coarse = num_epochs_coarse
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.coarse_dp: Coarse_DP3 = coarse_dp
        self.fine_dp: Fine_DP3 = fine_dp
            
        self.coarse_cache = None
        self.idx = None
        
        self.coarse_cache_idx = 0
        self.coarse_ratio = 0.8 # coarse和fine的loss的比例
        self.debug = debug

        self.reset()  # 初始化缓存
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    # ========= set  ============
    def set_normalizer(self, coarse_normalizer: LinearNormalizer, fine_normalizer: LinearNormalizer=None):
        # normalized_data = (data - mean) / std
        if fine_normalizer is None:
            # 如果没有提供fine_normalizer，则fine DP使用coarse_normalizer
            fine_normalizer = coarse_normalizer
        self.coarse_dp.set_normalizer(coarse_normalizer)
        self.fine_dp.set_normalizer(fine_normalizer)

        
    def set_idx(self, idx):
        # 用于传入self.idx
        self.idx = idx

    # ========= inference  ============
    def reset(self):
        """
        清除缓存,每次跑完env后都需要清除缓存。
        """
        if self.coarse_cache is not None:
            del self.coarse_cache
        self.coarse_cache = None
        self.coarse_cache_idx = 0

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], sample_idx=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        sample_idx: optional, if provided, will use this index to select the coarse DP action.
        result: must include "action" key
        """

        # 获取维度信息
        value = next(iter(obs_dict.values()))
        B, To = value.shape[:2] # B：批次大小
        T = self.fine_dp.horizon
        Da = self.fine_dp.action_dim # Dim of action
        Do = self.fine_dp.obs_feature_dim # Dim of obs
        To = self.fine_dp.n_obs_steps # Number of observation steps

        # Coarse DP推理
        if self.coarse_cache is None:
            result_coarse = self.coarse_dp.predict_action(obs_dict, pre_action=None, history_idxs=self.coarse_cache_idx)
        else:
            result_coarse = self.coarse_dp.predict_action(obs_dict, pre_action=self.coarse_cache['action_pred'], history_idxs=self.coarse_cache_idx)
        self.coarse_cache = result_coarse
            
        # Fine DP推理
        if self.debug:
            print(f"[Double_DP3 predict_action] coarse_cache: {self.coarse_cache['action'].shape}")
        
        # build input
        device = self.device
        dtype = self.dtype
        pre_action = torch.zeros(size=(B, T, Da), device=device, dtype=dtype) # 产生fine DP的输入动作。
        pre_action[:, 1, :] = self.coarse_cache['action'][:, self.coarse_cache_idx, :] 
        pre_action[:, -1, :] = self.coarse_cache['action'][:, self.coarse_cache_idx+1, :]
        
        act_position = torch.tensor(self.coarse_cache_idx+1, device=device)
        # 注意：coarse_cache_idx+1主要是因为coarse_cache_idx是从0开始，sample_idx是从1开始的。
        action = self.fine_dp.predict_action(
            obs_dict=obs_dict, 
            pre_action=pre_action, 
            act_position=act_position)
        
        self.coarse_cache_idx +=1
        if self.coarse_cache_idx >= self.coarse_cache['action'].shape[1]-1:
            self.coarse_cache = None
            self.coarse_cache_idx = 0
            
        return action

    # ========= training  ============
    def compute_loss(self, batch: Dict[str, torch.Tensor], only_coarse: True) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        B = batch['action'].shape[0] # B：批次大小
        T = self.fine_dp.horizon
        Da = self.fine_dp.action_dim # Dim of action
        Do = self.fine_dp.obs_feature_dim # Dim of obs
        To = self.fine_dp.n_obs_steps # Number of observation steps
        
        # build input
        device = self.device
        dtype = self.dtype
        
        # dataset design 
        coarse_batch, refine_batch, sample_idx, history_idxs = self.split_batch(batch)

        coarse_batch = dict_apply(coarse_batch, lambda x: x.to(device, non_blocking=True))
        refine_batch = dict_apply(refine_batch, lambda x: x.to(device, non_blocking=True))

        # compute loss
        coarse_loss, coarse_loss_dict = self.coarse_dp.compute_loss(coarse_batch, history_idxs=history_idxs)

        if only_coarse:
            return coarse_loss, coarse_loss_dict, None
        
        coarse_action = coarse_loss_dict["pred"] # coarse_action: [B, 16, Da]
        if self.debug:
           print(f"[Double_DP3 compute_loss] coarse_action: {coarse_action.shape}")

        pre_action = torch.zeros(size=(B, T, Da), device=device, dtype=dtype) # 产生fine DP的输入动作。
        for i in range(B):
            pre_action[i, 1, :] = coarse_action[i, sample_idx[i], :]
            pre_action[i, -1, :] = coarse_action[i, sample_idx[i]+1, :]

        if self.debug:
           print(f"sample_idx: {sample_idx}, coarse_action: {coarse_action.shape}, pre_action: {pre_action.shape}")
        # sample_idx 为从 1 到 14 (idx_len-2) 抽取的索引，0和15弃用
        # idx=1 对应 coarse_batch 的第一个动作，即 coarse_batch['action'][:, 1, :] , 注意coarse_batch['action'][:, 0, :]是obs的真实动作，1~15才是coarse预测的动作。
        
        # 注意：refine_batch 和 pre_action 的 idx 对齐
        act_position = torch.tensor(sample_idx, device=device)
        refine_loss , refine_loss_dict = self.fine_dp.compute_loss(batch=refine_batch, pre_action=pre_action, act_position=act_position)
        if self.debug:
            print(f"[Double_DP3 compute_loss] coarse_loss: {coarse_loss}, refine_loss: {refine_loss}")
        
        loss = self.coarse_ratio * coarse_loss + (1 - self.coarse_ratio) * refine_loss
        return loss, coarse_loss_dict, refine_loss_dict
        
    # ========= data process  ============
 
    def split_batch(self, batch):
        # 根据 self.idx 分割 batch
        nobs = batch['obs'] # B,To,Do
        nactions = batch['action'] # B,T,Da
        
        # 获取维度信息
        batch_coarse = {}
        batch_fine = {}
        idx_len = len(self.idx)
        batch_size = nactions.shape[0] # 获取批次大小
        device = nactions.device
        batch_indices = torch.arange(batch_size)
        
        # 生成 history_idx, 用于coarse DP
        history_idxs = [random.randint(0, idx_len - 2) for _ in range(batch_size)] # 随机生成每个样本的history_idx，范围是0到idx_len-2。
        # 注意：history_idx=0代表当前没有历史动作，从头训练。
        
        # 生成 sample_idx, 用于fine DP
        sample_idxs = [random.randint(1, idx_len - 2) for _ in range(batch_size) ] # 1到idx_len-2之间抽取一个索引。去掉0和最后一个idx
        fine_start = [self.idx[sample_idx] - 1 for sample_idx in sample_idxs] # 为了加上先前的obs，因此 -1
        fine_end = [self.idx[sample_idx+1] + 1 for sample_idx in sample_idxs] # 为了加上idx[sample_idx+1]之后的action，因此 +1
        # 设sample_idx为1，则fine_start=0，fine_end=11，fine_start:fine_end=0~10,正好11帧。
        
        # 获取first_obs_idxs，用于coarse DP
        first_obs_idxs = [self.idx[i+1]-1 for i in history_idxs]  # 获取第一帧obs的索引
        first_obs_idxs = torch.tensor(first_obs_idxs, device=device)
        
        # 分割batch['action']
        batch_coarse['action'] = nactions[:, self.idx]
        batch_coarse['action'][batch_indices, 0, :] = nactions[batch_indices, first_obs_idxs, :] # 将第一帧obs的动作放在初始帧
        batch_fine['action'] = torch.stack([nactions[i, fine_start[i]:fine_end[i]] for i in range(batch_size)])

        # 分割batch['obs']
        batch_coarse['obs'] = {}
        batch_fine['obs'] = {}
        for kk in nobs:
            v = nobs[kk]
            batch_coarse['obs'][kk] = v[:, self.idx]
            batch_coarse['obs'][kk][batch_indices, 0, :] = v[batch_indices, first_obs_idxs, :] # 将第一帧obs的动作放在初始帧
            batch_fine['obs'][kk] = torch.stack([v[i, fine_start[i]:fine_end[i]] for i in range(batch_size)])

        return batch_coarse, batch_fine, sample_idxs, history_idxs


    # ========= trainer  ============
    def training_step(self, batch, batch_idx):
        raw_loss, coarse_loss_dict, fine_loss_dict = self.compute_loss(batch, only_coarse=(self.current_epoch<self.num_epochs_coarse))
        self.log('train/loss', raw_loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/mse_loss', coarse_loss_dict['mse_loss'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        if fine_loss_dict is not None:
            self.log('train/fine_loss', fine_loss_dict['loss'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return raw_loss
    
    def validation_step(self, batch, batch_idx):
        raw_loss, coarse_loss_dict, fine_loss_dict = self.compute_loss(batch, only_coarse=(self.current_epoch<self.num_epochs_coarse))
        self.log('val/loss', raw_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/mse_loss', coarse_loss_dict['mse_loss'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if fine_loss_dict is not None:
            self.log('val/fine_loss', fine_loss_dict['loss'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return raw_loss

    def configure_optimizers(self):
        optim_groups = self.coarse_dp.model.get_optim_groups(
            weight_decay=self.optimizer_cfg.transformer_weight_decay)
        optim_groups.append({
            "params": self.coarse_dp.obs_encoder.parameters(),
            "weight_decay": self.optimizer_cfg.obs_encoder_weight_decay
        })
        
        optim_groups = optim_groups + self.fine_dp.model.get_optim_groups(
            weight_decay=self.optimizer_cfg.transformer_weight_decay)
        optim_groups.append({
            "params": self.fine_dp.obs_encoder.parameters(),
            "weight_decay": self.optimizer_cfg.obs_encoder_weight_decay
        })

        optimizer = AdamW(
            optim_groups,
            lr=self.optimizer_cfg.lr, 
            betas=self.optimizer_cfg.betas
        )
        lr_scheduler = get_scheduler(
            self.scheduler_cfg.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.scheduler_cfg.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }