from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy

from lightning.pytorch import LightningModule
from model.common.normalizer import LinearNormalizer
from model.diffusion.mask_generator_ddp import CoarseMaskGenerator
from common.pytorch_util import dict_apply
from model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from model.diffusion.editable_diffusion import EditableDiffusion
from model.diffusion.position_network import PositionNetwork

import random
import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint

class Coarse_DP2(LightningModule):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            condition_type="cross_attention_add",
            encoder_output_dim=256,
            crop_shape=None,
            debug=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        
        # 将obs_feature_dim转换为int
        if isinstance(obs_feature_dim, (torch.Size, tuple, list)):
            obs_feature_dim = int(obs_feature_dim[0])
        else:
            obs_feature_dim = int(obs_feature_dim)
        
        position_network = PositionNetwork(
            horizon=horizon,
            hidden_dim=obs_feature_dim,
            action_dim=action_dim,
            debug=debug
        )

        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        
        
        self.debug = debug

        model = EditableDiffusion(
            input_dim=input_dim,
            output_dim=input_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            cond_dim=global_cond_dim,
            n_layer=8,
            n_head=12,
            n_emb=768,
            p_drop_emb=0.1,
            p_drop_attn=0.1,
            debug=debug,
        )
        
        self.obs_encoder = obs_encoder
        self.position_network = position_network
        self.model = model
        self.noise_scheduler = noise_scheduler
        # DDPM scheduler 相当于denoise(采样器）
        
        self.mask_generator = CoarseMaskGenerator(
            action_dim=action_dim,
            max_n_obs_steps=n_obs_steps,
            use_first_action=True,
            debug=debug,
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.edit_loss_ratio = 0.7
        self.edit_action_idx = None
        self.edit_action_enhanced_prob = 0.4

        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            act_position=None,
            edit_action_idx=None,
            edit_action=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        condition_data: 动作
        condition_mask: 真实数据的mask
        condition_data_pc: 没用上
        condition_mask_pc: 没用上
        local_cond: obs_as_global_cond=True时没用上
        global_cond: 全局条件
        generator: 没用上
        """
        # 采样过程，是DP核心。
        # 这部分需要深入理解。
        model = self.model
        scheduler = self.noise_scheduler

        # 生成噪声trajectory
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # trajectory = trajectory + edit_action_weight.unsqueeze(-1) * edit_action
            trajectory[torch.arange(trajectory.shape[0], device=self.device), edit_action_idx] = edit_action.squeeze()
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. 根据cond，对noise action(trajectory)进行预处理。
            model_output = model(sample=trajectory,
                                 timestep=t, 
                                 cond=global_cond,
                                 act_pos=act_position)
            # 3. compute previous image: x_t -> x_t-1
            # 执行单步去噪
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
        # finally make sure conditioning is enforced
        trajectory[torch.arange(trajectory.shape[0], device=self.device), edit_action_idx] = edit_action.squeeze()
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], pre_action=None, history_idxs=None, edit_action=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        edit_action = self.normalizer['action'].normalize(edit_action)
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2] # B：批次大小
        T = self.horizon
        Da = self.action_dim # Dim of action
        Do = self.obs_feature_dim # Dim of obs
        To = self.n_obs_steps # Number of observation steps

        # build input
        device = self.device
        dtype = self.dtype
        
        # handle different ways of passing observation 处理obs
        
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            
            if pre_action is None:
                cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
                cond_data[:,:To,...] = nobs['qpos'][:,:To,...]
            else:
                cond_data = self.normalizer['action'].normalize(pre_action)
            cond_mask = self.mask_generator(cond_data.shape, history_idxs=history_idxs, device=device)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        
        if self.edit_action_idx is None:
            batch_indices = torch.arange(B)
            history_action = cond_data[batch_indices, history_idxs, :] # B,Da
            history_action = history_action.unsqueeze(1) # B,1,Da
            last_history_action = history_action.expand(-1, self.horizon, -1) # B,T,Da
            pre_action = torch.where(cond_mask, cond_data, last_history_action)

            edit_action = edit_action.unsqueeze(1) # B,1,Da
            edit_action_position = self.position_network(
                sample=pre_action,
                obs=global_cond,
                edit_action=edit_action,
                history_idxs=history_idxs
            )
            self.edit_action_idx = edit_action_position.softmax(dim=-1).argmax(dim=-1)
            
            # 处理越界问题
            if self.edit_action_idx.item() == 0:
                self.edit_action_idx = 1
            elif self.edit_action_idx.item() == self.horizon - 1:
                self.edit_action_idx = self.horizon - 2

        # run sampling 丢给模型进行采样
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond, # local_cond = None
            global_cond=global_cond,
            act_position=history_idxs,
            edit_action_idx=self.edit_action_idx,
            edit_action=edit_action,
            **self.kwargs)
        
        # unnormalize prediction 
        # 解归一化
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        start = 0
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        # get prediction
        result = {
            'action': action,
            'action_pred': action_pred,
            'edit_action_idx': self.edit_action_idx,
        }
        return result

    # ========= training  ============
    def topp(self, trajectory_window: torch.Tensor, edit_action_noisy: torch.Tensor, N_step: float = 5):
        """
        输入:
            trajectory_window: [B, 5, 1, 7]  5 个单步起点
            edit_action_noisy: [B, 1, 7]     共同终点
            dt_des: 输出步长
        返回:
            q_out: [B, 5, N, 7]  每段 N 步，满足 vel/acc 限
        """
        device = trajectory_window.device
        B, S, _, D = trajectory_window.shape          # S = 5
        vel_lim = torch.tensor([2.0] * D, device=device)
        acc_lim = torch.tensor([5.0] * D, device=device)

        # 1. 构造起点终点 [B, 5, 7]
        q0 = trajectory_window.squeeze(2)             # [B, 5, 7]
        q1 = edit_action_noisy.expand(-1, S, -1)      # [B, 5, 7]

        # 2. 逐段极限时长（向量化）
        dq = (q1 - q0).abs()                          # [B, 5, 7]
        t_vel = dq / vel_lim
        t_acc = 2 * (dq / acc_lim).sqrt()
        T_min = torch.maximum(t_vel, t_acc).max(dim=-1, keepdim=True)[0]  # [B, 5, 1]
        T = T_min * 1.2                               # 留 20 % 余量

        # 3. 离散时间网格（每段 N 步）
        t_out = torch.linspace(0, 1, N_step, device=device)  # 归一化 [0,1]

        # 4. 七次多项式系数
        a0 = q0.unsqueeze(-2)                         # [B, 5, 1, 7]
        dq_ = q1 - q0
        a6 = 35 * dq_.unsqueeze(-2)                   # [B, 5, 1, 7]
        a7 = -35 * dq_.unsqueeze(-2)

        # 5. 计算轨迹 [B, 5, N, 7]
        t_ = t_out.view(1, 1, -1, 1)                  # [1, 1, N, 1]
        q = a0 + a6 * t_**6 + a7 * t_**7              # [B, 5, N, 7]
        return q


    def set_normalizer(self, normalizer: LinearNormalizer):
        # normalized_data = (data - mean) / std
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, edit_action_idx=None, history_idxs=None):
        """
        1. normalize input
        2. obs => condition 使用 self.obs_encoder 
        3. generate impainting mask  推理时mask后边的obs，计算loss时mask前边的obs
        4. sample noise and add to trajectory => noise_trajectory 预生成含噪声的trajectory
        5. model: noise_trajectory + cond => pred 使用 self.model降噪
        6. compute loss: pred vs target 
        7. return loss and loss_dict
        
        """

        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        global_cond = None
        trajectory = nactions
        
        # 把obs处理为condition
        batch_indices = torch.arange(batch_size)
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            # 将两个观测obs给拼接起来。 TODO:逻辑比较复杂，需要检查有没有错误。 
            this_nobs = dict_apply(nobs, lambda x: torch.stack([x[:, 0, ...],x[batch_indices, history_idxs, ...]], dim=1).reshape(-1, *x.shape[2:])) 
            nobs_features = self.obs_encoder(this_nobs) #使用obs_encoder对obs进行编码，变成点云

            if "cross_attention" in self.condition_type:
                # treat as a sequence 
                # Transformer时，把nobs_features处理为序列输入
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                # CNN时，把nobs_features处理为特征图 Batch_size x dim_obs
                global_cond = nobs_features.reshape(batch_size, -1)

        else:
            raise NotImplementedError("Not implemented obs_as_global_cond=False")

        # generate impainting mask
        # condition_mask提取当前时刻之前的trajectory
        condition_mask = self.mask_generator(trajectory.shape, history_idxs=history_idxs, device=self.device)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        batch_size = trajectory.shape[0] # 也是batch_size
        device=trajectory.device
        # Sample a random timestep for each image
        # 采样一个随机时间步timesteps，然后给trajectory加timesteps步的噪声。
        # 随机是因为实际中可能每个图像的噪声的程度不同，所以需要随机采样，以训练模型对于噪声强度的泛化能力。
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        
        pre_action = torch.zeros_like(nactions) # B,T,Da
        pre_action[condition_mask] = nactions[condition_mask]
        history_action = nactions[batch_indices, history_idxs, :] # B,Da
        history_action = history_action.unsqueeze(1) # B,1,Da
        last_history_action = history_action.expand(-1, horizon, -1) # B,T,Da
        pre_action = torch.where(condition_mask, pre_action, last_history_action)

        edit_action = nactions[batch_indices, edit_action_idx, :] # 获取edit_action_idx对应的edit_action
        edit_action = edit_action.unsqueeze(1) # B,1,Da

        edit_action_position = self.position_network(
            sample=pre_action,
            obs=global_cond,
            edit_action=edit_action,
            history_idxs=history_idxs
        )
        
        edit_loss = F.cross_entropy(edit_action_position, edit_action_idx)

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy_trajectory[torch.arange(batch_size, device=self.device), edit_action_idx.view(-1, 1), :-1] = edit_action[..., :-1]
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(sample=noisy_trajectory, timestep=timesteps, cond=global_cond, act_pos=history_idxs)
        # 预测下一时间步的动作。实际上相当于trajectory中每一项t都变成t+1。

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        mse_loss = F.mse_loss(pred, target, reduction='none')

        loss = (1 - self.edit_loss_ratio) * mse_loss + self.edit_loss_ratio * edit_loss
        # loss = edit_loss
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        loss_dict = {
                'pred': pred,
                'loss': loss.item(),
                'edit_loss': edit_loss.mean().item(),
                'mse_loss': mse_loss.mean().item(),
            }



        return loss, loss_dict