from typing import Dict
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy

from source.model.common.normalizer import LinearNormalizer
from source.policy.base_policy import BasePolicy, BaseImagePolicy
from source.model.DDP2.mask_generator import LowdimMaskGenerator
from source.common.pytorch_util import dict_apply
from source.model.DDP2.multi_image_obs_encoder import MultiImageObsEncoder
from source.model.DDP2.editable_diffusion import EditableDiffusion

class Fine_DP2(BaseImagePolicy):
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
        obs_feature_dim = int(obs_feature_dim[0])
        
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
        self.model = model
        self.noise_scheduler = noise_scheduler
        # DDPM scheduler 相当于denoise(采样器）
        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
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

        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None, act_position=None,
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
            # 1. apply conditioning 
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. 根据cond，对noise action(trajectory)进行预处理。
            # model_output = model(sample=trajectory,
            #                     timestep=t, 
            #                     local_cond=local_cond, global_cond=global_cond)
            model_output = model(sample=trajectory,
                                 timestep=t, 
                                 cond=global_cond,
                                 act_pos=act_position)
            # 3. compute previous image: x_t -> x_t-1
            # 执行单步去噪
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], pre_action=None, act_position=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        
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
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                cond_data = self.normalizer['action'].normalize(pre_action)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                cond_mask[:, 0, :] = True
                cond_mask[:, -1, :] = True
        else:
            raise NotImplementedError("Not implemented obs_as_global_cond=False")

        # run sampling 丢给模型进行采样
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond, # local_cond = None
            global_cond=global_cond,
            act_position=act_position, # act_position = None
            **self.kwargs)
        
        # unnormalize prediction 
        # 解归一化
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1 # 0
        end = start + self.n_action_steps + 1 # 0 + 8 + 1 = 9
        action = action_pred[:,start:end] # 0~9
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        # normalized_data = (data - mean) / std
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, obs_dict, trajectory, pre_action=None, act_position=None):
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        trajectory = self.normalizer['action'].normalize(trajectory)
        if pre_action is not None:
            pre_action = self.normalizer['action'].normalize(pre_action)
        
        batch_size = trajectory.shape[0]
        horizon = trajectory.shape[1]
        To = self.n_obs_steps

        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs) #使用obs_encoder对obs进行编码，变成点云
            if "cross_attention" in self.condition_type:
                # Transformer时，把nobs_features处理为序列输入
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # CNN时，把nobs_features处理为特征图 Batch_size x dim_obs
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            raise NotImplementedError("Not implemented obs_as_global_cond=False")
        
        # condition_mask提取当前时刻之前的trajectory
        # NOTE: condition_mask全为False，这一行保留是为了和原结构兼容；后续 loss_mask 直接显式构造。
        condition_mask = self.mask_generator(trajectory.shape, device=self.device)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0] # 也是batch_size
        device=trajectory.device

        # 采样一个随机时间步timesteps，然后给trajectory加timesteps步的噪声。
        # 随机是因为实际中可能每个图像的噪声的程度不同，所以需要随机采样，以训练模型对于噪声强度的泛化能力。
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=device
        ).long()


        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # 设置cond，注意：condition_mask 此处恒全 False，这一行实际无副作用，仅保留为占位。
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        if pre_action is not None:
            noisy_trajectory[:, 0, :] = pre_action[:, 0, :]
            noisy_trajectory[:, -1, :] = pre_action[:, -1, :]

        pred = self.model(sample=noisy_trajectory, timestep=timesteps, cond=global_cond, act_pos=act_position)

        # ===== compute loss =====
        loss_mask = ~condition_mask

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        mse_loss = F.mse_loss(pred, target, reduction='none')

        loss_mask = loss_mask.to(mse_loss.dtype)
        mask_num = loss_mask.sum().clamp_min(1.0)
        loss = (mse_loss * loss_mask).sum() / mask_num
        loss_dict = {
                'pred': pred,
                'loss': loss.item(),
                'mse_loss': mse_loss.mean().item(),
            }
        return loss, loss_dict


