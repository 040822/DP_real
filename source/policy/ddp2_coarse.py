from typing import Dict
import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy

from source.model.common.normalizer import LinearNormalizer
from source.policy.base_policy import BasePolicy, BaseImagePolicy
from source.model.DDP2.mask_generator import CoarseMaskGenerator
from source.common.pytorch_util import dict_apply
from source.model.DDP2.pointnet_extractor import DP3Encoder
from source.model.DDP2.editable_diffusion import EditableDiffusion

class Coarse_DP2(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            condition_type="cross_attention_add",
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
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


        obs_encoder = DP3Encoder(observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            debug = debug
        )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color #是否使用点云颜色，不使用则只使用坐标
        self.pointnet_type = pointnet_type
        cprint(f"[DP3 init] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DP3 init] pointnet_type: {self.pointnet_type}", "yellow")
        
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
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
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
        
        self.reset()

        
    # ========= inference  ============
    def reset(self):
        pass
    
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
            # 把当前时刻之前的替换为真实trajectory，因为是真实的所以不用噪声。
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
        trajectory[condition_mask] = condition_data[condition_mask]   

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], pre_action=None, history_idxs=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if pre_action is not None:
            pre_action = self.normalizer['action'].normalize(pre_action)

        if not self.use_pc_color:
            nobs['pointcloud'] = nobs['pointcloud'][..., :3]
        
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
                cond_data[:,:To,...] = nobs['agent_pos'][:,:To,...]
            else:
                cond_data = pre_action
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

        # run sampling 丢给模型进行采样
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond, # local_cond = None
            global_cond=global_cond,
            act_position=history_idxs,
            **self.kwargs)
        
        # unnormalize prediction 
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        # normalized_data = (data - mean) / std
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, history_idxs=None):
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

        if not self.use_pc_color:
            nobs['pointcloud'] = nobs['pointcloud'][..., :3]
        
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

        bsz = trajectory.shape[0] # 也是batch_size
        device=trajectory.device
        # Sample a random timestep for each image
        # 采样一个随机时间步timesteps，然后给trajectory加timesteps步的噪声。
        # 随机是因为实际中可能每个图像的噪声的程度不同，所以需要随机采样，以训练模型对于噪声强度的泛化能力。
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=device
        ).long()
        
        #timesteps需不需要乘上self.horizon_internal？ 学长曰：不需要

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(sample=noisy_trajectory, timestep=timesteps, cond=global_cond, act_pos=history_idxs)

        # compute loss mask
        loss_mask = torch.zeros_like(condition_mask, dtype=torch.bool)
        loss_mask = ~loss_mask # 设置loss_mask全为True

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

        loss = mse_loss * loss_mask.type(mse_loss.dtype)
        
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        loss_dict = {
                'pred': pred,
                'loss': loss.item(),
                'mse_loss': mse_loss.mean().item(),
            }

        return loss, loss_dict
    
    def compute_loss_2(self, obs_dict, trajectory, history_idxs, pre_action=None):
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        trajectory = self.normalizer['action'].normalize(trajectory)
        if pre_action is not None:
            pre_action = self.normalizer['action'].normalize(pre_action)

        if not self.use_pc_color:
            nobs['pointcloud'] = nobs['pointcloud'][..., :3]
        
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
        condition_mask = self.mask_generator(trajectory.shape, history_idxs=history_idxs, device=self.device)

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
        if pre_action is not None:
            # self-forcing，把上一轮生成的动作作为输入。
            noisy_trajectory[condition_mask] = pre_action[condition_mask]
        else:
            noisy_trajectory[condition_mask] = trajectory[condition_mask]
            # pass

        pred = self.model(sample=noisy_trajectory, timestep=timesteps, cond=global_cond, act_pos=history_idxs)

        # compute loss mask
        loss_mask = torch.zeros_like(condition_mask, dtype=torch.bool)
        loss_mask = ~loss_mask # 设置loss_mask全为True

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        mse_loss = F.mse_loss(pred, target, reduction='none')

        loss = mse_loss * loss_mask.type(mse_loss.dtype)
        
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        # unnormalize prediction 
        action_pred = self.normalizer['action'].unnormalize(pred)

        # get action
        start = 0
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        loss_dict = {
                'pred': pred,
                'loss': loss.item(),
                'mse_loss': mse_loss.mean().item(),
                'action': action,
            }

        return loss, loss_dict
