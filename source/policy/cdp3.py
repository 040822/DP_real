from typing import Dict
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from einops import reduce

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
from lightning.pytorch import LightningModule

from source.common.pytorch_util import dict_apply
from source.model.common.normalizer import LinearNormalizer
from source.model.common.lr_scheduler import get_scheduler
from source.model.CDP3.transformer_for_diffusion_causal import TransformerForDiffusion
from source.model.CDP3.mask_generator import LowdimMaskGenerator
from source.model.CDP3.pointnet_extractor import DP3Encoder

class CDP3(LightningModule):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            optimizer_cfg: DictConfig,
            scheduler_cfg: DictConfig,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # arch
            n_layer=8,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            obs_as_cond=True,
            use_point_crop=False,
            condition_type="cross_attention",
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            temporally_constant_weight=0.0,
            temporally_increasing_weight=0.0,
            temporally_random_weights=0.0,
            chunk_wise_weight=1.0,
            buffer_init="zero",
            # parameters passed to step
            with_causal=False,
            causal_condition_noise_weight=15.0,
            training_mode_thres=0.2,
            **kwargs):
        super().__init__()

        self.condition_type = condition_type
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

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
                        )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = obs_feature_dim + action_dim
        cond_dim = 0
        if obs_as_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                cond_dim = obs_feature_dim
            else:
                cond_dim = obs_feature_dim * n_obs_steps
        self.cond_dim = cond_dim
        output_dim = input_dim


        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[TransformerBasedDP3] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[TransformerBasedDP3] pointnet_type: {self.pointnet_type}", "yellow")


        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            with_causal=with_causal
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=with_causal
        )
        
        self.normalizer = LinearNormalizer()
        # if (horizon < n_obs_steps - 1 + n_action_steps) or (horizon % 4 != 0):
        #     raise ValueError(
        #         "Horizon must be longer than (To-1) + Ta \n Also, the horizon must be divisible by 4 for the UNet to accept it."
        #         % (horizon - n_obs_steps, n_action_steps)
        #     )
            
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # TEDi action buffer
        self.action_buffer = None

        # KV cache
        self.k_cache, self.v_cache = None, None
        self.obs_k_cache, self.obs_v_cache = None, None
        self.cache_start_idx = 0
        
        # causal
        self.with_causal = with_causal
        self.causal_condition_noise_weight = causal_condition_noise_weight
        self.training_mode_thres = training_mode_thres

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    # ========= common  ============
    def reset(self):
        self.reset_cache()
    
    def reset_cache(self):
        self.action_buffer = None

        self.k_cache, self.v_cache = None, None
        self.obs_k_cache, self.obs_v_cache = None, None
        self.cache_start_idx = 0
    
    def push_buffer(self, new_value):
        self.action_buffer = torch.cat([self.action_buffer, new_value], dim=1)

    def pop_cache(self):
        assert self.k_cache is not None
        assert self.v_cache is not None
        assert self.obs_k_cache is not None
        assert self.obs_v_cache is not None
        
        self.k_cache = self.k_cache[:, self.n_action_steps:]
        self.v_cache = self.v_cache[:, self.n_action_steps:]
        self.obs_k_cache = self.obs_k_cache[:, self.n_action_steps:]
        self.obs_v_cache = self.obs_v_cache[:, self.n_action_steps:]

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        """
        Sample from the diffusion model conditioned on condition_data. Unlike EDM, we just do one
        condition_data: (B, T, C) The conditioning data
        Returns:
            action_pred: (B, T, Da) The predicted action including the observation steps
        """
        Tp = condition_data.shape[1]
        Ta = self.n_action_steps
        To = self.n_obs_steps

        model = self.model
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        
        if self.action_buffer is None:
            self.action_buffer = torch.randn(size=condition_data.shape, dtype=condition_data.dtype,device=condition_data.device)
            if self.with_causal:
                self.action_buffer[:, :To, :] = 0.0

        condition_data[condition_mask] = self.action_buffer[condition_mask]

        # We need to denoise the first T_o+T_a steps, i.e. push their sigma to 0
        for t in scheduler.timesteps:
            # 1. apply conditioning
            self.action_buffer[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            if self.with_causal:
                if t == self.num_inference_steps - 1:
                    if self.cache_start_idx == 0:
                        model_output, self.k_cache, self.v_cache, self.obs_k_cache, self.obs_v_cache = model.forward_with_cache(
                            self.action_buffer,
                            t,
                            tpe_start=self.cache_start_idx,
                            cond=cond,
                            diff_step_idx=self.num_inference_steps-t-1,
                            k_cache=None,
                            v_cache=None,
                            cond_k_cache=None,
                            cond_v_cache=None
                        )
                    else:
                        model_output, self.k_cache, self.v_cache, self.obs_k_cache, self.obs_v_cache = model.forward_with_cache(
                            self.action_buffer[:, To - Ta:],
                            t,
                            tpe_start=self.cache_start_idx + To - Ta,
                            cond=cond,
                            diff_step_idx=self.num_inference_steps-t-1,
                            k_cache=self.k_cache,
                            v_cache=self.v_cache,
                            cond_k_cache=self.obs_k_cache,
                            cond_v_cache=self.obs_v_cache
                        )
                else:
                    model_output = model.forward_with_cache(
                        self.action_buffer[:, To:],
                        t,
                        tpe_start=self.cache_start_idx + To,
                        cond=cond[:, To:],
                        diff_step_idx=self.num_inference_steps-t-1,
                        k_cache=self.k_cache,
                        v_cache=self.v_cache,
                        cond_k_cache=self.obs_k_cache,
                        cond_v_cache=self.obs_v_cache
                    )
            else:
                model_output = model.forward_without_cache(
                    self.action_buffer,
                    t,
                    tpe_start=self.cache_start_idx,
                    cond=cond
                )

           # 3. compute previous image: x_t -> x_t-1
            if self.with_causal:
                self.action_buffer[:, To:] = scheduler.step(
                    model_output,
                    t,
                    self.action_buffer[:, To:],
                    generator=generator,
                    **kwargs,
                ).prev_sample
            else:
                self.action_buffer = scheduler.step(
                    model_output,
                    t,
                    self.action_buffer,
                    generator=generator,
                    **kwargs,
                ).prev_sample
        
        # Finally, make sure conditioning is enforced
        self.action_buffer[condition_mask] = condition_data[condition_mask]

        # Return whole buffer as prediction, we slice later
        action_pred = self.action_buffer  # (B, T, Da) or (B, T, Da+Do)

        if self.with_causal:
            # noise = torch.randn(size=condition_data.shape, dtype=condition_data.dtype,device=condition_data.device)
            # self.action_buffer[:, To:To+Ta] += noise[:, To:To+Ta] / self.causal_condition_noise_weight

            self.action_buffer = self.action_buffer[:, Ta:To+Ta]

            B = condition_data.shape[0]
            new_noise = torch.randn(
                size=(B, Tp - To, self.action_buffer.shape[-1]),
                dtype=self.dtype,
                device=self.device,
            )

            self.push_buffer(new_noise)

            self.pop_cache()
        else:
            self.action_buffer = None

        self.cache_start_idx += self.n_action_steps

        return action_pred

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        # if not self.use_pc_color:
        #     nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        # this_n_point_cloud = nobs['point_cloud']


        if 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        Ta = self.n_action_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            # condition through global feature
            if self.with_causal:
                if self.cache_start_idx == 0:
                    this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                else:
                    this_nobs = dict_apply(nobs, lambda x: x[:,To-Ta:To,...].reshape(-1,*x.shape[2:]))
            else:
                this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))

            nobs_features = self.obs_encoder(this_nobs)
            shape = (B, T, Da)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                cond = nobs_features.reshape(B, -1, self.cond_dim)
            else:
                # reshape back to B, Do
                cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            if self.with_causal:
                cond_mask[:,:To] = True
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            if self.with_causal:
                cond_mask[:,:To] = True
            else:
                cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        # if not self.use_pc_color:
        #     nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        if 'point_cloud' in nobs:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            # this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            # this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()
        
        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape, device=trajectory.device)

        B = trajectory.shape[0]
        T = trajectory.shape[1]
        To = self.n_obs_steps
        Ta = self.n_action_steps

        # Get noise on shape (B, T, D)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        start_idx = batch['sample_start_idx']
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        if self.with_causal:
            mask_temp = torch.arange(self.horizon, device=start_idx.device).unsqueeze(0).expand(B, -1)
            mask = mask_temp < start_idx.unsqueeze(1)  # 形状为 (batch_size, To // self.n_action_steps)
            mask = mask.unsqueeze(-1).expand(-1, -1, self.action_dim)  # 扩展到 (batch_size, To // self.n_action_steps, action_dim)
            # 使用掩码操作
            noisy_trajectory[mask] = 0.0
            noisy_trajectory[condition_mask] += noise[condition_mask] / self.causal_condition_noise_weight
        
        cache_start_idx = torch.where(start_idx == 0, batch['buffer_start_idx'], To-start_idx)

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps,
            tpe_start=cache_start_idx,
            cond=cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        loss_dict = {
            'bc_loss': loss.item(),
        }
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val/loss', loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        # obs_dict = batch['obs']
        # gt_action = batch['action']
        
        # result = self.predict_action(obs_dict)
        # pred_action = result['action_pred']
        # mse = F.mse_loss(pred_action, gt_action)
        # self.log('val/pred_action_mse', mse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optim_groups = self.model.get_optim_groups(
            weight_decay=self.optimizer_cfg.transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
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