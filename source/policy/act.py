from typing import Dict
from omegaconf import DictConfig
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F

from source.policy.base_policy import BasePolicy
from source.model.common.normalizer import LinearNormalizer
from source.common.pytorch_util import dict_apply


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACT(BasePolicy):
    def __init__(self, 
            optimizer_cfg: DictConfig,
            scheduler_cfg: DictConfig,
            shape_meta: dict,
            n_obs_steps: int,
            n_action_steps: int,
            
            obs_encoder,
            vae_encoder,
            vae_decoder,
            
            chunk_size: int,
            hidden_dim: int,
            latent_dim: int,
            kl_weight: float,
            temporal_agg: bool,
            temporal_agg_const: float,
            **kwargs):
        super().__init__(optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg)
        
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'point_cloud':
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        self.chunk_size = chunk_size
        self.action_dim = shape_meta['action']['shape'][0]
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.temporal_agg = temporal_agg
        self.temporal_agg_const = temporal_agg_const
        if temporal_agg:
            self.temporal_agg_init = False
            exp_weights = torch.exp(-self.temporal_agg_const * torch.arange(self.chunk_size // self.n_action_steps))
            self.exp_weights = exp_weights / exp_weights.sum()
            
        
        self.state_dim = 0
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'low_dim':
                self.state_dim = self.state_dim + shape[0]
            elif type == 'rgb':
                pass
            elif type == 'point_cloud':
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        self.obs_encoder = obs_encoder
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder

        # encoder extra parameters
        self.latent_dim = latent_dim
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(self.action_dim, hidden_dim) # project action to embedding
        self.encoder_state_proj = nn.Linear(self.state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+chunk_size, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.action_head = nn.Linear(hidden_dim, self.action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        self.normalizer = LinearNormalizer()
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.kl_weight = kl_weight
        

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        bs = batch['action'].shape[0]
        
        nobs = self.normalizer.normalize(batch['obs'])
        nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        # project action sequence to embedding dim, and concat with a CLS token
        lowdim_feature = list()
        for key in self.lowdim_keys:
            data = nobs[key]
            if bs is None:
                bs = data.shape[0]
            else:
                assert bs == data.shape[0]
            lowdim_feature.append(data)
        lowdim_feature = torch.cat(lowdim_feature, dim=-1)
        state_embed = self.encoder_state_proj(lowdim_feature)  # (bs, hidden_dim)
        state_embed = torch.unsqueeze(state_embed, axis=1)  # (bs, 1, hidden_dim)
        action_embed = self.encoder_action_proj(nactions) # (bs, seq, hidden_dim)
        
        cls_embed = self.cls_embed.weight # (1, hidden_dim)
        cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
        encoder_input = torch.cat([cls_embed, state_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
        encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
        # do not mask cls token
        cls_joint_is_pad = torch.full((bs, 2), False).to(state_embed.device) # False: not a padding
        is_pad = torch.full((bs, nactions.shape[-2]), False).to(state_embed.device)
        src_key_padding_mask = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
        # obtain position embedding
        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
        # query model
        encoder_output = self.vae_encoder(encoder_input, pos=pos_embed, src_key_padding_mask=src_key_padding_mask)
        encoder_output = encoder_output[0] # take cls output only
        latent_info = self.latent_proj(encoder_output)
        mu = latent_info[:, :self.latent_dim]
        logvar = latent_info[:, self.latent_dim:]
        latent_sample = reparametrize(mu, logvar)
        latent_input = self.latent_out_proj(latent_sample)


        # Image observation features and position embeddings
        src, pos, proprio_input = self.obs_encoder(nobs)
        hs = self.vae_decoder(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(nactions, a_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        kl = total_kld[0]
        loss = l1 + kl * self.kl_weight
          
        loss_dict = {
            'bc_loss': loss.item(),
        }
        return loss, loss_dict

    # ========= inference  ============
    def reset(self):
        if self.temporal_agg:
            self.temporal_agg_init = False
            exp_weights = torch.exp(-self.temporal_agg_const * torch.arange(self.chunk_size // self.n_action_steps))
            self.exp_weights = exp_weights / exp_weights.sum()
        
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        example obs_dict:
        {
            obs={
                robot_qpos=[]
                robot_ee_pose-[]
            },
            action=[]
        }
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        value = next(iter(nobs.values()))
        bs = value.shape[0]
        
        mu = logvar = None
        latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(value.device)
        latent_input = self.latent_out_proj(latent_sample)

        # Image observation features and position embeddings
        src, pos, proprio_input = self.obs_encoder(nobs)
        hs = self.vae_decoder(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        
        action_pred = self.normalizer['action'].unnormalize(a_hat)
        
        if self.temporal_agg:
            if not self.temporal_agg_init:
                self.action_chunk_buffer = action_pred.unsqueeze(1).repeat(1, self.chunk_size // self.n_action_steps, 1, 1)   # [bs, self.chunk_size//self.n_action_steps, self.chunk_size, self.action_dim]
                self.temporal_agg_init = True
            else:
                new_action_chunk_buffer = torch.zeros([bs, self.chunk_size // self.n_action_steps, self.chunk_size, self.action_dim]).cuda()
                new_action_chunk_buffer[:, self.n_action_steps:, :-self.n_action_steps, :] = self.action_chunk_buffer[:, :-self.n_action_steps, self.n_action_steps:, :]
                new_action_chunk_buffer[:, 0, ...] = action_pred
                self.action_chunk_buffer = new_action_chunk_buffer
            action_pred = torch.einsum('btad,t->bad', self.action_chunk_buffer, self.exp_weights.to(self.action_chunk_buffer.device))
        
        # get action
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        
        return result
    
        # ========= trainer  ============
    def training_step(self, batch, batch_idx):
        loss, _ = self.compute_loss(batch)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _  = self.compute_loss(batch)
        self.log('val/loss', loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss