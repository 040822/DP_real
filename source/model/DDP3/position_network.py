import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import numpy as np
import logging

from einops.layers.torch import Rearrange
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

from source.model.DDP3.attention import TemporalAxialAttention
from source.model.common.module_attr_mixin import ModuleAttrMixin
from source.model.DDP3.pointnet_extractor import PointNetEncoderXYZ, PointNetEncoderXYZRGB, create_mlp


class FiLM(nn.Module):
    def __init__(self,
            out_channels, 
            cond_dim):
        super().__init__()

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch (two t) -> batch two t 1',two=2),
        )
        self.reshape = Rearrange('batch dim 1 -> batch 1 dim')

    def forward(self, x, cond):
        # x shape: B,T,256
        embed = self.cond_encoder(cond)
        scale = embed[:,0,...] # B,256,1
        bias = embed[:,1,...] # B,256,1
        scale = self.reshape(scale) # B,1,256
        bias = self.reshape(bias) # B,1,256
        return scale * x + bias


class PositionBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout: float = 0.1,
        mlp_ratio=4.0
    ):
        super().__init__()
        self.t_norm1 = nn.LayerNorm(hidden_size, eps=1e-5, bias=True)
        self.t_attn = TemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads
        )
        self.t_dropout1 = nn.Dropout(dropout)
        
        self.t_norm2 = nn.LayerNorm(hidden_size, eps=1e-5, bias=True)
        self.film = FiLM(
            hidden_size,
            hidden_size
        )
        self.t_dropout2 = nn.Dropout(dropout)
        
        self.t_norm3 = nn.LayerNorm(hidden_size, eps=1e-5, bias=True)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.t_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
    
    def forward(self, x, cond_act, cond_obs):
        # x: B,T,256
        # cond_act: B,1,128
        # cond_obs: B,128
        
        # cross attn
        x_norm = self.t_norm1(x)
        x_sttn = self.t_attn(x_norm, cond_act) # B,T,256
        x = x + self.t_dropout1(x_sttn)
        
        # film
        x_norm = self.t_norm2(x)
        x_cttn = self.film(x_norm, cond_obs) # B,T,256
        x = x + self.t_dropout2(x_cttn)
        
        # mlp
        x = x + self.t_mlp(self.t_norm3(x))
        return x


class PositionNetwork(ModuleAttrMixin):
    def __init__(self,
            horizon: int,
            hidden_dim: int = 128,
            n_head: int = 4,
            action_dim: int = 26,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            n_layer: int = 4,
            debug: bool = False
        ) -> None:
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        self.input_emb = nn.Sequential(
            *create_mlp(action_dim, hidden_dim, [hidden_dim], nn.ReLU)
        )
        self.distance_emb = nn.Sequential(
            *create_mlp(1, hidden_dim, [hidden_dim], nn.ReLU)
        )
        self.register_buffer("pos_emb", self.get_temporal_pos_embed())
        self.drop = nn.Dropout(p_drop_emb)


        self.cond_act_emb = nn.Sequential(
            *create_mlp(action_dim, hidden_dim*2, [hidden_dim, hidden_dim*2], nn.ReLU)
        )
        self.cond_obs_emb = nn.Sequential(
            *create_mlp(hidden_dim*2, hidden_dim*2, [hidden_dim*2, hidden_dim*2], nn.ReLU)
        )

        self.blocks = nn.ModuleList([
            PositionBlock(
                hidden_dim*2,
                n_head,
                dropout=p_drop_attn,
            )
            for _ in range(n_layer)
        ])

        self.out_mlp = nn.Sequential(
            *create_mlp(2*hidden_dim, 1, [hidden_dim, hidden_dim], nn.ReLU)
        )
        self.softmax = nn.Softmax(dim=1)
        
        self.debug = debug

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(self.hidden_dim*2, self.horizon, scale=1.0)
        
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed
    
    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            nn.ModuleList,
            nn.Mish,
            nn.SiLU,
            nn.GELU,
            TemporalAxialAttention,
            PositionNetwork,
            nn.Sequential,
            nn.Identity,
            Rearrange
            )
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
        

    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, TemporalAxialAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Identity)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add("_dummy_variable")
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups
    
    def forward(self, sample, obs, edit_action, history_idxs) -> torch.Tensor:
        """
        Forward pass of the position network.
        """
        batch_size = sample.shape[0]
        
        # 预处理history_idx
        history_idxs_tensor = torch.tensor(history_idxs, device=sample.device, dtype=sample.dtype) # B
        if len(history_idxs_tensor.shape) == 0: # 当history_idxs是标量int时，需要从无维度变成有维度
            history_idxs_tensor = history_idxs_tensor.unsqueeze(0)
        history_idxs_tensor = history_idxs_tensor.unsqueeze(-1) # B,1
        history_idxs_tensor = history_idxs_tensor.expand(-1,self.horizon) # B,T

        # 1. input embeding
        token_embeddings = self.input_emb(sample) # B,T,129
        
        # 2. distance embeding
        arange = torch.arange(self.horizon, device=sample.device, dtype=sample.dtype) # T
        arange = arange.unsqueeze(0).repeat(batch_size, 1) # B,T

        distance = arange - history_idxs_tensor # B,T
        distance = distance.clamp(min=0)
        distance = distance.unsqueeze(-1) # B,T,1
        distance_embeddings = self.distance_emb(distance) # B,T,128
        
        x = torch.cat([token_embeddings, distance_embeddings], dim=-1) # B,T,256
        x = self.drop(x)
        # x = self.drop(x + self.pos_emb)
    
        # 3. condition embeding
        cond_act_embeddings = self.cond_act_emb(edit_action) # B,1,256

        obs = obs.reshape(batch_size,-1) # B,256  为了过FiLM层，必须reshape成两维度。
        cond_obs_embeddings = self.cond_obs_emb(obs) # B,256

        # 4. transformer
        for block in self.blocks:
            x = block(x, cond_act_embeddings, cond_obs_embeddings)
        
        # 5. output
        x = self.out_mlp(x).squeeze(-1)
        return x

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb