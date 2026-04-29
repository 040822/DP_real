
from typing import Union, Optional, Tuple
import logging
import torch
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange

from source.model.DDP3.attention import TemporalAxialAttention
from source.model.common.positional_embedding import SinusoidalPosEmb
from source.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

def modulate(x, shift, scale):
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // scale.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def gate(x, g):
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x

class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        n_obs_steps,
        n_action_steps,
        dropout: float = 0.1,
        mlp_ratio=4.0
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        self.t_norm1 = nn.LayerNorm(hidden_size, eps=1e-5, bias=True)
        self.t_attn1 = TemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads
        )
        self.t_dropout1 = nn.Dropout(dropout)
        
        self.t_norm2 = nn.LayerNorm(hidden_size, eps=1e-5, bias=True)
        self.t_attn2 = TemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads
        )
        self.t_dropout2 = nn.Dropout(dropout)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.t_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size)
            )
        self.t_norm3 = nn.LayerNorm(hidden_size, eps=1e-5, bias=True)
    
    def forward(self, x, cond):
        # self attn
        x_norm = self.t_norm1(x)
        x_sttn = self.t_attn1(x_norm, x_norm)
        x = x + self.t_dropout1(x_sttn)
        
        # cross attn
        x_norm = self.t_norm2(x)
        x_cttn = self.t_attn2(x_norm, cond)
        x = x + self.t_dropout2(x_cttn)
        
        # mlp
        x = x + self.t_mlp(self.t_norm3(x))
        return x

class EditableDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int,
            n_action_steps: int,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            debug: bool = False,
        ) -> None:
        super().__init__()
        
        self.n_emb = n_emb
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        
        # input embedding
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.drop = nn.Dropout(p_drop_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.horizon, n_emb))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, self.n_obs_steps+2, n_emb))

        # condition embedding
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        # self.time_emb = nn.Sequential(
        #     Rearrange('b t -> (b t)'),  # reshape to (B*T,)
        #     SinusoidalPosEmb(n_emb),
        #     Rearrange('(b t) d -> b (t d)', d=n_emb, t = horizon-n_obs_steps),    # reshape to (B, T*dsed)
        #     nn.Linear((horizon-n_obs_steps) * n_emb, n_emb * 4),
        #     nn.Mish(),
        #     nn.Linear(n_emb * 4, n_emb),
        # )
        self.time_emb = SinusoidalPosEmb(n_emb) # 目前使用的是dp3的timestep，每个batch的timestep统一。
        self.act_pos_emb = SinusoidalPosEmb(n_emb)

        # causal mask
        self.editable_mask = None # 在forward中创建
        
        # attn
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(
                n_emb,
                n_head,
                n_obs_steps,
                n_action_steps,
                dropout=p_drop_attn,
            )
            for _ in range(n_layer)
        ])
        self.n_layer = n_layer
        
        # output
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        self.debug = debug
        self.batch_size = 0
    
    def create_editable_mask(self):
        mask = torch.ones(self.batch_size, self.horizon, self.n_emb)

        # 待编辑
        return mask
            
    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb,
            TemporalTransformerBlock,
            nn.ModuleList,
            nn.Mish,
            nn.SiLU,
            nn.GELU,
            TemporalAxialAttention,
            EditableDiffusion,
            nn.Sequential,
            nn.Identity,
            Rearrange)
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
        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")
        
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


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer


    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor,
        act_pos: None,
        **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        self.batch_size = sample.shape[0] # 获取batch_size。
        self.editable_mask = self.create_editable_mask() # 创建editable_mask
        
        # 确保timestep维度正确
        if not torch.is_tensor(timestep): # 推理的时候，传入的timestep可能是float或者int
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0: # 推理的时候，传入的timestep可能是标量tensor
            timestep = timestep.unsqueeze(0).to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        

        # 1. input embeding
        token_embeddings = self.input_emb(sample)
        tesz = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :tesz, :]
        x = self.drop(token_embeddings + position_embeddings)
    
        # 2. condition embeding
        cond_embeddings = self.cond_obs_emb(cond)
        #timestep = timestep.unsqueeze(1) # [batch] => [batch, 1]
        time_emb = self.time_emb(timestep).unsqueeze(1)
        cond_embeddings = torch.cat([cond_embeddings, time_emb], dim=1)

        # 确保act_pos维度正确
        if act_pos is not None:
            if not torch.is_tensor(act_pos): # 推理的时候，传入的timestep可能是float或者int
                act_pos = torch.tensor(act_pos, dtype=torch.long, device=sample.device)
            if torch.is_tensor(act_pos) and len(act_pos.shape) == 0: # 推理的时候，传入的timestep可能是标量tensor
                act_pos = act_pos.unsqueeze(0).to(sample.device)
            act_pos = act_pos.expand(sample.shape[0])
            act_pos_emb = self.act_pos_emb(act_pos).unsqueeze(1)
            cond_embeddings = torch.cat([cond_embeddings, act_pos_emb], dim=1)

        cesz = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :cesz, :]
        cond_embeddings = self.drop(cond_embeddings + position_embeddings)

        # 3. transformer
        self.editable_mask = self.editable_mask.to(x.device)
        if self.debug:
            print(f"[EditableDiffusion forward] x shape: {x.shape}, cond_embeddings shape: {cond_embeddings.shape}")
            print(f"[EditableDiffusion forward] editable_mask shape: {self.editable_mask.shape}")
        for block in self.blocks:
            x = self.editable_mask * block(x, cond_embeddings) + (1 - self.editable_mask) * x
        
        # 4. output
        x = self.head(self.ln_f(x))
        return x
