"""
Autoregressive Policy model for sequential token generation.
Supports chunked causal transformers with flexible token types and prediction heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Dict, Optional, Union, Any
from collections import defaultdict
from copy import deepcopy
import itertools
from lightning.pytorch import LightningModule

from source.policy.base_policy import BasePolicy
from source.model.ARP.types_and_registry import (
    TokenType, ModelConfig, IncompleteToken, 
    AttnDirectionsType, PerChunk, SampleFunctionT,
    register_token_embedding, register_token_predictor
)
from source.model.ARP.token_coder import TokenCoder
from source.model.ARP.token_embedding import ChunkEmbedding
from source.model.ARP.token_predictor import TokenPredictorInterface
from source.model.ARP.chunk_transformer import ChunkTransformerLayer
from source.model.ARP.utils import cat_uneven_blc_tensors, flatten_per_chunk_dict, map2

class AutoRegressivePolicy(BasePolicy):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Validate token IDs
        for i, tk in enumerate(cfg.tokens):
            assert tk['id'] == i, f"Token {i} has incorrect id {tk['id']}"
        
        self.token_coder = TokenCoder(cfg.tokens)
        self.chunk_embedder = ChunkEmbedding(cfg.n_embd, cfg.max_chunk_size, cfg.tokens)

        self.token_embedders = nn.ModuleList()
        self.token_name_2_ids = {}
        self.f_token_name_2_ids = lambda name: self.token_name_2_ids.get(name, name)
        for tk_id, tk in enumerate(cfg.tokens):
            assert tk['embedding'] in register_token_embedding.map, f"token embedding type: {tk['embedding']} not found!"
            self.token_name_2_ids[tk['name']] = tk_id
            self.token_embedders.append(register_token_embedding.map[tk['embedding']](cfg.n_embd, tk, **tk['embedding_kwargs']))

        self.token_predictors: List[TokenPredictorInterface] = nn.ModuleList()
        for tk in cfg.tokens:
            assert tk["predictor"] in register_token_predictor.map, f"token predictor type: {tk['predictor']} not found!"
            if tk['is_control']:
                self.token_predictors.append(nn.Identity())
            else:
                predictor_class = register_token_predictor.map[tk['predictor']]
                self.token_predictors.append(
                    predictor_class(cfg.n_embd, tk, **tk['predictor_kwargs'])
                )

        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.n_embd)

        self.blocks = nn.ModuleList()
        for layer_cfg in cfg.layers: 
            layer = ChunkTransformerLayer(
                cfg.n_embd,
                layer_cfg['n_head'],
                mlp_ratio=layer_cfg['mlp_ratio'],
                mlp_dropout=layer_cfg['mlp_dropout'],
                attn_kwargs=layer_cfg['attn_kwargs'],
                cond_attn_kwargs=layer_cfg['cond_attn_kwargs'],
                conditional=layer_cfg['condition_on'],
                AdaLN=layer_cfg.get('AdaLN', False),
                norm_before_AdaLN=layer_cfg.get('norm_before_AdaLN', False)
            )
            self.blocks.append(layer)

        self.drop = nn.Dropout(cfg.embd_pdrop)
        if cfg.layer_norm_every_block:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(cfg.n_embd) for _ in range(len(cfg.layers))])
        else:
            self.final_ln = nn.LayerNorm(cfg.n_embd)

        self.cfg = cfg
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    torch.nn.init.ones_(module.weight)
        
        self.apply(_basic_init)
        
        # Initialize AdaLN modulation layers
        for block in self.blocks:
            if block.AdaLN:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def token_codes_to_embeddings(
        self,
        tk_codes: Tensor,
        tk_ids: Tensor,
        **extra_contexts
    ) -> Tensor:
        """
        Convert token codes to embeddings.
        
        Args:
            tk_codes: Token codes of shape (B, L, dim)
            tk_ids: Token type IDs of shape (B, L)
            **extra_contexts: Additional context for embeddings
            
        Returns:
            Embeddings of shape (B, L, n_embd)
        """
        dev = tk_codes.device
        shape = tk_ids.shape
        embs = torch.zeros(list(shape) + [self.cfg.n_embd, ], dtype=torch.float, device=dev)
        for i in map(int, torch.unique(tk_ids)):
            mask = tk_ids == i
            embedder = self.token_embedders[i]
            embs[mask, :] = embedder(tk_codes[mask], **extra_contexts).to(embs.dtype)
        return embs
    
    @staticmethod
    def filter_context(
        curr_chk_id: int,
        contexts: Dict[str, Union[Tensor, Dict]]
    ) -> Dict[str, Tensor]:
        """
        Filter contexts for current chunk.
        
        Args:
            curr_chk_id: Current chunk ID
            contexts: Dict mapping keys to tensors or per-chunk dicts
            
        Returns:
            Filtered context dict for current chunk
        """
        chk_contexts = {}
        for ki, vi in contexts.items():
            if isinstance(vi, Tensor) or vi is None:
                chk_contexts[ki] = vi
            else:
                # Per-chunk context
                for kj, vj in vi.items():
                    if str(curr_chk_id) in str(kj):
                        chk_contexts[ki] = vj
                        break
                
                if ki not in chk_contexts and 'default' in vi:
                    chk_contexts[ki] = vi['default']
        
        return chk_contexts
    
    def forward(
        self,
        embs: Union[Tensor, Tuple[Tensor, Tensor]],
        chk_ids: Tensor,
        layer_ids: Optional[List[int]] = None,
        contexts: Dict[str, Tensor] = {},
        dependency_attn_mask: Optional[Tensor] = None,
        training: bool = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through transformer blocks.
        
        Args:
            embs: Embeddings (single tensor for inference, tuple for training)
            chk_ids: Chunk IDs of shape (B|1, L)
            layer_ids: List of layer indices to use (default: all)
            contexts: Additional context tensors
            dependency_attn_mask: Attention mask for token dependencies
            training: Override training mode
            
        Returns:
            Output embeddings (same format as input)
        """
        if layer_ids is None:
            layer_ids = list(range(len(self.blocks)))
        
        if training is None:
            training = self.training
        
        # Get dimensions
        if training:
            dev, (bs, L) = embs[0].device, embs[0].shape[:2]
        else:
            dev, (bs, L) = embs.device, embs.shape[:2]
        
        # Add positional embeddings
        pos_emb = self.pos_emb(torch.arange(0, L, dtype=torch.long, device=dev))[None, ...]
        if training:
            # Training mode: interleaved forward
            train_masks = ChunkTransformerLayer.train_attn_masks(chk_ids)
            embs = [self.drop(e + pos_emb) for e in embs]
            for layer_id in layer_ids:
                block: ChunkTransformerLayer = self.blocks[layer_id]
                cond = contexts[block.conditional] if block.conditional else None
                embs = block.forward_train(embs, cond, train_masks, dependency_attn_mask=dependency_attn_mask)
                if self.cfg.layer_norm_every_block:
                    embs = [self.layer_norms[layer_id](e) for e in embs] 
            if not self.cfg.layer_norm_every_block:
                embs = [self.final_ln(e) for e in embs]
        else:
            # Inference mode: standard forward
            eval_mask = ChunkTransformerLayer.eval_attn_mask(chk_ids)
            if dependency_attn_mask is not None: 
                eval_mask = eval_mask & dependency_attn_mask
            embs = self.drop(embs + pos_emb)
            for layer_id in layer_ids:
                block: ChunkTransformerLayer = self.blocks[layer_id]
                cond = contexts[block.conditional] if block.conditional else None
                embs = block.forward_inference(embs, cond, eval_mask)
                if self.cfg.layer_norm_every_block:
                    embs = self.layer_norms[layer_id](embs)        
            if not self.cfg.layer_norm_every_block:
                embs = self.final_ln(embs)
        return embs
    
    def compute_loss(
        self,
        tks: Tensor,
        chk_ids: Optional[Tensor] = None,
        valid_tk_mask: Optional[Tensor] = None,
        skip_tokens: List[int] = [],
        block_attn_directions: AttnDirectionsType = [],
        match_layer: str = "",
        contexts: Dict[str, Tensor] = {},
        log_prob: bool = False
    ) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Tensor]]:
        """
        Compute training loss.
        
        Args:
            tks: Tokens of shape (B, L, d+1), last dim is token IDs
            chk_ids: Chunk IDs of shape (B|1, L)
            valid_tk_mask: Mask for valid tokens of shape (B, L)
            skip_tokens: List of token IDs to skip in loss computation
            block_attn_directions: List of attention directions to block
            match_layer: Layer name pattern to match
            contexts: Additional context tensors
            log_prob: Whether to return log probabilities
            
        Returns:
            Loss dict, and optionally log probabilities
        """
        tk_ids, tks = tks[:, :, -1], tks[:, :, :-1]
        dev, batch_size = tks.device, len(tks)
        losses = defaultdict(list)
        
        # Default chunk IDs
        if chk_ids is None:
            chk_ids = torch.arange(0, tk_ids.size(1), device=dev)[None, ...]
        if len(chk_ids.shape) == 1:
            chk_ids = chk_ids[None, ...]
        
        assert chk_ids.size(1) == tk_ids.size(1), \
            "Chunk IDs and tokens should have the same length"
        
        # Encode tokens
        tk_codes = self.token_coder.encode(tks, tk_ids)
        
        # Create dependency attention mask
        dependency_attn_mask = None
        if block_attn_directions:
            dependency_attn_mask = ChunkTransformerLayer.dependency_attn_mask(
                tk_ids,
                map2(self.f_token_name_2_ids, block_attn_directions)
            )
        
        # Get embeddings
        embs_star = self.token_codes_to_embeddings(tk_codes, tk_ids, **contexts)
        embs_hat = self.chunk_embedder(chk_ids, tk_ids)
        
        # Filter layers
        if match_layer:
            layer_ids = [i for i, ln in enumerate(self.cfg.layers) if match_layer in ln['name']]
        else:
            layer_ids = list(range(len(self.blocks)))

        # Forward pass (interleaved)
        embs_star, embs_hat = self(
            [embs_star, embs_hat], chk_ids,
            contexts=contexts,
            layer_ids=layer_ids,
            dependency_attn_mask=dependency_attn_mask,
            training=True
        )

        cond_log_probs = torch.zeros(batch_size, tks.shape[1], device=dev) if log_prob else None
        for i in map(int, tk_ids.unique()):
            token = self.cfg.tokens[i]
            
            # Skip control tokens and specified tokens
            if token.get('is_control', False):
                continue
            if i in skip_tokens:
                continue
            
            # Create mask
            mask = tk_ids == i
            if valid_tk_mask is not None:
                mask = mask & valid_tk_mask
            
            # Select appropriate token representation
            if token['is_continuous'] and not self.token_predictors[i].IS_CONTINUOUS:
                _tks = tk_codes
            else:
                _tks = tks
            is_training = self.token_predictors[i].training
            self.token_predictors[i].train(True)
            loss_dict = self.token_predictors[i](
                embs_hat[mask], 
                label=_tks[mask][..., :token['dim']], 
                log_prob=log_prob,
                **contexts
            )
            self.token_predictors[i].train(is_training)
            
            # Extract log probabilities
            if log_prob:
                ll = sum(loss_dict.pop('log_prob'))
                cond_log_probs[mask] = ll
            
            # Accumulate losses
            for k, v in loss_dict.items():
                losses[f'{token["name"]}.{k}'] += v
        
        # Average losses
        loss_dict = {k: sum(v) / len(v) for k, v in losses.items()}
        if log_prob:
            return loss_dict, cond_log_probs
        else:
            return loss_dict
    
    @torch.no_grad()
    def generate(
        self,
        prompt_tks: Tensor,
        future_tk_chk_ids: List[IncompleteToken],
        sample: bool = False,
        contexts: Dict[str, PerChunk[Tensor]] = {},
        block_attn_directions: AttnDirectionsType = [],
        match_layer: PerChunk[str] = "",
        sample_function: PerChunk[SampleFunctionT] = {}
    ) -> Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            prompt_tks: Prompt tokens of shape (B, L, d+1)
            future_tk_chk_ids: List of incomplete tokens to generate
            sample: Whether to sample or use mode
            contexts: Additional context tensors (can be per-chunk)
            block_attn_directions: Attention directions to block
            match_layer: Layer pattern to match (can be per-chunk)
            sample_function: Custom sampling function (can be per-chunk)
            
        Returns:
            Completed sequence of shape (B, L+N, d+1)
        """
        assert not self.training, "Model should be in eval mode during generation"
        
        future_tk_chk_ids = deepcopy(future_tk_chk_ids)
        dev, batch_size = prompt_tks.device, len(prompt_tks)
        tk_ids, tks = prompt_tks[:, :, -1], prompt_tks[:, :, :-1]
        
        # Flatten per-chunk dicts
        if isinstance(sample_function, dict):
            sample_function = flatten_per_chunk_dict(sample_function)
        
        if not isinstance(match_layer, str):
            match_layer = flatten_per_chunk_dict(match_layer)
        
        # Initialize
        chk_ids = torch.arange(0, prompt_tks.size(1), device=dev)[None, ...]
        tk_codes = self.token_coder.encode(tks, tk_ids)
        
        def to_seq(codes, ids):
            """Convert codes and IDs to sequence format."""
            vals = self.token_coder.decode(codes, ids)
            return torch.cat([vals, ids[..., None]], dim=-1)
        
        # Generate tokens chunk by chunk
        # running tensors: tk_codes, chk_ids, tk_ids
        while len(future_tk_chk_ids) > 0:
            curr_chk_id = future_tk_chk_ids[0]['chk_id']
            assert curr_chk_id >= prompt_tks.size(1), "Future chunk ID should >= prompt length"
            # Collect tokens in current chunk
            curr_tokens: List[TokenType] = []
            next_chunk: List[IncompleteToken] = []
            while len(future_tk_chk_ids) > 0 and future_tk_chk_ids[0]['chk_id']== curr_chk_id:
                next_chunk.append(future_tk_chk_ids.pop(0))
                curr_tokens.append(self.cfg.tokens[next_chunk[-1]['tk_id']])
            # Prepare next chunk tensors
            next_tk_codes = torch.zeros(batch_size, len(next_chunk), tk_codes.size(-1), device=dev, dtype=tk_codes.dtype)

            next_chk_ids = torch.as_tensor([v['chk_id'] for v in next_chunk], device=dev)[None, :]
            next_tk_ids_lst = [v['tk_id'] for v in next_chunk]
            next_tk_ids = torch.as_tensor(next_tk_ids_lst, device=dev, dtype=tk_ids.dtype)[None, :].repeat(batch_size, 1)
            chk_ids = torch.cat([chk_ids, next_chk_ids], dim=1)

            # Handle control tokens
            if all([curr_token.get('is_control', False) for curr_token in curr_tokens]): 
                next_tk_codes[:, :, :1] = torch.as_tensor([v['tk_val'] for v in next_chunk], device=dev)[None, :, None]
                tk_codes = torch.cat([tk_codes, next_tk_codes], dim=1)
                tk_ids = torch.cat([tk_ids, next_tk_ids], dim=1)
                continue
            
            # Get chunk-specific contexts
            chk_contexts = self.filter_context(curr_chk_id, contexts)
            
            # Prepare embeddings
            prompt_embs = self.token_codes_to_embeddings(tk_codes, tk_ids, **chk_contexts)
            chunk_embs = self.chunk_embedder(next_chk_ids, next_tk_ids)
            embs = torch.cat([prompt_embs, chunk_embs], dim=1) 
            # Filter layers
            match_layer_ = match_layer if isinstance(match_layer, str) else match_layer.get(curr_chk_id, "")
            if match_layer_:
                layer_ids = [i for i, ln in enumerate(self.cfg.layers) if match_layer in ln['name']]
            else:
                layer_ids = list(range(len(self.blocks)))

            # Create dependency mask
            dependency_attn_mask = None
            if block_attn_directions:
                full_tk_ids = torch.cat([tk_ids, next_tk_ids], dim=1)
                dependency_attn_mask = ChunkTransformerLayer.dependency_attn_mask(
                    full_tk_ids,
                    map2(self.f_token_name_2_ids, block_attn_directions)
                )
            
            # Forward pass
            embs = self(
                embs, 
                chk_ids,
                contexts=chk_contexts,
                layer_ids=layer_ids,
                dependency_attn_mask=dependency_attn_mask,
                training=False
            )
            embs = embs[:, -len(next_chunk):, :]  # Current chunk only
            
            # Get predictions
            predicts = [None] * len(next_tk_ids_lst)
            for tk_id_key, _ in itertools.groupby(next_tk_ids_lst):
                _indices = [_i for _i, v in enumerate(next_tk_ids_lst) if v == tk_id_key]
                predict_output = self.token_predictors[tk_id_key](
                    embs[:, _indices, :],
                    split_distributions=True,
                    **chk_contexts
                )
                
                if isinstance(predict_output, Tensor):
                    predict_output = predict_output.reshape(
                        batch_size, len(_indices), *predict_output.shape[2:]
                    )
                    for _i, _ind in enumerate(_indices):
                        predicts[_ind] = predict_output[:, _i]
                else:
                    for _i, _ind in enumerate(_indices):
                        predicts[_ind] = predict_output[_i]
            
            # Sample tokens
            if callable(sample_function):
                next_tk_codes = sample_function(predicts)
            elif curr_chk_id in sample_function:
                next_tk_codes = sample_function[curr_chk_id](predicts)
            else:
                # Default sampling
                next_tk_codes, start = [], 0
                for tk_id_key, group in itertools.groupby(next_tk_ids_lst):
                    count = len(list(group))
                    output = self.token_predictors[tk_id_key].sample(
                        predicts[start:start + count],
                        do_sample=sample,
                        **chk_contexts
                    )
                    output_val_or_code = torch.cat(
                        [o[:, None, :] for o in output], dim=1
                    )
                    
                    # Encode if needed
                    token = self.cfg.tokens[tk_id_key]
                    predictor = self.token_predictors[tk_id_key]
                    if self.token_coder.need_encoding(token, predictor.IS_CONTINUOUS):
                        output_code = self.token_coder.encode_ith(
                            output_val_or_code, tk_id_key
                        )
                    else:
                        output_code = output_val_or_code
                    
                    next_tk_codes.append(output_code)
                    start += count
                
                next_tk_codes = cat_uneven_blc_tensors(*next_tk_codes)
            
            # Update running tensors
            tk_codes = cat_uneven_blc_tensors(tk_codes, next_tk_codes)
            tk_ids = torch.cat([tk_ids, next_tk_ids], dim=1)
        
        return to_seq(tk_codes, tk_ids)