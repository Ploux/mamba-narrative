"""
Dual-track Mamba implementation
Two parallel SSM tracks with different temporal dynamics
"""

import math
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.models.mixer_seq_simple import MixerModel
from mamba_ssm.models.config_mamba import MambaConfig


class DualTrackMamba(nn.Module):
    """
    Dual-track Mamba block with fast and slow SSM tracks
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution width
        expand: Expansion factor
        dt_rank: Rank of delta projection (auto if "auto")
        fast_dt_scale: Scaling factor for fast track discretization (>1 = faster forgetting)
        slow_dt_scale: Scaling factor for slow track discretization (<1 = slower forgetting)
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        fast_dt_scale=2.0,
        slow_dt_scale=0.5,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.fast_dt_scale = fast_dt_scale
        self.slow_dt_scale = slow_dt_scale
        
        # Fast track - forgets quickly, handles immediate context
        self.fast_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            **kwargs
        )
        
        # Slow track - remembers longer, handles persistent info
        self.slow_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            **kwargs
        )
        
        # Learnable combination weights
        self.track_weights = nn.Parameter(torch.ones(2))
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, hidden_states, inference_params=None):
        """
        Args:
            hidden_states: (batch, seqlen, d_model)
        Returns:
            output: (batch, seqlen, d_model)
        """
        # Process through both tracks
        fast_out = self.fast_mamba(hidden_states, inference_params=inference_params)
        slow_out = self.slow_mamba(hidden_states, inference_params=inference_params)
        
        # Weighted combination
        weights = torch.softmax(self.track_weights, dim=0)
        combined = weights[0] * fast_out + weights[1] * slow_out
        
        # Output projection
        output = self.out_proj(combined)
        
        return output


class DualTrackBlock(nn.Module):
    """
    Dual-track block replacing standard Mamba block
    """
    def __init__(
        self,
        d_model,
        norm_epsilon=1e-5,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
        fast_dt_scale=2.0,
        slow_dt_scale=0.5,
        **mamba_kwargs
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.layer_idx = layer_idx
        
        # Norm
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)
        
        # Dual-track Mamba
        self.mixer = DualTrackMamba(
            d_model=d_model,
            fast_dt_scale=fast_dt_scale,
            slow_dt_scale=slow_dt_scale,
            layer_idx=layer_idx,
            **mamba_kwargs
        )
        
    def forward(self, hidden_states, residual=None, inference_params=None):
        """
        Pass through dual-track Mamba with residual connection
        """
        if residual is None:
            residual = hidden_states
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
        
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        
        return hidden_states, residual


class DualTrackMambaLMHeadModel(nn.Module):
    """
    Language model with dual-track Mamba backbone
    Drop-in replacement for standard MambaLMHeadModel
    """
    def __init__(
        self,
        config: MambaConfig,
        fast_dt_scale=2.0,
        slow_dt_scale=0.5,
        initializer_cfg=None,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.config = config
        
        # Embedding
        self.backbone = nn.ModuleDict({
            'embedding': nn.Embedding(config.vocab_size, config.d_model),
            'layers': nn.ModuleList([
                DualTrackBlock(
                    d_model=config.d_model,
                    layer_idx=i,
                    fast_dt_scale=fast_dt_scale,
                    slow_dt_scale=slow_dt_scale,
                    **config.ssm_cfg
                )
                for i in range(config.n_layer)
            ]),
            'norm_f': nn.LayerNorm(config.d_model)
        })
        
        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.backbone['embedding'].weight
        
        # Initialize
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, inference_params=None):
        """
        Args:
            input_ids: (batch, seqlen)
        Returns:
            logits: (batch, seqlen, vocab_size)
        """
        hidden_states = self.backbone['embedding'](input_ids)
        
        residual = None
        for layer in self.backbone['layers']:
            hidden_states, residual = layer(hidden_states, residual, inference_params)
        
        # Final norm
        if residual is None:
            residual = hidden_states
        hidden_states = self.backbone['norm_f'](residual.to(dtype=hidden_states.dtype))
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        return type('Output', (), {'logits': logits})()


def create_dual_track_model(config: MambaConfig, fast_dt_scale=2.0, slow_dt_scale=0.5):
    """
    Convenience function to create dual-track model
    """
    model = DualTrackMambaLMHeadModel(
        config=config,
        fast_dt_scale=fast_dt_scale,
        slow_dt_scale=slow_dt_scale
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Dual-track model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    return model