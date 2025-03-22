import torch
import torch.nn as nn

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.attention import Attention
from typing import Any, Dict, Optional, Tuple, Union

from .utils import AdapterEraser



def setup_cogvideo_adapter_eraser(model, eraser_rank, device, dtype):
    def replace_transformer_block(model):
        for name, module in model.named_modules():
            if isinstance(module, CogVideoXBlock):
                print("changing: ",name)
                original_attention = module.attn1
                modified_attention = CogVideoXWithEraser(original_attention, eraser_rank).to(device = device, dtype = dtype)
                module.attn1 = modified_attention

    replace_transformer_block(model)
    erasers = {}
    for name, module in model.named_modules():
        if isinstance(module, CogVideoXWithEraser):
            eraser_name = f'{name}.adapter'
            print(eraser_name)
            erasers[eraser_name] = module.adapter
    return erasers


def inject_eraser(transformer, eraser_ckpt, eraser_rank, device, dtype, eraser_type='adapter'):
    for name, module in transformer.named_modules():
        if isinstance(module, CogVideoXBlock):
            print("changing: ",name)
            original_attention = module.attn1
            modified_attention = CogVideoXWithEraser(original_attention, eraser_rank)
            module.attn1 = modified_attention
            eraser_name = f'{name}.attn1.{eraser_type}'
            module.attn1.adapter.load_state_dict(eraser_ckpt[eraser_name])
            module.attn1.adapter.to(device = device, dtype = dtype)
            #setattr(module, name, block_w_adapter)
        

class CogVideoXWithEraser(nn.Module):
    def __init__(
        self,
        attn,
        eraser_rank
    ):
        super().__init__()
        self.attn = attn
        self.adapter = AdapterEraser(attn.to_v.weight.shape[-1], eraser_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        
        hidden_states, encoder_hidden_states = self.attn(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            **cross_attention_kwargs,
        )

        if self.adapter.use_eraser:
            hidden_states = hidden_states + self.adapter(hidden_states)

        return hidden_states, encoder_hidden_states
    
