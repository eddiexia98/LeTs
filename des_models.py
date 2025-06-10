import torch
import torch.nn as nn
from functools import partial

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from deit_ori import VisionTransformer, _cfg


__all__ = [
    # L8-H6
    'deit_small_patch16_224_L8', 
    
    # L12-H12
    'deit_base_patch16_224_L12',
    
    # L16-H12
    'deit_base_patch16_224_L16_H12'
]


@register_model
def deit_small_patch16_224_L8(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model



@register_model
def deit_base_patch16_224_L12(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    return model



@register_model
def deit_base_patch16_224_L16_H12(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=16, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    return model

