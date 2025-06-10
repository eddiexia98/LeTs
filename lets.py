"""
These functions are modified from ligo.py in https://github.com/VITA-Group/LiGO/tree/main.
"""
import argparse
import glob
import json
import logging
import os
import pickle
import random
import math
import re
import shutil
import sys
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange
from timm.models.layers.helpers import to_2tuple

from deit_ori import VisionTransformer, _cfg
from transmodule import normalized_uniform_init, stackbert_init, TransDepthParams, TransWidthParams, TransHeadLinear, TransLinear, TransLayeredNorm, TransFcNorm, TransPatchEmbedding



sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

def isdeit(model):
    return isinstance(model, VisionTransformer)

def num_layer_of(model):
    if isdeit(model):
        return len(model.blocks)
    else:
        raise NotImplementedError

def is_encoder_layer(name):
    return name.startswith('blocks')


K2N = lambda s: '__sm_' + '_'.join(s.split('.'))
N2K = lambda s: '.'.join(s[5:].split('_'))




@torch.no_grad()
def create_auxmodel_with_lets(model_large, args, source_model):

    model_small = source_model

    dict_model_small = model_small.state_dict()

    # save map from module to name
    dict_M2N = {}

    for name, module in model_large.named_modules():

        if not is_encoder_layer(name):
            dict_M2N[id(module)] = name
        else:
            dict_M2N[id(module)] = '.'.join(name.split('.')[1:])

    M2N = lambda m: dict_M2N[id(m)]


    for name, param in dict_model_small.items():
        if not is_encoder_layer(name):
            if name == 'cls_token' or name == 'pos_embed':
                continue
            if args.tune_small_model:
                model_large.register_parameter(K2N(name), nn.Parameter(param, requires_grad=True))
            else:
                model_large.register_buffer(K2N(name), param, persistent=True)

    if isdeit(model_small):
        enc_layers = model_small.blocks
        template_key = 'blocks'
    else:
        raise NotImplementedError


    for name, param in enc_layers[0].named_parameters():
        
        for l, _ in enumerate(enc_layers):
            k = f'{template_key}.{l}.{name}'
            w = dict_model_small[k]

            if args.tune_small_model:
                model_large.register_parameter(K2N(f'{l}.{name}'), nn.Parameter(w, requires_grad=True))
            else:
                model_large.register_buffer(K2N(f'{l}.{name}'), w, persistent=True)



    kwargs_depth_param = dict(learnable=args.tune_depth, init_scheme=args.trans_init_scheme_depth, init_noise=args.trans_init_noise_depth)
    kwargs_width_param = dict(learnable=args.tune_width, init_scheme=args.trans_init_scheme_width, init_noise=args.trans_init_noise_width)


    if isdeit(model_small) and isdeit(model_large):
        emb_small, emb_large = model_small.patch_embed, model_large.patch_embed
    else:
        raise NotImplementedError



    def create_patch_embed_layer(model_large, module_large, args, width_out_tie=None):
        return TransPatchEmbedding(model=model_large, module_name=M2N(module_large), embedding_dim=module_large.weight.shape[0],
            init_scheme_width=args.trans_init_scheme_width, init_noise_width=args.trans_init_noise_width, learn_width=args.tune_width,
            residual=args.tune_residual, residual_noise=args.tune_residual_noise, width_out_tie=width_out_tie
        )

    def create_head_layer(model_large, module_large, args, layer_index=-1, depth_tie=None, width_in_tie=None):
        return TransHeadLinear(model_large, M2N(module_large), module_large.in_features, module_large.out_features,
            layer_index=layer_index, 
            init_scheme_depth=args.trans_init_scheme_depth, init_noise_depth=args.trans_init_noise_depth, learn_depth=args.tune_depth,
            init_scheme_width=args.trans_init_scheme_width, init_noise_width=args.trans_init_noise_width, learn_width=args.tune_width,
            residual=args.tune_residual, residual_noise=args.tune_residual_noise, depth_tie=depth_tie, width_in_tie=width_in_tie
        )

    def create_lin_layer(model_large, module_index, module_large, args, layer_index=-1, depth_tie=None, width_in_tie=None, width_out_tie=None):
        return TransLinear(model_large, module_index, module_large.in_features, module_large.out_features,
            layer_index=layer_index, 
            init_scheme_depth=args.trans_init_scheme_depth, init_noise_depth=args.trans_init_noise_depth, learn_depth=args.tune_depth,
            init_scheme_width=args.trans_init_scheme_width, init_noise_width=args.trans_init_noise_width, learn_width=args.tune_width,
            residual=args.tune_residual, residual_noise=args.tune_residual_noise, depth_tie=depth_tie, width_in_tie=width_in_tie, width_out_tie=width_out_tie
        )

    def create_ln_layer(model_large, module_index, module_large, args, layer_index=-1, depth_tie=None, width_out_tie=None):
        return TransLayeredNorm(model_large, module_index, module_large.normalized_shape,  
            layer_index=layer_index, 
            eps=module_large.eps,
            init_scheme_depth=args.trans_init_scheme_depth, init_noise_depth=args.trans_init_noise_depth, learn_depth=args.tune_depth,
            init_scheme_width=args.trans_init_scheme_width, init_noise_width=args.trans_init_noise_width, learn_width=args.tune_width,
            residual=args.tune_residual, residual_noise=args.tune_residual_noise, depth_tie=depth_tie, width_out_tie=width_out_tie
        )

    def create_ln_layer_fc(model_large, module_large, args, layer_index=-1, depth_tie=None, width_out_tie=None):
        return TransFcNorm(model_large, M2N(module_large), module_large.normalized_shape,  
            layer_index=layer_index, 
            eps=module_large.eps,
            init_scheme_depth=args.trans_init_scheme_depth, init_noise_depth=args.trans_init_noise_depth, learn_depth=args.tune_depth,
            init_scheme_width=args.trans_init_scheme_width, init_noise_width=args.trans_init_noise_width, learn_width=args.tune_width,
            residual=args.tune_residual, residual_noise=args.tune_residual_noise, depth_tie=depth_tie, width_out_tie=width_out_tie
        )



    # for deit
    if args.trans_tie_param:
        setattr(emb_large, 'trans_width_emb', TransWidthParams(emb_small.proj.weight.shape[0], emb_large.proj.weight.shape[0], **kwargs_width_param))
    else:
        setattr(emb_large, 'trans_width_emb', None)


    g_e = getattr(emb_large, 'trans_width_emb')


    # for deit
    setattr(emb_large, 'proj', create_patch_embed_layer(model_large=model_large, module_large=emb_large.proj, args=args, width_out_tie=g_e))
    

    # Encoder layers
    if isdeit(model_small) and isdeit(model_large):
        small_layers, large_layers = model_small.blocks, model_large.blocks
    else:
        raise NotImplementedError
    

    def coeff_mapping_sheme_1():
        return ({ 
            0 : [0, 1], 1 : [0, 1], 2 : [0, 1], 3 : [0, 1],
            4 : [2, 3], 5 : [2, 3], 6 : [2, 3], 7 : [2, 3],
            8 : [4, 5], 9 : [4, 5], 10 : [4, 5], 11 : [4, 5],
            12 : [6, 7], 13 : [6, 7], 14 : [6, 7], 15 : [6, 7]
            },
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
            [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14]
        )


    if args.src_arch == 'deit_small_patch16_224_L8' and args.tgt_arch == 'deit_base_patch16_224_L16_H12' and args.coeff_mapping == 'e2':
        coeff_mapping_dict = coeff_mapping_sheme_1()
    else:
        raise NotImplementedError



    for i, l_large in enumerate(large_layers):
        
        if i != coeff_mapping_dict[2][i]:
            continue

        num_layers_comb = len(coeff_mapping_dict[0][i])
        
        for module_i in ['norm1.trans_depth_weight', 'attn.q.trans_depth_weight', 'attn.k.trans_depth_weight', 'attn.v.trans_depth_weight', 'attn.proj.trans_depth_weight', 'norm2.trans_depth_weight', 'mlp.fc1.trans_depth_weight', 'mlp.fc2.trans_depth_weight']:
            setattr(l_large, module_i, TransDepthParams(i, num_layers_comb, bias=True,
                learnable=args.tune_depth, init_scheme=args.trans_init_scheme_depth, init_noise=args.trans_init_noise_depth)
            )



    for i, l_large in enumerate(large_layers):
        if args.trans_tie_param:
            setattr(l_large, 'trans_width_k', TransWidthParams(small_layers[0].attn.k.weight.shape[0], l_large.attn.k.weight.shape[0], **kwargs_width_param))

            setattr(l_large, 'trans_width_q', TransWidthParams(small_layers[0].attn.q.weight.shape[0], l_large.attn.q.weight.shape[0], **kwargs_width_param))

            setattr(l_large, 'trans_width_v', TransWidthParams(small_layers[0].attn.v.weight.shape[0], l_large.attn.v.weight.shape[0], **kwargs_width_param))

            setattr(l_large, 'trans_width_ffn', TransWidthParams(small_layers[0].mlp.fc1.weight.shape[0], l_large.mlp.fc1.weight.shape[0], **kwargs_width_param))
        else:
            setattr(l_large, 'trans_width_k', None)
            setattr(l_large, 'trans_width_q', None)
            setattr(l_large, 'trans_width_v', None)
            setattr(l_large, 'trans_width_ffn', None)

        if i == len(model_small.blocks) - 1:
            break



    def layer_mapping_sheme_1():
        return {
            0 : [0, 1], 1 : [0, 1], 2 : [0, 1], 3 : [0, 1],
            4 : [2, 3], 5 : [2, 3], 6 : [2, 3], 7 : [2, 3],
            8 : [4, 5], 9 : [4, 5], 10 : [4, 5], 11 : [4, 5],
            12 : [6, 7], 13 : [6, 7], 14 : [6, 7], 15 : [6, 7]
        }


    if args.src_arch == 'deit_small_patch16_224_L8' and args.tgt_arch == 'deit_base_patch16_224_L16_H12':
        layer_mapping_dict = layer_mapping_sheme_1()
    else:
        raise NotImplementedError

    
    fla = 0

    for i, l_large in enumerate(large_layers):
        
        n1 = getattr(l_large, 'norm1.trans_depth_weight')
        n2 = getattr(l_large, 'norm2.trans_depth_weight')
        aq = getattr(l_large, 'attn.q.trans_depth_weight')
        ak = getattr(l_large, 'attn.k.trans_depth_weight')
        av = getattr(l_large, 'attn.v.trans_depth_weight')
        ap = getattr(l_large, 'attn.proj.trans_depth_weight')
        m1 = getattr(l_large, 'mlp.fc1.trans_depth_weight')
        m2 = getattr(l_large, 'mlp.fc2.trans_depth_weight')

        break


    for i, l_large in enumerate(large_layers):

        if coeff_mapping_dict[2][i] != fla:

            fla = coeff_mapping_dict[2][i]

            n1 = getattr(l_large, 'norm1.trans_depth_weight')
            n2 = getattr(l_large, 'norm2.trans_depth_weight')
            aq = getattr(l_large, 'attn.q.trans_depth_weight')
            ak = getattr(l_large, 'attn.k.trans_depth_weight')
            av = getattr(l_large, 'attn.v.trans_depth_weight')
            ap = getattr(l_large, 'attn.proj.trans_depth_weight')
            m1 = getattr(l_large, 'mlp.fc1.trans_depth_weight')
            m2 = getattr(l_large, 'mlp.fc2.trans_depth_weight')


        # corresponding to small layers
        weight_map = layer_mapping_dict[i]


        ############# MHA - norm1
        weight_map_list = []
        width_out_list = []

        for layer_i in weight_map:
            weight_map_list.append(f'{layer_i}.norm1')
            width_out_list.append(g_e)
            

        setattr(l_large, 'norm1', 
            create_ln_layer(
                model_large, weight_map_list, l_large.norm1, args, layer_index=i, 
                width_out_tie=width_out_list, depth_tie = n1
            )
        )


        ############# MHA - Attention
        
        attn_large = l_large.attn
        for name in ['q', 'k', 'v']:

            weight_map_list = []
            width_in_list = []
            width_out_list = []

            for layer_i in weight_map:
                weight_map_list.append(f'{layer_i}.attn.{name}')
                width_in_list.append(g_e)

                for layer_ii, l_large_i in enumerate(large_layers):
                    if layer_ii == layer_i:
                        width_out_list.append(getattr(l_large_i, f'trans_width_{name}'))
                        break

            if name == 'q':
                setattr(attn_large, name, 
                    create_lin_layer(
                        model_large, weight_map_list, getattr(attn_large, name), args, layer_index=i, 
                        width_in_tie=width_in_list, width_out_tie=width_out_list, depth_tie = aq)
                )
            elif name == 'k':
                setattr(attn_large, name, 
                    create_lin_layer(
                        model_large, weight_map_list, getattr(attn_large, name), args, layer_index=i, 
                        width_in_tie=width_in_list, width_out_tie=width_out_list, depth_tie = ak)
                )
            elif name == 'v':
                setattr(attn_large, name, 
                    create_lin_layer(
                        model_large, weight_map_list, getattr(attn_large, name), args, layer_index=i, 
                        width_in_tie=width_in_list, width_out_tie=width_out_list, depth_tie = av)
                )
            else:
                raise NotImplementedError



        ############# MHA - W_o

        weight_map_list = []
        width_in_list = []
        width_out_list = []

        for layer_i in weight_map:
            weight_map_list.append(f'{layer_i}.attn.proj')
            width_out_list.append(g_e)

            for layer_ii, l_large_i in enumerate(large_layers):
                if layer_ii == layer_i:
                    width_in_list.append(getattr(l_large_i, f'trans_width_v'))
                    break

        setattr(l_large.attn, 'proj', 
            create_lin_layer(
                model_large, weight_map_list, l_large.attn.proj, args, layer_index=i, 
                width_in_tie=width_in_list, width_out_tie=width_out_list, depth_tie = ap)
        )



        ############# FFN norm2
        weight_map_list = []
        width_out_list = []

        for layer_i in weight_map:
            weight_map_list.append(f'{layer_i}.norm2')
            width_out_list.append(g_e)
            

        setattr(l_large, 'norm2', 
            create_ln_layer(
                model_large, weight_map_list, l_large.norm2, args, layer_index=i, 
                width_out_tie=width_out_list, depth_tie = n2
            )
        )


        ############# FFN - Layer 1
        weight_map_list = []
        width_in_list = []
        width_out_list = []

        for layer_i in weight_map:
            weight_map_list.append(f'{layer_i}.mlp.fc1')
            width_in_list.append(g_e)

            for layer_ii, l_large_i in enumerate(large_layers):
                if layer_ii == layer_i:
                    width_out_list.append(getattr(l_large_i, f'trans_width_ffn'))
                    break

        setattr(l_large.mlp, 'fc1', 
            create_lin_layer(
                model_large, weight_map_list, l_large.mlp.fc1, args, layer_index=i, 
                width_in_tie=width_in_list, width_out_tie=width_out_list, depth_tie = m1)
        )

        
        ############# FFN - Layer 2
        weight_map_list = []
        width_in_list = []
        width_out_list = []

        for layer_i in weight_map:
            weight_map_list.append(f'{layer_i}.mlp.fc2')
            width_out_list.append(g_e)

            for layer_ii, l_large_i in enumerate(large_layers):
                if layer_ii == layer_i:
                    width_in_list.append(getattr(l_large_i, f'trans_width_ffn'))
                    break

        setattr(l_large.mlp, 'fc2', 
            create_lin_layer(
                model_large, weight_map_list, l_large.mlp.fc2, args, layer_index=i, 
                width_in_tie=width_in_list, width_out_tie=width_out_list, depth_tie = m2)
        )


    setattr(model_large, 'fc_norm', 
        create_ln_layer_fc(
            model_large, model_large.fc_norm, args, layer_index=-1, 
            width_out_tie=g_e)
    )



    ############# Classifier
    if isdeit(model_small) and isdeit(model_large):
        head_small, head_large = model_small.head, model_large.head
        if args.trans_tie_param:
            setattr(model_large, 'trans_width_cls', 
                TransWidthParams(
                    head_small.weight.shape[1], head_large.weight.shape[1], **kwargs_width_param)
            )
        else:
            setattr(model_large, 'trans_width_cls', None)
        
        setattr(model_large, 'head', 
            create_head_layer(
                model_large, head_large, args, layer_index=-1, 
                depth_tie=None, width_in_tie=getattr(model_large, 'trans_width_cls'))
        )
    
    else:
        raise NotImplementedError

    return model_large



