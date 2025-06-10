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



K2N = lambda s: '__sm_' + '_'.join(s.split('.'))
N2K = lambda s: '.'.join(s[5:].split('_'))



def normalized_uniform_init(w, init_scheme):
    init_weight = torch.rand_like(w)
    # nn.init.uniform_(init_weight, 0.0, 1.0)
    if 'softmax' in init_scheme:
        init_weight = F.softmax(init_weight, -1) # softmax normalize
    else:
        init_weight = init_weight / torch.sum(init_weight, -1, keepdim=True) # normalize
    w.copy_(init_weight)


def stackbert_init(w, layer_index, init_scheme, init_noise=0.03):
    init_weight = torch.zeros_like(w)
    if 'noisy' in init_scheme:
        init_weight.uniform_(0.0, init_noise)
    init_weight[layer_index % len(init_weight)] = 1.

    init_weight = init_weight / torch.sum(init_weight) # normalize
    w.copy_(init_weight)




class TransDepthParams(nn.Module):

    def __init__(self, layer_index, num_layers, bias=False, learnable=True, init_scheme='rand', init_noise=0.03):

        super(TransDepthParams, self).__init__()

        assert init_scheme in ['rand', 'rand_softmax', 'stackbert', 'interlace', 'stackbert_noisy', 'interlace_noisy']

        self.layer_index = layer_index
        self.init_scheme = init_scheme
        self.init_noise = init_noise

        if learnable:
            self.trans_weight = nn.Parameter(torch.zeros(num_layers))
            if bias:
                self.trans_bias = nn.Parameter(torch.zeros(num_layers))
            else:
                self.trans_bias = None
        else:
            self.register_buffer('trans_weight', torch.zeros(num_layers), persistent=True)
            if bias:
                self.register_buffer('trans_bias', torch.zeros(num_layers), persistent=True)
            else:
                self.trans_bias = None

        self.reset_parameters()


    def reset_parameters(self):
        # init depth
        if self.init_scheme in ['rand', 'rand_softmax']:
            normalized_uniform_init(self.trans_weight, self.init_scheme)
            if self.trans_bias is not None:
                normalized_uniform_init(self.trans_bias, self.init_scheme)
        
        elif self.init_scheme in ['stackbert', 'stackbert_noisy']:
            stackbert_init(self.trans_weight, self.layer_index, self.init_scheme, self.init_noise)
            if self.trans_bias is not None:
                stackbert_init(self.trans_bias, self.layer_index, self.init_scheme, self.init_noise)
        



class TransWidthParams(nn.Module):

    def __init__(self, small_dim, large_dim, learnable=False, init_scheme='rand', init_noise=0.03):

        super(TransWidthParams, self).__init__()

        assert init_scheme in ['rand', 'rand_softmax', 'sel', 'sel_noisy']

        self.init_scheme = init_scheme
        self.init_noise = init_noise

        if large_dim - small_dim > 0:
            if learnable:
                self.trans_weight = nn.Parameter(torch.zeros(large_dim - small_dim, small_dim))
            else:
                self.register_buffer('trans_weight', torch.zeros(large_dim - small_dim, small_dim))
        else:
            self.trans_weight = None

        self.reset_parameters()


    def reset_parameters(self):
        if self.trans_weight is not None:
            if self.init_scheme in ['rand', 'rand_softmax']:
                normalized_uniform_init(self.trans_weight, self.init_scheme)
            
            elif self.init_scheme in ['sel', 'sel_noisy']:
                sel = torch.randint(0, self.trans_weight.shape[1], (self.trans_weight.shape[0],))
                init_weight = torch.zeros_like(self.trans_weight, dtype=torch.float32)
                
                if 'noisy' in self.init_scheme:
                    init_weight.uniform_(0.0, self.init_noise)
                init_weight[torch.arange(self.trans_weight.shape[0]), sel] = 1.
                
                self.trans_weight.copy_(init_weight)




class TransHeadLinear(nn.Module):

    def __init__(self, model, module_name, in_features, out_features, layer_index=-1, 
    init_scheme_depth='rand', init_noise_depth=0.03, learn_depth=True,
    init_scheme_width='rand', init_noise_width=0.03, learn_width=True, 
    residual=False, depth_tie=None, width_in_tie=None, residual_noise=0.01, rank_w=8):

        super(TransHeadLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # weights for attention layers if depth expansion, small model
        self.get_weights = lambda: getattr(model, K2N(module_name) + '_weight')
        self.get_bias = lambda: getattr(model, K2N(module_name) + '_bias', None)

        self.bias = (self.get_bias() is not None)

        self.residual = residual
        self.residual_noise = residual_noise

        if residual:
            self.residual_weight = nn.Parameter(torch.empty((self.out_features, self.in_features)))
            if self.bias:
                self.residual_bias = nn.Parameter(torch.empty(self.out_features))
            else:
                self.register_parameter('residual_bias', None)

        if width_in_tie is None:
            hidden_dim_small = self.get_weights().shape
            self.trans_width_in = TransWidthParams(hidden_dim_small[1], self.in_features,
                learnable=learn_width, init_scheme=init_scheme_width, init_noise=init_noise_width, rank=rank_w)
            width_in_tie = self.trans_width_in

        self.get_width_weight = lambda: width_in_tie.trans_weight

        self.reset_parameters()



    def reset_parameters(self):

        if self.residual:
            nn.init.uniform_(self.residual_weight, -self.residual_noise, self.residual_noise)
            if self.bias:
                nn.init.uniform_(self.residual_bias, -self.residual_noise, self.residual_noise)

        if hasattr(self, 'trans_depth_weight'):
            self.trans_depth_weight.reset_parameters()
        if hasattr(self, 'trans_width_in'):
            self.trans_width_in.reset_parameters()


    def get_params(self):

        bias = None

        weight = self.get_weights()
        if self.bias:
            bias = self.get_bias()

        in_dim_expand = self.get_width_weight()

        if in_dim_expand is not None:
            in_dim_expand = torch.transpose(in_dim_expand, 0, 1)
            weight = torch.cat([weight, torch.matmul(weight, in_dim_expand)], 1) # expand in dimension
        

        if self.residual:
            weight = weight + self.residual_weight
            if self.bias:
                bias = bias + self.residual_bias

        return weight, bias


    def forward(self, input):
        weight, bias = self.get_params()
        return F.linear(input, weight, bias)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )




class TransLinear(nn.Module):

    def __init__(self, model, module_index, in_features, out_features, layer_index=-1, 
    init_scheme_depth='rand', init_noise_depth=0.03, learn_depth=True,
    init_scheme_width='rand', init_noise_width=0.03, learn_width=True, 
    residual=False, depth_tie=None, width_in_tie=None, width_out_tie=None, residual_noise=0.01, rank_w=8):

        super(TransLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weights = []
        self.bias = []
        self.module_index = module_index

        for mi in self.module_index:
            self.weights.append(getattr(model, K2N(mi) + '_weight'))   
            self.bias.append(getattr(model, K2N(mi) + '_bias', None))


        self.residual = residual
        self.residual_noise = residual_noise

        if residual:
            self.residual_weight = [nn.Parameter(torch.empty((self.out_features, self.in_features))) for i in self.module_index]
            if self.bias:
                self.residual_bias = [nn.Parameter(torch.empty(self.out_features)) for i in self.module_index]
            else:
                self.register_parameter('residual_bias', None)


        # for embedding or classifier layer, specify layer_index to -1
        if layer_index >= 0:
            if depth_tie is None:
                # num_layers_small = self.get_weights().shape[-1]
                num_layers_comb = len(self.module_index)

                self.trans_depth_weight = TransDepthParams(layer_index, num_layers_comb, bias=self.bias,
                    learnable=learn_depth, init_scheme=init_scheme_depth, init_noise=init_noise_depth)
                depth_tie = self.trans_depth_weight

            self.get_depth_weight = lambda: (depth_tie.trans_weight, getattr(depth_tie, 'trans_bias', depth_tie.trans_weight))


        if width_in_tie is None:
            self.trans_width_in = []
            for mi in self.module_index:
                hidden_dim_small = self.weights[0].shape
                self.trans_width_in.append(TransWidthParams(hidden_dim_small[1], self.in_features,
                        learnable=learn_width, init_scheme=init_scheme_width, init_noise=init_noise_width, rank=rank_w))

            width_in_tie = self.trans_width_in



        if width_out_tie is None:
            self.trans_width_out = []
            for mi in self.module_index:
                hidden_dim_small = self.weights[0].shape
                self.trans_width_out.append(TransWidthParams(hidden_dim_small[0], self.out_features,
                        learnable=learn_width, init_scheme=init_scheme_width, init_noise=init_noise_width, rank=rank_w))

            width_out_tie = self.trans_width_out


        self.get_width_weight = lambda: ([width_in_tie_i.trans_weight for width_in_tie_i in width_in_tie], [width_out_tie_i.trans_weight for width_out_tie_i in width_out_tie])

        self.reset_parameters()



    def reset_parameters(self):

        if self.residual:
            for rw, rb in zip(self.residual_weight, self.residual_bias):
                nn.init.uniform_(rw, -self.residual_noise, self.residual_noise)
                nn.init.uniform_(rb, -self.residual_noise, self.residual_noise)

        if hasattr(self, 'trans_depth_weight'):
            self.trans_depth_weight.reset_parameters()
        if hasattr(self, 'trans_width_in'):
            self.trans_width_in.reset_parameters()
        if hasattr(self, 'trans_width_out'):
            self.trans_width_out.reset_parameters()


    def get_params(self):

        in_dim_expand, out_dim_expand = self.get_width_weight()
        
        weight_in_trans = []
        
        if in_dim_expand is not None: 
            for weight_i, in_expand_i in zip(self.weights, in_dim_expand):
                in_expand_i = torch.transpose(in_expand_i, 0, 1)
                weight_t = torch.cat([weight_i, torch.matmul(weight_i, in_expand_i)], 1) # expand in dimension

                weight_in_trans.append(weight_t)


        weight_out_trans = []
        bias_out_trans = []

        if out_dim_expand is not None:
            res_i = 0
            for weight_i, bias_i, out_expand_i in zip(weight_in_trans, self.bias, out_dim_expand):
                weight_t = torch.cat([weight_i, torch.matmul(out_expand_i, weight_i)], 0) # expand out dimension
                if self.bias:
                    bias_t = torch.cat([bias_i, torch.matmul(out_expand_i, bias_i)], 0) # expand out dimension

                if self.residual:
                    weight_t = weight_t + self.residual_weight[res_i]
                    if self.bias:
                        bias_t = bias_t + self.residual_bias[res_i]
                    res_i = res_i + 1
                
                weight_out_trans.append(weight_t)
                bias_out_trans.append(bias_t)

            if len(weight_out_trans) > 1:
                weight_out_trans = torch.stack(weight_out_trans, -1)
                bias_out_trans = torch.stack(bias_out_trans, -1)


        if hasattr(self, 'get_depth_weight'):
            trans_weights, trans_bias = self.get_depth_weight()
            weight = torch.sum(weight_out_trans * trans_weights, -1)
            if self.bias:
                bias = torch.sum(bias_out_trans * trans_bias, -1)
        else:
            weight = weight_out_trans
            if self.bias:
                bias = bias_out_trans

        return weight, bias


    def forward(self, input):
        weight, bias = self.get_params()
        return F.linear(input, weight, bias)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class TransLayeredNorm(nn.Module):

    def __init__(self, model, module_index, normalized_shape, layer_index=-1, init_scheme_depth='rand', init_noise_depth=0.03, learn_depth=True,
        init_scheme_width='rand', init_noise_width=0.03, learn_width=True, eps=1e-5, residual=False, depth_tie=None, width_out_tie=None, residual_noise=0.01, rank_w=8):

        super(TransLayeredNorm, self).__init__()
        
        self.weights = []
        self.bias = []
        self.module_index = module_index

        for mi in self.module_index:
            self.weights.append(getattr(model, K2N(mi) + '_weight'))   
            self.bias.append(getattr(model, K2N(mi) + '_bias', None))

        self.normalized_shape = normalized_shape
        self.elementwise_affine = True # only support elementwise_affine
        self.eps = eps
        self.residual = residual
        self.residual_noise = residual_noise

        if residual:
            self.residual_weight = [nn.Parameter(torch.empty(self.normalized_shape)) for i in self.module_index]
            self.residual_bias = [nn.Parameter(torch.empty(self.normalized_shape)) for i in self.module_index]

        # for embedding or classifier layer, specify layer_index to -1
        if layer_index >= 0:
            if depth_tie is None:
                num_layers_comb = len(self.module_index)
                self.trans_depth_weight = TransDepthParams(layer_index, num_layers_comb, bias=True,
                    learnable=learn_depth, init_scheme=init_scheme_depth, init_noise=init_noise_depth)
                depth_tie = self.trans_depth_weight

            self.get_depth_weight = lambda: (depth_tie.trans_weight, getattr(depth_tie, 'trans_bias', depth_tie.trans_weight))


        if width_out_tie is None:
            
            self.trans_width_out = []
            for mi in self.module_index:
                hidden_dim_small = self.weights[0].shape
                self.trans_width_out.append(TransWidthParams(hidden_dim_small[0], self.normalized_shape,
                        learnable=learn_width, init_scheme=init_scheme_width, init_noise=init_noise_width, rank=rank_w)
                )

            width_out_tie = self.trans_width_out


        self.get_width_weight = lambda: [width_out_tie_i.trans_weight for width_out_tie_i in width_out_tie]


    def reset_parameters(self):

        if self.residual:
            for rw, rb in zip(self.residual_weight, self.residual_bias):
                nn.init.uniform_(rw, -self.residual_noise, self.residual_noise)
                nn.init.uniform_(rb, -self.residual_noise, self.residual_noise)

        if hasattr(self, 'trans_depth_weight'):
            self.trans_depth_weight.reset_parameters()
        if hasattr(self, 'trans_width_out'):
            self.trans_width_out.reset_parameters()


    def get_params(self):
        
        out_dim_expand = self.get_width_weight()
        
        weight_trans = []
        bias_trans = []

        if out_dim_expand is not None:
            res_i = 0
            for weight_i, bias_i, expand_i in zip(self.weights, self.bias, out_dim_expand):
                weight_t = torch.cat([weight_i, torch.matmul(expand_i, weight_i)], 0) # expand out dimension
                bias_t = torch.cat([bias_i, torch.matmul(expand_i, bias_i)], 0) # expand out dimension

                if self.residual:
                    weight_t = weight_t + self.residual_weight[res_i]
                    if bias_i:
                        bias_t = bias_t + self.residual_bias[res_i]
                    res_i = res_i + 1
                
                weight_trans.append(weight_t)
                bias_trans.append(bias_t)

            weight_trans = torch.stack(weight_trans, -1)
            bias_trans = torch.stack(bias_trans, -1)
            

        if hasattr(self, 'get_depth_weight'):
            trans_weights, trans_bias = self.get_depth_weight()
            weight = torch.sum(weight_trans * trans_weights, -1)
            bias = torch.sum(bias_trans * trans_bias, -1)

        else:
            weight = weight_trans
            bias = bias_trans

        return weight, bias


    def forward(self, input):
        weight, bias = self.get_params()
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
    

    def extra_repr(self):
        return '{}, eps={}, elementwise_affine={}'.format(self.normalized_shape, self.eps, self.elementwise_affine)





class TransFcNorm(nn.Module):

    def __init__(self, model, module_name, normalized_shape, layer_index=-1, init_scheme_depth='rand', init_noise_depth=0.03, learn_depth=True,
        init_scheme_width='rand', init_noise_width=0.03, learn_width=True, eps=1e-5, residual=False, depth_tie=None, width_out_tie=None, residual_noise=0.01, rank_w=8):
    
        super(TransFcNorm, self).__init__()

        self.get_weights = lambda: getattr(model, K2N(module_name) + '_weight')
        self.get_bias = lambda: getattr(model, K2N(module_name) + '_bias', None)

        self.normalized_shape = normalized_shape
        self.elementwise_affine = True # only support elementwise_affine
        self.eps = eps
        self.residual = residual
        self.residual_noise = residual_noise

        if residual:
            self.residual_weight = nn.Parameter(torch.empty(self.normalized_shape)) 
            self.residual_bias = nn.Parameter(torch.empty(self.normalized_shape))

        # for embedding or classifier layer, specify layer_index to -1
        if layer_index >= 0:
            if depth_tie is None:
                num_layers_comb = len(self.module_index)
                self.trans_depth_weight = TransDepthParams(layer_index, num_layers_comb, bias=True,
                    learnable=learn_depth, init_scheme=init_scheme_depth, init_noise=init_noise_depth)
                depth_tie = self.trans_depth_weight

            self.get_depth_weight = lambda: (depth_tie.trans_weight, getattr(depth_tie, 'trans_bias', depth_tie.trans_weight))

        if width_out_tie is None:
            hidden_dim_small = self.get_weights().shape[0]
            self.trans_width_out = TransWidthParams(hidden_dim_small, self.normalized_shape,
                learnable=learn_width, init_scheme=init_scheme_width, init_noise=init_noise_width, rank=rank_w)
            width_out_tie = self.trans_width_out


        self.get_width_weight = lambda: width_out_tie.trans_weight


    def reset_parameters(self):

        if self.residual:
            nn.init.uniform_(self.residual_weight, -self.residual_noise, self.residual_noise)
            nn.init.uniform_(self.residual_bias, -self.residual_noise, self.residual_noise)

        if hasattr(self, 'trans_depth_weight'):
            self.trans_depth_weight.reset_parameters()
        if hasattr(self, 'trans_width_out'):
            self.trans_width_out.reset_parameters()


    def get_params(self):

        weight = self.get_weights()
        bias = self.get_bias()

        out_dim_expand = self.get_width_weight()


        if out_dim_expand is not None:
            weight = torch.cat([weight, torch.matmul(out_dim_expand, weight)], 0) # expand out dimension
            bias = torch.cat([bias, torch.matmul(out_dim_expand, bias)], 0) # expand out dimension

        if self.residual:
            weight = weight + self.residual_weight
            bias = bias + self.residual_bias

        return weight, bias


    def forward(self, input):
        weight, bias = self.get_params()
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
    

    def extra_repr(self):
        return '{}, eps={}, elementwise_affine={}'.format(self.normalized_shape, self.eps, self.elementwise_affine)





class TransPatchEmbedding(nn.Module):

    def __init__(self, model, module_name, embedding_dim, 
        img_size=224, patch_size=16, in_chans=3, norm_layer=None, flatten=True,
        init_scheme_width='rand', init_noise_width=0.03, learn_width=True,
        residual=False, width_out_tie=None, residual_noise=0.01, rank_w=8):

        super(TransPatchEmbedding, self).__init__()

        self.get_weights = lambda: getattr(model, K2N(module_name) + '_weight')
        self.get_bias = lambda: getattr(model, K2N(module_name) + '_bias')

        self.embedding_dim = embedding_dim

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        
        self.residual = residual
        self.residual_noise = residual_noise

        if residual:
            self.residual_weight = nn.Parameter(torch.empty((embedding_dim, self.in_chans, self.patch_size, self.patch_size)))
            self.residual_bias = nn.Parameter(torch.empty(embedding_dim))

        if width_out_tie is None:
            hidden_dim_small = self.get_weights().shape
            self.trans_width_out = TransWidthParams(hidden_dim_small[0], self.embedding_dim,
                learnable=learn_width, init_scheme=init_scheme_width, init_noise=init_noise_width, rank=rank_w)
            width_out_tie = self.trans_width_out

        self.get_width_weight = lambda: width_out_tie.trans_weight


    def reset_parameters(self):
        if self.residual:
            nn.init.uniform_(self.residual_weight, -self.residual_noise, self.residual_noise)

        if hasattr(self, 'trans_width_out'):
            self.trans_width_out.reset_parameters()


    def get_params(self):

        weight = self.get_weights()
        bias = self.get_bias()

        out_dim_expand = self.get_width_weight()
        if out_dim_expand is not None:
            weight = torch.cat([weight, torch.matmul(out_dim_expand, weight.view(weight.shape[0], -1)).view(-1, self.in_chans, self.patch_size[0], self.patch_size[1])], 0) # expand out dimension
            bias = torch.cat([bias, torch.matmul(out_dim_expand, bias)], 0) # expand out dimension

        if self.residual:
            weight = weight + self.residual_weight
            bias = bias + self.residual_bias

        return weight, bias


    def forward(self, input):
        weight, bias = self.get_params()
        return F.conv2d(input, weight, bias,stride=self.patch_size)


    def extra_repr(self):
        s = '{}'.format(self.embedding_dim)
        if self.img_size is not None:
            s += ', img_size={}'.format(self.img_size)
        if self.patch_size is not None:
            s += ', patch_size={}'.format(self.patch_size)
        if self.num_patches is not None:
            s += ', num_patches={}'.format(self.num_patches)
        return s.format(**self.__dict__)






