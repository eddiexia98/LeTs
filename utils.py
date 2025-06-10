"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)



class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))



def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()



def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()



def is_main_process():
    return get_rank() == 0



def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)



def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)




@torch.no_grad()
def initialize_model_with_lets_seq(args, target_model):


    def layer_mapping_sheme_1():
        return {
            0 : [0, 1], 1 : [0, 1], 2 : [0, 1], 3 : [0, 1],
            4 : [2, 3], 5 : [2, 3], 6 : [2, 3], 7 : [2, 3],
            8 : [4, 5], 9 : [4, 5], 10 : [4, 5], 11 : [4, 5],
            12 : [6, 7], 13 : [6, 7], 14 : [6, 7], 15 : [6, 7]
        }


    if args.mapping_scheme == '8_16':
        layer_mapping_dict = layer_mapping_sheme_1()
    else:
        raise NotImplementedError


    dict_model_coeff = torch.load(args.load_gene, map_location=torch.device('cpu'))

    model_coeff = dict_model_coeff['model']


    dict_ckpt = {}
    
    embedding_dim = target_model.patch_embed.proj.weight.data.shape[0]
    comp_dim = embedding_dim - model_coeff['__sm_patch_embed_proj_weight'].data.shape[0]



    ################################### initialize pos_embed
    dict_ckpt["pos_embed"] = model_coeff["pos_embed"].data[:, :, 0 : embedding_dim]
    



    ################################### initialize patch_proj 
    src_patch_proj_weight = '__sm_patch_embed_proj_weight'
    src_patch_proj_bias = '__sm_patch_embed_proj_bias'

    tgt_patch_proj_weight = 'patch_embed.proj.weight'
    tgt_patch_proj_bias = 'patch_embed.proj.bias'
    
    # expand out dimension
    patch_embed_t_width_weight = 'patch_embed.trans_width_emb.trans_weight'

    # catch more times
    comp_dim_tmp = comp_dim

    out_dim_expand = model_coeff[patch_embed_t_width_weight].data[ 0 : comp_dim_tmp, :]

    dict_ckpt[tgt_patch_proj_weight] = model_coeff[src_patch_proj_weight]
    dict_ckpt[tgt_patch_proj_bias] = model_coeff[src_patch_proj_bias]


    dict_ckpt[tgt_patch_proj_weight] = torch.cat([dict_ckpt[tgt_patch_proj_weight], 
    torch.matmul(out_dim_expand, model_coeff[src_patch_proj_weight].view(model_coeff[src_patch_proj_weight].shape[0], -1)).view(-1, model_coeff[src_patch_proj_weight].shape[1], model_coeff[src_patch_proj_weight].shape[2], model_coeff[src_patch_proj_weight].shape[3])], 0) 

    dict_ckpt[tgt_patch_proj_bias] = torch.cat([dict_ckpt[tgt_patch_proj_bias], torch.matmul(out_dim_expand, model_coeff[src_patch_proj_bias])], 0) 



    mapping_layer = args.mapping_layer.split(',')
    mapping_layer = [int(stri) for stri in mapping_layer]  

    mapping_layer_coeff = args.mapping_layer_coeff.split(',')
    mapping_layer_coeff = [int(stri) for stri in mapping_layer_coeff] 

    for layer_i in range(0, len(target_model.blocks)):
        

        layer_mapping_dict_i = layer_mapping_dict[mapping_layer[layer_i]]

        ################################### initialize norm1 
        norm1_weights = []
        norm1_bias = []
        for mi in layer_mapping_dict_i:
            src_norm1_weight = '__sm_{}_norm1_weight'.format(mi)
            src_norm1_bias = '__sm_{}_norm1_bias'.format(mi)

            norm1_weights.append(model_coeff[src_norm1_weight])   
            norm1_bias.append(model_coeff[src_norm1_bias])


        patch_embed_t_width_weight = 'patch_embed.trans_width_emb.trans_weight'
        src_patch_proj_weight = '__sm_patch_embed_proj_weight'

        # catch more times
        
        comp_dim_tmp = comp_dim

        out_dim_expand = [model_coeff[patch_embed_t_width_weight].data[ 0 : comp_dim_tmp, :] for i in layer_mapping_dict_i]

        weight_trans = []
        bias_trans = []

        for weight_i, bias_i, expand_i in zip(norm1_weights, norm1_bias, out_dim_expand):
            
            weight_t = weight_i
            bias_t = bias_i

            weight_t = torch.cat([weight_t, torch.matmul(expand_i, weight_i)], 0) # expand out dimension
            bias_t = torch.cat([bias_t, torch.matmul(expand_i, bias_i)], 0) # expand out dimension
            
            weight_trans.append(weight_t)
            bias_trans.append(bias_t)

        weight_trans = torch.stack(weight_trans, -1)
        bias_trans = torch.stack(bias_trans, -1)



        norm1_t_depth_weight = 'blocks.{}.norm1.trans_depth_weight.trans_weight'.format(mapping_layer_coeff[layer_i])
        norm1_t_depth_bias = 'blocks.{}.norm1.trans_depth_weight.trans_bias'.format(mapping_layer_coeff[layer_i])

        weight = torch.sum(weight_trans * model_coeff[norm1_t_depth_weight], -1)
        bias = torch.sum(bias_trans * model_coeff[norm1_t_depth_bias], -1)

        tgt_norm1_weight = 'blocks.{}.norm1.weight'.format(layer_i)
        tgt_norm1_bias = 'blocks.{}.norm1.bias'.format(layer_i)

        dict_ckpt[tgt_norm1_weight] = weight
        dict_ckpt[tgt_norm1_bias] = bias



        ################################### initialize MHA
        ############## q
        q_weights = []
        q_bias = []
        for mi in layer_mapping_dict_i:
            src_q_weight = '__sm_{}_attn_q_weight'.format(mi)
            src_q_bias = '__sm_{}_attn_q_bias'.format(mi)

            q_weights.append(model_coeff[src_q_weight])   
            q_bias.append(model_coeff[src_q_bias])

        patch_embed_t_width_weight = 'patch_embed.trans_width_emb.trans_weight'
        src_patch_proj_weight = '__sm_patch_embed_proj_weight'

        # catch more times
        comp_dim_tmp = comp_dim


        in_dim_expand = [model_coeff[patch_embed_t_width_weight].data[ 0 : comp_dim_tmp, :] for mi in layer_mapping_dict_i]
        out_dim_expand = []
        for mi in layer_mapping_dict_i:
            out_dim_expand.append(model_coeff[f'blocks.{mi}.trans_width_q.trans_weight'].data[ 0 : comp_dim_tmp, :])


        weight_in_trans = []

        for weight_i, in_expand_i in zip(q_weights, in_dim_expand):
            weight_t = weight_i

            in_expand_i = torch.transpose(in_expand_i, 0, 1)

            weight_t = torch.cat([weight_t, torch.matmul(weight_i, in_expand_i)], 1) # expand in dimension
            
            weight_in_trans.append(weight_t)


        weight_out_trans = []
        bias_out_trans = []

        for weight_i, bias_i, out_expand_i in zip(weight_in_trans, q_bias, out_dim_expand):

            weight_t = weight_i
            bias_t = bias_i

            weight_t = torch.cat([weight_t, torch.matmul(out_expand_i, weight_i)], 0) # expand out dimension
            bias_t = torch.cat([bias_t, torch.matmul(out_expand_i, bias_i)], 0) # expand out dimension
            
            weight_out_trans.append(weight_t)
            bias_out_trans.append(bias_t)


        weight_out_trans = torch.stack(weight_out_trans, -1)
        bias_out_trans = torch.stack(bias_out_trans, -1)


        q_t_depth_weight = 'blocks.{}.attn.q.trans_depth_weight.trans_weight'.format(mapping_layer_coeff[layer_i])
        q_t_depth_bias = 'blocks.{}.attn.q.trans_depth_weight.trans_bias'.format(mapping_layer_coeff[layer_i])


        weight = torch.sum(weight_out_trans * model_coeff[q_t_depth_weight], -1)
        bias = torch.sum(bias_out_trans * model_coeff[q_t_depth_bias], -1)

        tgt_q_weight = 'blocks.{}.attn.q.weight'.format(layer_i)
        tgt_q_bias = 'blocks.{}.attn.q.bias'.format(layer_i)

        dict_ckpt[tgt_q_weight] = weight
        dict_ckpt[tgt_q_bias] = bias



        ############## k
        k_weights = []
        k_bias = []
        for mi in layer_mapping_dict_i:
            src_k_weight = '__sm_{}_attn_k_weight'.format(mi)
            src_k_bias = '__sm_{}_attn_k_bias'.format(mi)

            k_weights.append(model_coeff[src_k_weight])   
            k_bias.append(model_coeff[src_k_bias])


            patch_embed_t_width_weight = 'patch_embed.trans_width_emb.trans_weight'
            src_patch_proj_weight = '__sm_patch_embed_proj_weight'

        # catch more times

        comp_dim_tmp = comp_dim


        in_dim_expand = [model_coeff[patch_embed_t_width_weight].data[ 0 : comp_dim_tmp, :] for mi in layer_mapping_dict_i]
        out_dim_expand = []
        for mi in layer_mapping_dict_i:
            out_dim_expand.append(model_coeff[f'blocks.{mi}.trans_width_k.trans_weight'].data[ 0 : comp_dim_tmp, :])


        weight_in_trans = []

        for weight_i, in_expand_i in zip(k_weights, in_dim_expand):
            weight_t = weight_i

            in_expand_i = torch.transpose(in_expand_i, 0, 1)

            weight_t = torch.cat([weight_t, torch.matmul(weight_i, in_expand_i)], 1) # expand in dimension
            
            weight_in_trans.append(weight_t)


        weight_out_trans = []
        bias_out_trans = []

        for weight_i, bias_i, out_expand_i in zip(weight_in_trans, k_bias, out_dim_expand):

            weight_t = weight_i
            bias_t = bias_i

            weight_t = torch.cat([weight_t, torch.matmul(out_expand_i, weight_i)], 0) # expand out dimension
            bias_t = torch.cat([bias_t, torch.matmul(out_expand_i, bias_i)], 0) # expand out dimension
            
            weight_out_trans.append(weight_t)
            bias_out_trans.append(bias_t)


        weight_out_trans = torch.stack(weight_out_trans, -1)
        bias_out_trans = torch.stack(bias_out_trans, -1)



        k_t_depth_weight = 'blocks.{}.attn.k.trans_depth_weight.trans_weight'.format(mapping_layer_coeff[layer_i])
        k_t_depth_bias = 'blocks.{}.attn.k.trans_depth_weight.trans_bias'.format(mapping_layer_coeff[layer_i])

        weight = torch.sum(weight_out_trans * model_coeff[k_t_depth_weight], -1)
        bias = torch.sum(bias_out_trans * model_coeff[k_t_depth_bias], -1)

        tgt_k_weight = 'blocks.{}.attn.k.weight'.format(layer_i)
        tgt_k_bias = 'blocks.{}.attn.k.bias'.format(layer_i)

        dict_ckpt[tgt_k_weight] = weight
        dict_ckpt[tgt_k_bias] = bias



        ############## v
        v_weights = []
        v_bias = []
        for mi in layer_mapping_dict_i:
            src_v_weight = '__sm_{}_attn_v_weight'.format(mi)
            src_v_bias = '__sm_{}_attn_v_bias'.format(mi)

            v_weights.append(model_coeff[src_v_weight])   
            v_bias.append(model_coeff[src_v_bias])


        patch_embed_t_width_weight = 'patch_embed.trans_width_emb.trans_weight'
        src_patch_proj_weight = '__sm_patch_embed_proj_weight'

        # catch more times
        comp_dim_tmp = comp_dim

        in_dim_expand = [model_coeff[patch_embed_t_width_weight].data[ 0 : comp_dim_tmp, :] for mi in layer_mapping_dict_i]
        out_dim_expand = []
        for mi in layer_mapping_dict_i:
            out_dim_expand.append(model_coeff[f'blocks.{mi}.trans_width_v.trans_weight'].data[ 0 : comp_dim_tmp, :])


        weight_in_trans = []

        for weight_i, in_expand_i in zip(v_weights, in_dim_expand):
            weight_t = weight_i

            in_expand_i = torch.transpose(in_expand_i, 0, 1)

            weight_t = torch.cat([weight_t, torch.matmul(weight_i, in_expand_i)], 1) # expand in dimension
            
            weight_in_trans.append(weight_t)


        weight_out_trans = []
        bias_out_trans = []

        for weight_i, bias_i, out_expand_i in zip(weight_in_trans, v_bias, out_dim_expand):

            weight_t = weight_i
            bias_t = bias_i

            weight_t = torch.cat([weight_t, torch.matmul(out_expand_i, weight_i)], 0) # expand out dimension
            bias_t = torch.cat([bias_t, torch.matmul(out_expand_i, bias_i)], 0) # expand out dimension
            
            weight_out_trans.append(weight_t)
            bias_out_trans.append(bias_t)

        weight_out_trans = torch.stack(weight_out_trans, -1)
        bias_out_trans = torch.stack(bias_out_trans, -1)


        v_t_depth_weight = 'blocks.{}.attn.v.trans_depth_weight.trans_weight'.format(mapping_layer_coeff[layer_i])
        v_t_depth_bias = 'blocks.{}.attn.v.trans_depth_weight.trans_bias'.format(mapping_layer_coeff[layer_i])

        weight = torch.sum(weight_out_trans * model_coeff[v_t_depth_weight], -1)
        bias = torch.sum(bias_out_trans * model_coeff[v_t_depth_bias], -1)

        tgt_v_weight = 'blocks.{}.attn.v.weight'.format(layer_i)
        tgt_v_bias = 'blocks.{}.attn.v.bias'.format(layer_i)

        dict_ckpt[tgt_v_weight] = weight
        dict_ckpt[tgt_v_bias] = bias



        ############## attn OUT
        proj_weights = []
        proj_bias = []
        for mi in layer_mapping_dict_i:
            src_proj_weight = '__sm_{}_attn_proj_weight'.format(mi)
            src_proj_bias = '__sm_{}_attn_proj_bias'.format(mi)

            proj_weights.append(model_coeff[src_proj_weight])   
            proj_bias.append(model_coeff[src_proj_bias])


        patch_embed_t_width_weight = 'patch_embed.trans_width_emb.trans_weight'
        src_patch_proj_weight = '__sm_patch_embed_proj_weight'

        # catch more times
        
        comp_dim_tmp = comp_dim

        out_dim_expand = [model_coeff[patch_embed_t_width_weight].data[ 0 : comp_dim_tmp, :] for mi in layer_mapping_dict_i]
        in_dim_expand = []
        for mi in layer_mapping_dict_i:
            in_dim_expand.append(model_coeff[f'blocks.{mi}.trans_width_v.trans_weight'].data[ 0 : comp_dim_tmp, :])


        weight_in_trans = []

        for weight_i, in_expand_i in zip(proj_weights, in_dim_expand):
            weight_t = weight_i

            in_expand_i = torch.transpose(in_expand_i, 0, 1)

            weight_t = torch.cat([weight_t, torch.matmul(weight_i, in_expand_i)], 1) # expand in dimension
            
            weight_in_trans.append(weight_t)


        weight_out_trans = []
        bias_out_trans = []

        for weight_i, bias_i, out_expand_i in zip(weight_in_trans, proj_bias, out_dim_expand):

            weight_t = weight_i
            bias_t = bias_i

            weight_t = torch.cat([weight_t, torch.matmul(out_expand_i, weight_i)], 0) # expand out dimension
            bias_t = torch.cat([bias_t, torch.matmul(out_expand_i, bias_i)], 0) # expand out dimension
            
            weight_out_trans.append(weight_t)
            bias_out_trans.append(bias_t)

        weight_out_trans = torch.stack(weight_out_trans, -1)
        bias_out_trans = torch.stack(bias_out_trans, -1)


        proj_t_depth_weight = 'blocks.{}.attn.proj.trans_depth_weight.trans_weight'.format(mapping_layer_coeff[layer_i])
        proj_t_depth_bias = 'blocks.{}.attn.proj.trans_depth_weight.trans_bias'.format(mapping_layer_coeff[layer_i])

        weight = torch.sum(weight_out_trans * model_coeff[proj_t_depth_weight], -1)
        bias = torch.sum(bias_out_trans * model_coeff[proj_t_depth_bias], -1)

        tgt_proj_weight = 'blocks.{}.attn.proj.weight'.format(layer_i)
        tgt_proj_bias = 'blocks.{}.attn.proj.bias'.format(layer_i)

        dict_ckpt[tgt_proj_weight] = weight
        dict_ckpt[tgt_proj_bias] = bias



        ################################### initialize norm2 
        norm2_weights = []
        norm2_bias = []
        for mi in layer_mapping_dict_i:
            src_norm2_weight = '__sm_{}_norm2_weight'.format(mi)
            src_norm2_bias = '__sm_{}_norm2_bias'.format(mi)

            norm2_weights.append(model_coeff[src_norm2_weight])   
            norm2_bias.append(model_coeff[src_norm2_bias])


        patch_embed_t_width_weight = 'patch_embed.trans_width_emb.trans_weight'
        src_patch_proj_weight = '__sm_patch_embed_proj_weight'

        # catch more times
        comp_dim_tmp = comp_dim

        out_dim_expand = [model_coeff[patch_embed_t_width_weight].data[ 0 : comp_dim_tmp, :] for mi in layer_mapping_dict_i]

        weight_trans = []
        bias_trans = []

        for weight_i, bias_i, expand_i in zip(norm2_weights, norm2_bias, out_dim_expand):
            
            weight_t = weight_i
            bias_t = bias_i


            weight_t = torch.cat([weight_t, torch.matmul(expand_i, weight_i)], 0) # expand out dimension
            bias_t = torch.cat([bias_t, torch.matmul(expand_i, bias_i)], 0) # expand out dimension
            
            weight_trans.append(weight_t)
            bias_trans.append(bias_t)

        weight_trans = torch.stack(weight_trans, -1)
        bias_trans = torch.stack(bias_trans, -1)


        norm2_t_depth_weight = 'blocks.{}.norm2.trans_depth_weight.trans_weight'.format(mapping_layer_coeff[layer_i])
        norm2_t_depth_bias = 'blocks.{}.norm2.trans_depth_weight.trans_bias'.format(mapping_layer_coeff[layer_i])

        weight = torch.sum(weight_trans * model_coeff[norm2_t_depth_weight], -1)
        bias = torch.sum(bias_trans * model_coeff[norm2_t_depth_bias], -1)

        tgt_norm2_weight = 'blocks.{}.norm2.weight'.format(layer_i)
        tgt_norm2_bias = 'blocks.{}.norm2.bias'.format(layer_i)

        dict_ckpt[tgt_norm2_weight] = weight
        dict_ckpt[tgt_norm2_bias] = bias



        ################################### initialize MLP
        ############## fc1
        fc1_weights = []
        fc1_bias = []
        for mi in layer_mapping_dict_i:
            src_fc1_weight = '__sm_{}_mlp_fc1_weight'.format(mi)
            src_fc1_bias = '__sm_{}_mlp_fc1_bias'.format(mi)

            fc1_weights.append(model_coeff[src_fc1_weight])   
            fc1_bias.append(model_coeff[src_fc1_bias])


        patch_embed_t_width_weight = 'patch_embed.trans_width_emb.trans_weight'
        src_patch_proj_weight = '__sm_patch_embed_proj_weight'

        # catch more times
        comp_dim_tmp = comp_dim

        in_dim_expand = [model_coeff[patch_embed_t_width_weight].data[ 0 : comp_dim_tmp, :] for mi in layer_mapping_dict_i]
        out_dim_expand = []
        for mi in layer_mapping_dict_i:
            out_dim_expand.append(model_coeff[f'blocks.{mi}.trans_width_ffn.trans_weight'].data[ 0 : comp_dim_tmp * 4, :])


        weight_in_trans = []

        for weight_i, in_expand_i in zip(fc1_weights, in_dim_expand):
            weight_t = weight_i

            in_expand_i = torch.transpose(in_expand_i, 0, 1)

            weight_t = torch.cat([weight_t, torch.matmul(weight_i, in_expand_i)], 1) # expand in dimension
            
            weight_in_trans.append(weight_t)

        weight_out_trans = []
        bias_out_trans = []

        for weight_i, bias_i, out_expand_i in zip(weight_in_trans, fc1_bias, out_dim_expand):

            weight_t = weight_i
            bias_t = bias_i

            weight_t = torch.cat([weight_t, torch.matmul(out_expand_i, weight_i)], 0) # expand out dimension
            bias_t = torch.cat([bias_t, torch.matmul(out_expand_i, bias_i)], 0) # expand out dimension
            
            weight_out_trans.append(weight_t)
            bias_out_trans.append(bias_t)

        weight_out_trans = torch.stack(weight_out_trans, -1)
        bias_out_trans = torch.stack(bias_out_trans, -1)


        fc1_t_depth_weight = 'blocks.{}.mlp.fc1.trans_depth_weight.trans_weight'.format(mapping_layer_coeff[layer_i])
        fc1_t_depth_bias = 'blocks.{}.mlp.fc1.trans_depth_weight.trans_bias'.format(mapping_layer_coeff[layer_i])

        weight = torch.sum(weight_out_trans * model_coeff[fc1_t_depth_weight], -1)
        bias = torch.sum(bias_out_trans * model_coeff[fc1_t_depth_bias], -1)

        tgt_fc1_weight = 'blocks.{}.mlp.fc1.weight'.format(layer_i)
        tgt_fc1_bias = 'blocks.{}.mlp.fc1.bias'.format(layer_i)

        dict_ckpt[tgt_fc1_weight] = weight
        dict_ckpt[tgt_fc1_bias] = bias


        ############## fc2
        fc2_weights = []
        fc2_bias = []
        for mi in layer_mapping_dict_i:
            src_fc2_weight = '__sm_{}_mlp_fc2_weight'.format(mi)
            src_fc2_bias = '__sm_{}_mlp_fc2_bias'.format(mi)

            fc2_weights.append(model_coeff[src_fc2_weight])   
            fc2_bias.append(model_coeff[src_fc2_bias])


        patch_embed_t_width_weight = 'patch_embed.trans_width_emb.trans_weight'
        src_patch_proj_weight = '__sm_patch_embed_proj_weight'

        # catch more times
        comp_dim_tmp = comp_dim

        out_dim_expand = [model_coeff[patch_embed_t_width_weight].data[ 0 : comp_dim_tmp, :] for mi in layer_mapping_dict_i]
        in_dim_expand = []
        for mi in layer_mapping_dict_i:
            in_dim_expand.append(model_coeff[f'blocks.{mi}.trans_width_ffn.trans_weight'].data[ 0 : comp_dim_tmp * 4, :])


        weight_in_trans = []

        for weight_i, in_expand_i in zip(fc2_weights, in_dim_expand):
            weight_t = weight_i

            in_expand_i = torch.transpose(in_expand_i, 0, 1)

            weight_t = torch.cat([weight_t, torch.matmul(weight_i, in_expand_i)], 1) # expand in dimension
            
            weight_in_trans.append(weight_t)


        weight_out_trans = []
        bias_out_trans = []

        for weight_i, bias_i, out_expand_i in zip(weight_in_trans, fc2_bias, out_dim_expand):

            weight_t = weight_i
            bias_t = bias_i


            weight_t = torch.cat([weight_t, torch.matmul(out_expand_i, weight_i)], 0) # expand out dimension
            bias_t = torch.cat([bias_t, torch.matmul(out_expand_i, bias_i)], 0) # expand out dimension
            
            weight_out_trans.append(weight_t)
            bias_out_trans.append(bias_t)

        weight_out_trans = torch.stack(weight_out_trans, -1)
        bias_out_trans = torch.stack(bias_out_trans, -1)



        fc2_t_depth_weight = 'blocks.{}.mlp.fc2.trans_depth_weight.trans_weight'.format(mapping_layer_coeff[layer_i])
        fc2_t_depth_bias = 'blocks.{}.mlp.fc2.trans_depth_weight.trans_bias'.format(mapping_layer_coeff[layer_i])

        weight = torch.sum(weight_out_trans * model_coeff[fc2_t_depth_weight], -1)
        bias = torch.sum(bias_out_trans * model_coeff[fc2_t_depth_bias], -1)

        tgt_fc2_weight = 'blocks.{}.mlp.fc2.weight'.format(layer_i)
        tgt_fc2_bias = 'blocks.{}.mlp.fc2.bias'.format(layer_i)

        dict_ckpt[tgt_fc2_weight] = weight
        dict_ckpt[tgt_fc2_bias] = bias




    ################################### initialize fc_norm 
    src_fcnorm_weight = '__sm_fc_norm_weight'
    src_fcnorm_bias = '__sm_fc_norm_bias'

    tgt_fcnorm_weight = 'fc_norm.weight'
    tgt_fcnorm_bias = 'fc_norm.bias'

    fc_norm_t_width_weight = 'patch_embed.trans_width_emb.trans_weight'
    src_patch_proj_weight = '__sm_patch_embed_proj_weight'


    # catch more times
    comp_dim_tmp = comp_dim

    weight_t = model_coeff[src_fcnorm_weight]
    bias_t = model_coeff[src_fcnorm_bias]

    out_dim_expand = model_coeff[fc_norm_t_width_weight].data[ 0 : comp_dim_tmp, :]

    # expand out dimension
    dict_ckpt[tgt_fcnorm_weight] = torch.cat([weight_t, torch.matmul(out_dim_expand, model_coeff[src_fcnorm_weight])], 0) 
    dict_ckpt[tgt_fcnorm_bias] = torch.cat([bias_t, torch.matmul(out_dim_expand, model_coeff[src_fcnorm_bias])], 0) 




    ################################### initialize head
    src_head_weight = '__sm_head_weight'
    src_head_bias = '__sm_head_bias'

    tgt_head_weight = 'head.weight'
    tgt_head_bias = 'head.bias'

    head_t_width_weight = 'trans_width_cls.trans_weight'
    src_patch_proj_weight = '__sm_patch_embed_proj_weight'


    # catch more times
    comp_dim_tmp = comp_dim

    weight_t = model_coeff[src_head_weight]

    # expand in dimension
    in_dim_expand = model_coeff[head_t_width_weight].data[ 0 : comp_dim_tmp, :]
    in_dim_expand = torch.transpose(in_dim_expand, 0, 1)
    dict_ckpt[tgt_head_weight] = torch.cat([weight_t, torch.matmul(model_coeff[src_head_weight], in_dim_expand)], 1) 
    dict_ckpt[tgt_head_bias] = model_coeff[src_head_bias]



    if args.data_set != 'IMNET':
        dict_ckpt[tgt_head_weight].data = dict_ckpt[tgt_head_weight].data[ 0 : args.nb_classes, : ]
        dict_ckpt[tgt_head_bias].data = dict_ckpt[tgt_head_bias].data[ 0 : args.nb_classes ]


    target_model.load_state_dict(dict_ckpt)
    

    del dict_ckpt


    return target_model



