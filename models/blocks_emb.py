import torch
import torch.nn as nn
import torch.nn.functional as F
#from collections import namedtuple, defaultdict
from models.utils import *
from models.layers import MlpBlock_Real, ColumnMaxPooling,  Concat, Identity,  Matmul#,ColumnSumPooling, MlpBlock_vec, AttentionBlock_vec, Permute, Matmul_zerodiag, Add, GraphAttentionLayer, GraphNorm, Diag, Rec_block, Recall_block


def block_emb(in_features, out_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'mlp3': MlpBlock_Real(in_features, out_features,depth_of_mlp,
            constant_n_vertices=constant_n_vertices)
    }

def block(in_features, out_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(in_features, out_features, depth_of_mlp,
            constant_n_vertices=constant_n_vertices), ['in']),
        'mlp2': (MlpBlock_Real(in_features, out_features, depth_of_mlp,
                constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul(), ['mlp1', 'mlp2']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(in_features+out_features, out_features,depth_of_mlp,
            constant_n_vertices=constant_n_vertices)
    }

def base_model(original_features_num, num_blocks, in_features,out_features, depth_of_mlp, block=block, constant_n_vertices=True):
    d = {'in': Identity()}
    last_layer_features = original_features_num
    for i in range(num_blocks-1):
        d['block'+str(i+1)] = block(last_layer_features, in_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
        last_layer_features = in_features
    d['block'+str(num_blocks)] = block(last_layer_features, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    return d

def node_embedding(original_features_num, num_blocks, in_features,out_features, depth_of_mlp,
     block=block, constant_n_vertices=True, **kwargs):
    d = {'in': Identity()}
    d['bm'] = base_model(original_features_num, num_blocks, in_features,out_features, depth_of_mlp, block, constant_n_vertices=constant_n_vertices)
    d['suffix'] = ColumnMaxPooling()
    return d
