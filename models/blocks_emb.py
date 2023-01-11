import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, defaultdict
from models.utils import *
from models.layers import MlpBlock_Real, ColumnMaxPooling, ColumnSumPooling, MlpBlock_vec, AttentionBlock_vec, Concat, Identity, Permute, Matmul, Matmul_zerodiag, Add, GraphAttentionLayer, GraphNorm, Diag


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

def block_inside(features, depth_of_mlp, constant_n_vertices=True, **kwargs):
    return block(features,features,depth_of_mlp,constant_n_vertices=constant_n_vertices)


def block_att_inside(in_features, depth_of_mlp, num_heads=1, constant_n_vertices=True, **kwargs):
    class config:
        n_embd = in_features
        n_heads = num_heads
        d_of_mlp = depth_of_mlp

    return {
        'in': Identity(),
        'mlp3': GraphAttentionLayer(config, dmt= not constant_n_vertices)
    }


def block_res(features, depth_of_mlp, constant_n_vertices=True, **kwargs):
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(features, features, depth_of_mlp, constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul(), ['in', 'mlp1']),
        #'mult': (Matmul_zerodiag(), ['in', 'mlp1']),
        #'gn_mult': (nn.InstanceNorm2d(out_features,track_running_stats=False), ['mult']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3i': MlpBlock_Real(2*features, features,depth_of_mlp, constant_n_vertices=constant_n_vertices),
       #'mlp1': (MlpBlock_Real(features, features, depth_of_mlp,constant_n_vertices=constant_n_vertices), ['in']),
        #'mlp2': (MlpBlock_Real(features, features, depth_of_mlp, constant_n_vertices=constant_n_vertices), ['in']),
        #'mult': (Matmul(), ['mlp1', 'mlp2']),
        #'cat':  (Concat(), ['mult', 'in']),
        #'mlp3i': MlpBlock_Real(2*features, features,depth_of_mlp, constant_n_vertices=constant_n_vertices),
        'mlp3': (Add(), ['in', 'mlp3i'])
    }
    
def block_diag(in_features, out_features, depth_of_mlp, constant_n_vertices=True, **kwargs):
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(in_features, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices), ['in']),
        'mlp2': (MlpBlock_Real(in_features, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul_zerodiag(), ['mlp1', 'mlp2']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(in_features+out_features, out_features,depth_of_mlp, constant_n_vertices=constant_n_vertices)
    }

def block_diag_inside(features, depth_of_mlp, constant_n_vertices=True, **kwargs):
    return block_diag(features,features,depth_of_mlp, constant_n_vertices=constant_n_vertices)

def block_sym(in_features, out_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(in_features, out_features, depth_of_mlp,constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul(), ['mlp1', 'mlp1']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(in_features+out_features, out_features,depth_of_mlp,constant_n_vertices=constant_n_vertices)
    }

def block_sym_inside(features, depth_of_mlp, constant_n_vertices=True, **kwargs):
    return block_sym(features, features, depth_of_mlp, constant_n_vertices=constant_n_vertices)

def block_sym_diag(in_features, out_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(in_features, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul(), ['mlp1', 'mlp1']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(in_features+out_features, out_features,depth_of_mlp, constant_n_vertices=constant_n_vertices)
    }

def block_sym_diag_inside(features, depth_of_mlp, constant_n_vertices=True, **kwargs):
    return block_sym_diag(features, features, depth_of_mlp, constant_n_vertices=constant_n_vertices)

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

def node_embedding_full(original_features_num, num_blocks, in_features,out_features, depth_of_mlp,
         block_init=block, block_inside=block_inside, constant_n_vertices=True, **kwargs):
    d = {'graphs': Identity()}
    d['bm'] = base_model_block(original_features_num, num_blocks, out_features, 
            depth_of_mlp, block_init, block_inside, constant_n_vertices=constant_n_vertices)
    d['cat_all'] = (Concat(), ['bm/block'+str(i+1)+'/mlp3' for i in range(num_blocks)])
    d['suffix'] = ColumnMaxPooling()
    return d

def base_model_block(original_features_num, num_blocks, out_features, depth_of_mlp, num_heads,
        block_init = block, block_inside=block_inside, constant_n_vertices=True):
    d = {'in': Identity()}
    #last_layer_features = original_features_num
    d['block1'] = block_init(original_features_num, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    for i in range(1,num_blocks):
        d['block'+str(i+1)] = block_inside(out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices, num_heads=num_heads)
    return d

def node_embedding_block(original_features_num, num_blocks, out_features, depth_of_mlp, num_heads,
        block_init = block, block_inside=block_inside, constant_n_vertices=True,  **kwargs):
    d = {'in': Identity()}
    d['bm'] = base_model_block(original_features_num=original_features_num, num_blocks=num_blocks, out_features=out_features, 
                    depth_of_mlp=depth_of_mlp, block_init =block_init, block_inside=block_inside,constant_n_vertices=constant_n_vertices, num_heads=num_heads)
    #d['last'] = block_att_inside(out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices, num_heads=num_heads)
    #d['suffix'] = Identity()
    d['suffix'] = ColumnMaxPooling()
    return d

def base_model_rec(original_features_num, num_blocks, out_features, depth_of_mlp, num_heads,
        block_init = block, block_inside=block_inside, constant_n_vertices=True):
    d = {'in': Identity()}
    #last_layer_features = original_features_num
    d['block1'] = block_init(original_features_num, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    d['block2'] = block_inside(out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices, num_heads=num_heads)
    for i in range(2,num_blocks):
        d['block'+str(i+1)] = d['block2']#block_inside(out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices, num_heads=num_heads)
    return d

def base_model_recall(original_features_num, num_blocks, out_features, depth_of_mlp, num_heads=1,
        block_init = block_emb, block_inside=block, constant_n_vertices=True):
    d = {'in': GraphNorm(original_features_num)}
    #last_layer_features = original_features_num
    d['block1'] = block_init(original_features_num, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    #d['gn1'] = (GraphNorm(out_features), ['block1/mlp3'])
    d['cat1'] = (Concat(), ['in', 'block1/mlp3'])
    #d['cat1'] = (Concat(), ['in', 'gn1'])
    d['block2'] = block_inside(original_features_num+out_features,out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices)
    #d['gn2'] = (GraphNorm(out_features), ['block2/mlp3'])
    #d['block3'] = block_init(2*out_features, out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices)
    for i in range(2,num_blocks):
        d['cat'+str(i)] = (Concat(), ['in', 'block'+str(i)+'/mlp3'])
        #d['cat'+str(i)] = (Concat(), ['in', 'gn'+str(i)])
        d['block'+str(i+1)] = d['block2']#block_inside(out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices, num_heads=num_heads)
        #d['block'+str(i+1)] = block_inside(original_features_num+out_features,out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices)
        #d['gn'+str(i+1)] = (GraphNorm(out_features), ['block'+str(i+1)+'/mlp3'])
    return d

def base_model_recall_nogn(original_features_num, num_blocks, out_features, depth_of_mlp, num_heads,
        block_init = block, block_inside=block_inside, constant_n_vertices=True):
    d = {'in': GraphNorm(original_features_num)}
    d['block1'] = block_init(original_features_num, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    #d['gn1'] = (GraphNorm(out_features), ['block1/mlp3'])
    d['cat1'] = (Concat(), ['in', 'block1/mlp3'])
    #d['cat1'] = (Concat(), ['in', 'gn1'])
    d['block2'] = block_inside(original_features_num+out_features,out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices)
    #d['gn2'] = (GraphNorm(out_features), ['block2/mlp3'])
    #d['block3'] = block_init(2*out_features, out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices)
    for i in range(2,num_blocks):
        d['cat'+str(i)] = (Concat(), ['in', 'block'+str(i)+'/mlp3'])
        #d['cat'+str(i)] = (Concat(), ['in', 'gn'+str(i)])
        #d['block'+str(i+1)] = d['block2']#block_inside(out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices, num_heads=num_heads)
        d['block'+str(i+1)] = block_inside(original_features_num+out_features,out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices)
        #d['gn'+str(i+1)] = (GraphNorm(out_features), ['block'+str(i+1)+'/mlp3'])
    return d

# def base_model_recall(original_features_num, num_blocks, out_features, depth_of_mlp, num_heads,
#         block_init = block, block_inside=block_inside, constant_n_vertices=True):
#     d = {'in': Identity()}
#     #last_layer_features = original_features_num
#     d['block1'] = block_init(original_features_num, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
#     d['block2'] = block_inside(out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices, num_heads=num_heads)
#     d['cat2'] = (Concat(), ['block1/mlp3', 'block2/mlp3'])
#     d['block3'] = block_init(2*out_features, out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices)
#     for i in range(3,num_blocks):
#         d['cat'+str(i)] = (Concat(), ['block1/mlp3', 'block'+str(i)+'/mlp3'])
#         d['block'+str(i+1)] = d['block3']#block_inside(out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices, num_heads=num_heads)
#     return d

def node_embedding_rec(original_features_num, num_blocks, out_features, depth_of_mlp, num_heads=1,
        block_init = block_emb, block_inside=block, constant_n_vertices=True,  **kwargs):
    d = {'in': Identity()}
    d['bm'] = base_model_recall(original_features_num=original_features_num, num_blocks=num_blocks, out_features=out_features, 
                    depth_of_mlp=depth_of_mlp, block_init =block_init, block_inside=block_inside,constant_n_vertices=constant_n_vertices, num_heads=num_heads)
    d['suffix'] = ColumnMaxPooling()
    return d

def block_multi(features, depth_of_mlp, constant_n_vertices=True):
    ''' imcm: in_mult_cat_mlp3'''
    return {
        'in': Identity(),
        'mult': (Matmul(), ['in', 'in']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(2*features, features,depth_of_mlp,constant_n_vertices=constant_n_vertices)
    }

def block_multi_mlp(features, depth_of_mlp, constant_n_vertices=True):
    ''' immcm: in_mult_mlp1_cat_mlp3'''
    return {
        'in': Identity(),
        'mult': (Matmul(), ['in', 'in']),
        'mlp1': (MlpBlock_Real(features, features, depth_of_mlp,constant_n_vertices=constant_n_vertices), ['mult']),
        'cat':  (Concat(), ['mlp1', 'in']),
        'mlp3': MlpBlock_Real(2*features, features,depth_of_mlp, constant_n_vertices=constant_n_vertices)
    }

def block_mix(features, depth_of_mlp, constant_n_vertices=True):
    ''' 
    mzcm: mlp1_zmult_cat_mlp3
    mmcm: mlp1_mult_cat_mlp3
    '''
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(features, features, depth_of_mlp, constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul(), ['in', 'mlp1']),
        #'mult': (Matmul_zerodiag(), ['in', 'mlp1']),
        #'gn_mult': (nn.InstanceNorm2d(out_features,track_running_stats=False), ['mult']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(2*features, features,depth_of_mlp, constant_n_vertices=constant_n_vertices)
        #'mlp3': (Layernorm(out_features), ['mlp3_i'])
    }

def block_diag_mix(features, depth_of_mlp, constant_n_vertices=True):
    ''' 
    mzcm: mlp1_zmult_cat_mlp3
    mmcm: mlp1_mult_cat_mlp3
    '''
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(features, features, depth_of_mlp, constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul_zerodiag(), ['in', 'mlp1']),
        #'gn_mult': (nn.InstanceNorm2d(out_features,track_running_stats=False), ['mult']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(2*features, features,depth_of_mlp, constant_n_vertices=constant_n_vertices)
        #'mlp3': (Layernorm(out_features), ['mlp3_i'])
    }

def node_embedding_feature(in_features,out_features, depth_of_mlp, constant_n_vertices=True):
    d = {'nodes_f': Identity()}
    d['suffix'] = MlpBlock_vec(in_features,out_features, depth_of_mlp)
    return d

def node_embedding_transformer(original_nfeatures_num, features, depth_of_mlp, nb_heads=8,  constant_n_vertices=False):
    d = {'nodes_f': Identity()}
    d['in_nef'] = node_embedding_feature(original_nfeatures_num, features, depth_of_mlp)
    d['perm1'] = (lambda x: x.permute(0,2,1), ['in_nef/suffix'])
    d['att1'] = AttentionBlock_vec(features, depth_of_mlp, nb_heads=nb_heads)
    d['att2'] = AttentionBlock_vec(features, depth_of_mlp, nb_heads=nb_heads)
    d['att3'] = AttentionBlock_vec(features, depth_of_mlp, nb_heads=nb_heads)
    d['att4'] = AttentionBlock_vec(features, depth_of_mlp, nb_heads=nb_heads)
    #d['att5'] = AttentionBlock_vec(out_features, depth_of_mlp, 8)
    #d['att6'] = AttentionBlock_vec(out_features, depth_of_mlp, 8)
    #d['att7'] = AttentionBlock_vec(out_features, depth_of_mlp, 8)
    #d['cat_all'] = (Concat(), ['neg/suffix' , 'nef/mlp'])
    d['suffix'] = (lambda x: x.permute(0,2,1), ['att4'])
    #d['suffix'] = ColumnMaxPooling()
    return d

def graph_embedding_block(original_gfeatures_num, original_nfeatures_num, num_blocks, out_features, depth_of_mlp, block=block, constant_n_vertices=True):
    d = {'graphs': Identity()}
    d['neg'] = node_embedding_rec(original_gfeatures_num, num_blocks, out_features, depth_of_mlp, block,constant_n_vertices=constant_n_vertices)
    d['nodes_f'] = Identity()
    d['nef'] = node_embedding_transformer(original_nfeatures_num, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    d['cat_all'] = (Concat(), ['neg/suffix' , 'nef/suffix'])
    d['suffix'] = ColumnMaxPooling()
    return d

def graph_embedding_feature(original_gfeatures_num, original_nfeatures_num, 
num_blocks, out_features, depth_of_mlp, constant_n_vertices=True):
    d = {'graphs': GraphNorm(original_gfeatures_num)}
    d['nodes_f'] = Identity()
    d['nef'] = node_embedding_transformer(original_nfeatures_num, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    d['replicate'] = (Identity(), ['nodes_f']) 
    d['nefg'] = node_embedding_feature(original_nfeatures_num,out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
            
    d['nefg_diag'] = (Diag(), ['nefg/suffix'])
    d['in_neg'] = (Add(), ['nefg_diag', 'graphs'])
    d['neg'] = node_embedding_rec(original_gfeatures_num, num_blocks, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    d['cat_all'] = (Concat(), ['neg/suffix' , 'nef/suffix'])
    d['suffix'] = ColumnMaxPooling()
    return d

def graph_embedding_transformer(original_gfeatures_num, original_nfeatures_num, num_blocks, in_features,out_features, depth_of_mlp, block=block, constant_n_vertices=True):
    d = {'nodes_f': Identity()}
    #d['neg'] = node_embedding_block(original_gfeatures_num, num_blocks, out_features, depth_of_mlp, block,constant_n_vertices=constant_n_vertices)
    #d['nodes_f'] = Identity()
    d['nef'] = node_embedding_transformer(original_nfeatures_num, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    #d['cat_all'] = (Concat(), ['neg/suffix' , 'nef/suffix'])
    d['suffix'] = ColumnMaxPooling()
    return d

#################################################################
def scaled_block(in_features, out_features, depth_of_mlp):
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(in_features, out_features, depth_of_mlp), ['in']),
        'mlp2': (MlpBlock_Real(in_features, out_features, depth_of_mlp), ['in']),
        'mult': (Matmul(), ['mlp1', 'mlp2']),
        'gn_mult': (nn.InstanceNorm2d(out_features,track_running_stats=False), ['mult']),
        'cat':  (Concat(), ['gn_mult', 'in']),
        'mlp3': MlpBlock_Real(in_features+out_features, out_features,depth_of_mlp),
        'gn_out': (nn.InstanceNorm2d(out_features,track_running_stats=False), ['mlp3'])
    }





def iter_model(original_features_num, num_blocks, in_features,out_features, depth_of_mlp):
    d = {'in': Identity()}
    #last_layer_features = original_features_num
    d['prod1'] = (MlpBlock_Real(original_features_num, in_features, depth_of_mlp), ['in'])
    for i in range(1,num_blocks-1):
        d['mlp'+str(i+1)] = (MlpBlock_Real(original_features_num, in_features, depth_of_mlp), ['in'])
        #last_layer_features = in_features
        d['mult'+str(i+1)] = (Matmul(), ['prod'+str(i), 'prod'+str(i)])
        d['cat'+str(i+1) ] = (Concat(), ['mult'+str(i+1), 'prod'+str(i)])
        d['prod'+str(i+1)] = (MlpBlock_Real(2*in_features, in_features, depth_of_mlp), ['cat'+str(i+1)])
    d['mlp'+str(num_blocks)] = (MlpBlock_Real(original_features_num, out_features, depth_of_mlp), ['in'])
    d['prod'+str(num_blocks)] = (Matmul(), ['prod'+str(num_blocks-1), 'mlp'+str(num_blocks)])
    return d

def node_embedding_iter(original_features_num, num_blocks, in_features,out_features, depth_of_mlp, **kwargs):
    d = {'graphs': Identity()}
    d['bm'] = iter_model(original_features_num, num_blocks, in_features,out_features, depth_of_mlp)
    d['cat_all'] = (Concat(), ['bm/prod'+str(i+1) for i in range(num_blocks)])
    #d['suffix'] = ColumnSumPooling()
    d['suffix'] = ColumnMaxPooling()
    return d



def graph_embedding_mlp(original_gfeatures_num, original_nfeatures_num, num_blocks, in_features,out_features, depth_of_mlp, block=block):
    d = {'nodes_f': Identity()}
    d['nef'] = node_embedding_feature(original_nfeatures_num, out_features, 3*depth_of_mlp)
    #d['cat_all'] = (Concat(), ['neg/suffix' , 'nef/mlp'])
    d['suffix'] = ColumnMaxPooling()
    return d



