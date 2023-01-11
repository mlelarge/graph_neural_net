from models.trainers import Siamese_Node_Exp, Graph_Classif_Exp #, scaled_block, block, block_sym, Graph_Classif_Exp
from toolbox.utils import load_json

from data_benchmarking_gnns.data_helper import NUM_LABELS, NUM_CLASSES

def get_siamese_model_exp(args, config_optim):  
    args_dict =  {'lr' : config_optim['lr'],
                'scheduler_decay': config_optim['scheduler_decay'],
                'scheduler_step': config_optim['scheduler_step']
    }
    original_features_num = args['original_features_num']
    node_emb = args['node_emb']
    print('Fetching model %s with (total = %s ) init %s and inside %s' % (node_emb['type'], node_emb['num_blocks'],
        node_emb['block_init'], node_emb['block_inside']))
    #print(node_emb)
    model =  Siamese_Node_Exp(original_features_num, node_emb, **args_dict)
    return model

def get_siamese_model_test(name):
    split_name = name.split("/")[-4]
    cname = name.split(split_name)[0]
    config = load_json(cname+'config.json')
    return Siamese_Node_Exp.load_from_checkpoint(name, original_features_num=2, node_emb=config['arch']['node_emb'])


def get_simple_model_exp(args, config_optim):  
    args_dict =  {'lr' : config_optim['lr'],
                'scheduler_decay': config_optim['scheduler_decay'],
                'scheduler_step': config_optim['scheduler_step'],
                'num_blocks': args['num_blocks'],
                'in_features': args['dim_features'],
                'out_features': args['dim_features'],
                'depth_of_mlp': args['depth_of_mlp'],
                'constant_n_vertices': False,
                'classifier': None
    }
    original_features_num = args['original_features_num']
    #node_emb = args['node_emb']
    #print('Fetching model %s with (total = %s ) init %s and inside %s' % (node_emb['type'], node_emb['num_blocks'],
    #    node_emb['block_init'], node_emb['block_inside']))
    model =  Graph_Classif_Exp(original_features_num, **args_dict)
    return model

def get_model_benchmark(args, config_optim, name_data):  
    args_dict =  {'lr' : config_optim['lr'],
                'scheduler_decay': config_optim['scheduler_decay'],
                'scheduler_step': config_optim['scheduler_step'],
                'num_blocks': args['node_emb']['num_blocks'],
                'in_features': args['node_emb']['in_features'],
                'out_features': args['node_emb']['out_features'],
                'depth_of_mlp': args['node_emb']['depth_of_mlp'],
                'constant_n_vertices': False,
                'classifier': None
    }
    original_features_num = 2 + NUM_LABELS[name_data]
    n_classes = NUM_CLASSES[name_data]
    #node_emb = args['node_emb']
    #print('Fetching model %s with (total = %s ) init %s and inside %s' % (node_emb['type'], node_emb['num_blocks'],
    #    node_emb['block_init'], node_emb['block_inside']))
    model =  Graph_Classif_Exp(original_features_num, n_classes = n_classes, **args_dict)
    return model