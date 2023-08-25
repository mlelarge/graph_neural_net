from models.trainers import Siamese_Node_Exp#, Graph_Classif_Exp #, scaled_block, block, block_sym, Graph_Classif_Exp
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

def get_siamese_model_test(name, config=None):
    if config is None:
        split_name = name.split("/")[-4]
        cname = name.split(split_name)[0]
        config = load_json(cname+'config.json')
    return Siamese_Node_Exp.load_from_checkpoint(name, original_features_num=2, node_emb=config['arch']['node_emb'])
