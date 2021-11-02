from models.siamese_net import Siamese_Model,Siamese_Model_Gen
from models.base_model import Simple_Node_Embedding, Simple_Edge_Embedding
from models.gcn_model import BaseGCN

def get_model(args):

    args_dict = {'arch_type': args['arch_type'],
                'original_features_num': args['original_features_num'],
                'num_blocks': args['num_blocks'],
                'in_features': args['in_features'],
                'out_features': args['out_features'],
                'depth_of_mlp': args['depth_of_mlp']
    }
    embedding_dict = {'node': Simple_Node_Embedding, 'edge': Simple_Edge_Embedding}

    arch = args['arch'].lower()
    arch_type = args['arch_type'].lower()
    embedding = args['embedding'].lower()

    if arch_type=='simple':
        Model_instance = embedding_dict[embedding]
    elif arch_type=='siamese':
        Model_instance = Siamese_Model
        args_dict['embedding_class'] = embedding_dict[embedding] #Add the type of embedding wanted
    else:
        raise NotImplementedError(f"{arch_type} architecture type not implemented")

    print('Fetching model %s - %s ' % (args['arch'], args['embedding'] + ' embedding'))

    model =  Model_instance(**args_dict)
    return model

def get_model_gen(args):
    fgnn_embedding_dict = {'node': Simple_Node_Embedding, 'edge': Simple_Edge_Embedding}

    args_dict = {'arch_load': args['arch_load'],
                'original_features_num': args['original_features_num'],
                'num_blocks': args['num_blocks'],
                'in_features': args['in_features'],
                'out_features': args['out_features'],
                'depth_of_mlp': args['depth_of_mlp']
    }

    arch = args['arch'].lower()
    arch_load = args['arch_load'].lower()
    embedding = args['embedding'].lower()

    if arch_load=='simple':
        loader_function = lambda model,*args,**kwargs : model(*args,**kwargs)
    elif arch_load=='siamese':
        loader_function = Siamese_Model_Gen

    if arch=='fgnn':
        try:
            Model_instance = fgnn_embedding_dict[embedding]
        except KeyError:
            raise NotImplementedError(f"{embedding} is not a keyword for the FGNN architecture (should be 'node' or 'edge'")
    elif arch=='gcn':
        Model_instance = BaseGCN
    else:
        raise NotImplementedError(f"{arch} architectuce not implemented")
    
    print('Fetching model %s %s - %s ' % (arch,args['arch_load'], args['embedding'] + ' embedding (if fgnn for now)'))

    model =  loader_function(Model_instance,**args_dict)
    return model