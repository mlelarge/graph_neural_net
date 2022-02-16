from models.siamese_net import Siamese_Model,Siamese_Model_Gen
from models.base_model import Simple_Node_Embedding, Simple_Edge_Embedding, RS_Node_Embedding
from models.gcn_model import BaseGCN
from models.gated_gcn import GatedGCN, GatedGCNNet_Edge, GatedGCNNet_Node

# def get_model(args):

#     args_dict = {'arch_type': args['arch_type'],
#                 'original_features_num': args['original_features_num'],
#                 'num_blocks': args['num_blocks'],
#                 'in_features': args['dim_features'],
#                 'out_features': args['dim_features'],
#                 'depth_of_mlp': args['depth_of_mlp']
#     }
#     embedding_dict = {'node': Simple_Node_Embedding, 'edge': Simple_Edge_Embedding}

#     arch = args['arch_gnn'].lower()
#     arch_type = args['arch_type'].lower()
#     embedding = args['embedding'].lower()

#     if arch_type=='simple':
#         Model_instance = embedding_dict[embedding]
#     elif arch_type=='siamese':
#         Model_instance = Siamese_Model
#         args_dict['embedding_class'] = embedding_dict[embedding] #Add the type of embedding wanted
#     else:
#         raise NotImplementedError(f"{arch_type} architecture type not implemented")

#     print('Fetching model %s - %s ' % (args['arch'], args['embedding'] + ' embedding'))

#     model =  Model_instance(**args_dict)
#     return model

def get_model(args):
    # used in the jupyter notebook (old config!)
    model_instance = _get_model_instance(args['arch'])

    print('Fetching model %s - %s ' % (args['arch'], args['model_name']))
    model =  model_instance(original_features_num=args['original_features_num'],
                num_blocks=args['num_blocks'],
                in_features=args['in_features'],
                out_features=args['out_features'],
                depth_of_mlp=args['depth_of_mlp'])
    return model

def _get_model_instance(arch):
    return {'Siamese_Model': Siamese_Model, 
    'Simple_Node_Embedding': Simple_Node_Embedding,
    'Simple_Edge_Embedding': Simple_Edge_Embedding}[arch]

def get_model_gen(args):

    args_dict = {'arch_load': args['arch_load'],
                'original_features_num': args['original_features_num'],
                'num_blocks': args['num_blocks'],
                'in_features': args['dim_features'],
                'out_features': args['dim_features'],
                'depth_of_mlp': args['depth_of_mlp'],
                'input_embed': args['input_embed']
    }

    arch = args['arch_gnn'].lower()
    arch_load = args['arch_load'].lower()
    embedding = args['embedding'].lower()

    if arch_load=='simple':
        loader_function = lambda model,*args,**kwargs : model(*args,**kwargs)
    elif arch_load=='siamese':
        loader_function = Siamese_Model_Gen
 
    fgnn_embedding_dict = {'node': Simple_Node_Embedding, 
        'rs_node': RS_Node_Embedding,
        'edge': Simple_Edge_Embedding}
    gatedgcn_embedding_dict = {'node': GatedGCNNet_Node, 'edge': GatedGCNNet_Edge}

    if arch=='fgnn':
        try:
            Model_instance = fgnn_embedding_dict[embedding]
        except KeyError:
            raise NotImplementedError(f"{embedding} is not a keyword for the FGNN architecture (should be 'node' or 'edge'")
    elif arch=='gcn':
        Model_instance = BaseGCN
    elif arch=='gatedgcn':
        try:
            Model_instance = gatedgcn_embedding_dict[embedding]
        except KeyError:
            raise NotImplementedError(f"{embedding} is not a keyword for the GatedGCN architecture (should be 'node' or 'edge'")
    else:
        raise NotImplementedError(f"{arch} architectuce not implemented")
    
    print('Fetching model %s %s - (%s  embedding if fgnn)' % (arch,args['arch_load'], args['embedding']))

    model =  loader_function(Model_instance,**args_dict)
    return model