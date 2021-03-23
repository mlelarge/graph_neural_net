from torch.nn.functional import embedding
from models.siamese_net import Siamese_Model
from models.base_model import Simple_Node_Embedding, BaseModel, Simple_Edge_Embedding

def get_model(args):

    args_dict = {'original_features_num': args['original_features_num'],
                'num_blocks': args['num_blocks'],
                'in_features': args['in_features'],
                'out_features': args['out_features'],
                'depth_of_mlp': args['depth_of_mlp']
    }
    embedding_dict = {'node': Simple_Node_Embedding, 'edge': Simple_Edge_Embedding}

    arch = args['arch'].lower()
    embedding = args['embedding'].lower()

    if arch=='simple':
        model_instance = embedding_dict[embedding]
    elif arch=='siamese':
        model_instance = Siamese_Model
        args_dict['embedding_class'] = embedding_dict[embedding] #Add the type of embedding wanted
    else:
        raise NotImplementedError(f"{arch} architecture not implemented")

    print('Fetching model %s - %s ' % (args['arch'], args['embedding'] + ' embedding'))

    model =  model_instance(**args_dict)
    return model
