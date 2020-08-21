from models.siamese_net import Siamese_Model
from models.base_model import Simple_Node_Embedding, BaseModel, Simple_Edge_Embedding

def get_model(args):
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
