"""
Usage:
    commander.py [options]

Option:
    -h --help                       show this screen.
    --name=<str>                    name of experiment [default: ER_std]
    --cpu=<str>                    use CPU [default: yes]
    --generative_model=<str>        so far ErdosRenyi, Regular or BarabasiAlbert [default: ErdosRenyi]
    --num_examples_train=<int>      [default: 20000]
    --num_examples_test=<int>       [default: 10]
    --num_examples_val=<int>        [default: 10]
    --edge_density=<float>          [default: 0.2]
    --n_vertices=<int>              [default: 50]
    --path_dataset=<str>            path where datasets are stored [default: dataset]
    --epoch=<int>                   [default: 5]
    --batch_size=<int>              [default: 32]
    --arch=<str>                    [default: Siamese_Model]
    --model_name=<str>              [default: Simple_Node_Embedding]
    --num_blocks=<int>              number of blocks [default: 3]
    --original_features_num=<int>   [default: 2]
    --in_features=<int>             [default: 3]
    --out_features=<int>            [default: 5]
    --depth_of_mlp=<int>            [default: 2]
    --print_freq=<int>              [default: 10]
    --lr=<float>                    learning rate [default: 1e-3]

"""


import os
import torch
from toolbox import logger, metrics
from docopt import docopt
from models import get_model
from loaders.siamese_loaders import siamese_loader
from loaders.data_generator import Generator
from toolbox.optimizer import get_optimizer
from toolbox.losses import get_criterion
import trainer as trainer

list_float = ['--lr', '--edge_density']

list_int = ['--num_blocks', '--original_features_num',
            '--in_features', '--out_features',
            '--depth_of_mlp','--print_freq',
            '--epoch', '--batch_size', '--n_vertices',
            '--num_examples_train', '--num_examples_test',
            '--num_examples_val']


def init_logger(args):
    # set loggers
    exp_name = args['--name']
    exp_logger = logger.Experiment(exp_name, args.__dict__)
    exp_logger.add_meters('train', metrics.make_meter_matching())
    exp_logger.add_meters('val', metrics.make_meter_matching())
    exp_logger.add_meters('hyperparams', {'learning_rate': args['--lr']})
    return exp_logger

def type_args(args):
    for k in list_int:
        args[k] = int(args[k])
    for k in list_float:
        args[k] = float(args[k])
    if args['--cpu'] == 'yes':
        args['--cpu'] = True
    else:
        args['--cpu'] = False
    return args

def main():
    """ Main func.
    """
    global args, best_score, best_epoch
    best_score, best_epoch = -1, -1
    args = docopt(__doc__)
    args = type_args(args)
    use_cuda = not bool(args['--cpu']) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    exp_logger = init_logger(args)
    
    print(args['--batch_size'])
    gene_train = Generator('train', args)
    _ = gene_train.load_dataset()
    train_loader = siamese_loader(gene_train,args['--batch_size'])

    model = get_model(args)
    optimizer, _ = get_optimizer(args,model)
    criterion = get_criterion(device)

    model.to(device)
    trainer.train_triplet(train_loader,model,criterion,optimizer,device,0)
    print(args['--num_examples_train'])

if __name__ == '__main__':
    main()
