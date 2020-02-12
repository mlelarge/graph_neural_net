"""
Usage:
    commander.py [options]

Option:
    -h --help                       show this screen.
    --name                          [default: 'ER_std']
    --cuda                          use GPU
    --generative_model=<str>        so far ErdosRenyi, Regular or BarabasiAlbert [default: 'ErdosRenyi']
    --num_examples_train=<int>      [default: 20000]
    --num_examples_test=<int>       [default: 1000]
    --num_examples_val=<int>        [default: 1000]
    --edge_density=<float>          [default: 0.2]
    --n_vertices=<int>              [default: 50]
    --path_dataset                  path where datasets are stored [default: '/dataset/']
    --epoch=<int>                   [default: 5]
    --batch_size=<int>              [default: 32]
    --num_blocks=<int>              [default: 3]
    --original_features_num=<int>   [default: 2]
    --in_features-<int>             [default: 3]
    --out_features=<int>            [default: 5]
    --depth_of_mlp=<int>            [default: 2]
    --print_freq=<int>              [default: 10]
    --lr=<float>                    learning rate [default: 1e-3]

"""


import os
from toolbox import logger, metrics
from docopt import docopt

def init_logger(args):
    # set loggers
    exp_name = args['--name']
    exp_logger = logger.Experiment(exp_name, args.__dict__)
    exp_logger.add_meters('train', metrics.make_meter_matching())
    exp_logger.add_meters('val', metrics.make_meter_matching())
    exp_logger.add_meters('hyperparams', {'learning_rate': args['--lr']})
    return exp_logger

def main():
    """ Main func.
    """
    args = docopt(__doc__)
    init_logger(args)
    print(args['--cuda'])
    print(args['--num_examples_train'])

if __name__ == '__main__':
    main()
