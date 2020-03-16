"""
Usage:
    commander.py [options]

Option:
    -h --help                       show this screen.
    --name=<str>                    name of experiment [default: ER_std]
    --cpu=<str>                     use CPU [default: yes]
    --generative_model=<str>        so far ErdosRenyi, Regular or BarabasiAlbert [default: ErdosRenyi]
    --num_examples_train=<int>      [default: 20000]
    --num_examples_test=<int>       [default: 10]
    --num_examples_val=<int>        [default: 1000]
    --edge_density=<float>          [default: 0.2]
    --noise=<float>                 [default: 0.05]
    --n_vertices=<int>              number of vertices [default: 50]
    --vertex_proba=<float>          parameter of the binomial distribution of vertices [default: 1.]
    --path_dataset=<str>            path where datasets are stored [default: dataset]
    --root_dir=<str>                [default: .]
    --epoch=<int>                   [default: 5]
    --batch_size=<int>              [default: 32]
    --arch=<str>                    [default: Siamese_Model]
    --model_name=<str>              [default: Simple_Node_Embedding]
    --num_blocks=<int>              number of blocks [default: 2]
    --original_features_num=<int>   [default: 2]
    --in_features=<int>             [default: 64]
    --out_features=<int>            [default: 64]
    --depth_of_mlp=<int>            [default: 3]
    --print_freq=<int>              [default: 100]
    --lr=<float>                    learning rate [default: 1e-3]
    --step=<int>                    scheduler step [default: 5]
    --lr_decay=<float>              scheduler decay [default: 0.9]
    --seed=<int>                    random seed [default: 42]

"""


import os
import shutil
import json
from docopt import docopt

import torch
from toolbox import logger, metrics
from models import get_model
from loaders.siamese_loaders import siamese_loader
from loaders.data_generator import Generator
from toolbox.optimizer import get_optimizer
from toolbox.losses import get_criterion
from toolbox import utils
import trainer as trainer


list_float = ['--lr', '--edge_density', '--lr_decay', '--noise',
              '--vertex_proba']

list_int = ['--num_blocks', '--original_features_num',
            '--in_features', '--out_features',
            '--depth_of_mlp','--print_freq',
            '--epoch', '--batch_size', '--n_vertices',
            '--num_examples_train', '--num_examples_test',
            '--num_examples_val', '--step', '--seed']


def init_logger(args):
    # set loggers
    exp_name = args['--name']
    exp_logger = logger.Experiment(exp_name, args)
    exp_logger.add_meters('train', metrics.make_meter_matching())
    exp_logger.add_meters('val', metrics.make_meter_matching())
    exp_logger.add_meters('hyperparams', {'learning_rate': metrics.ValueMeter()})
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

def update_args(args):
    args['--log_dir'] = '{}/runs/{}/'.format(args['--root_dir'], args['--name'])
    #args['--res_dir'] = '{}/runs/{}/res'.format(args['--root_dir'], args['--name'])
    return args

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    utils.check_dir(args['--log_dir'])
    filename = os.path.join(args['--log_dir'], filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args['--log_dir'], 'model_best.pth.tar'))

    fn = os.path.join(args['--log_dir'], 'checkpoint_epoch{}.pth.tar')
    torch.save(state, fn.format(state['epoch']))

    if (state['epoch'] - 1 ) % 5 != 0:
      #remove intermediate saved models, e.g. non-modulo 5 ones
      if os.path.exists(fn.format(state['epoch'] - 1 )):
          os.remove(fn.format(state['epoch'] - 1 ))

    path_logger = os.path.join(args['--log_dir'], 'logger.json')
    state['exp_logger'].to_json(log_dir=args['--log_dir'],filename='logger.json')


def main():
    """ Main func.
    """
    global args, best_score, best_epoch
    best_score, best_epoch = -1, -1
    args = docopt(__doc__)
    args = type_args(args)
    args = update_args(args)
    use_cuda = not bool(args['--cpu']) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    # init random seeds 
    utils.setup_env(args)

    utils.init_output_env(args)

    exp_logger = init_logger(args)
    
    print(args['--batch_size'])
    gene_train = Generator('train', args)
    gene_train.load_dataset()
    train_loader = siamese_loader(gene_train, args['--batch_size'],
                                  gene_train.constant_n_vertices)
    gene_val = Generator('val', args)
    gene_val.load_dataset()
    val_loader = siamese_loader(gene_val, args['--batch_size'],
                                gene_val.constant_n_vertices)
    model = get_model(args)
    optimizer, scheduler = get_optimizer(args,model)
    criterion = get_criterion(device)

    exp_logger = init_logger(args)

    model.to(device)

    is_best = True
    for epoch in range(args['--epoch']):
        print('Current epoch: ', epoch)
        trainer.train_triplet(train_loader,model,criterion,optimizer,exp_logger,device,epoch,eval_score=metrics.accuracy_max)
        scheduler.step()
    #print(args['--num_examples_train'])

        acc = trainer.val_triplet(val_loader,model,criterion,exp_logger,device,epoch,eval_score=metrics.accuracy_linear_assigment)

        # remember best acc and save checkpoint
        is_best = acc > best_score
        best_score = max(acc, best_score)
        if True == is_best:
            best_epoch = epoch

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'best_epoch': best_epoch,
            'exp_logger': exp_logger,
        }, is_best)

if __name__ == '__main__':
    main()
