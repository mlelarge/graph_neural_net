import os
import shutil
import json
from sacred import Experiment

import torch
import torch.backends.cudnn as cudnn
from toolbox import logger, metrics
from models import get_model
from loaders.siamese_loaders import siamese_loader
from loaders.tsp_data import TSP
from toolbox.optimizer import get_optimizer
from toolbox.losses import tsp_loss
from toolbox import utils
import trainer as trainer

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

### BEGIN Sacred setup
ex = Experiment()
ex.add_config('default_tsp.yaml')

@ex.config_hook
def set_experiment_name(config, command_name, logger):
    ex.path = config['name']
    return config

@ex.config_hook
def update_config(config, command_name, logger):
    config.update(log_dir='{}/runs/{}/TSP_{}_{}_{}_{}_{}_{}/'.format(
        config['root_dir'], config['name'], config['arch']['arch'], config['data']['n_vertices'],
        config['arch']['num_blocks'],config['arch']['in_features'],
        config['arch']['out_features'], config['arch']['depth_of_mlp']),
                   # Some funcs need path_dataset while not requiring the whole data dict
                   path_dataset=config['data']['path_dataset'])
                   #res_dir='{}/runs/{}/res'.format(config['root_dir'], config['name'])
    return config

@ex.config_hook
def init_observers(config, command_name, logger):
    neptune = config['observers']['neptune']
    if neptune['enable']:
        from neptunecontrib.monitoring.sacred import NeptuneObserver
        ex.observers.append(NeptuneObserver(project_name=neptune['project']))
    return config

@ex.post_run_hook
def clean_observer(observers):
    """ Observers that are added in a config_hook need to be cleaned """
    neptune = observers['neptune']
    if neptune['enable']:
        from neptunecontrib.monitoring.sacred import NeptuneObserver
        ex.observers = [obs for obs in ex.observers if not isinstance(obs, NeptuneObserver)]

### END Sacred setup

@ex.capture
def init_logger(name, _config, _run):
    # set loggers
    exp_logger = logger.Experiment(name, _config, run=_run)
    exp_logger.add_meters('train', metrics.make_meter_tsp())
    exp_logger.add_meters('val', metrics.make_meter_tsp())
    #exp_logger.add_meters('test', metrics.make_meter_matching())
    exp_logger.add_meters('hyperparams', {'learning_rate': metrics.ValueMeter()})
    return exp_logger
 
@ex.capture
def setup_env(cpu):
    # Randomness is already controlled by Sacred
    # See https://sacred.readthedocs.io/en/stable/randomness.html
    if not cpu:
        cudnn.benchmark = True

# create necessary folders and config files
@ex.capture
def init_output_env(_config, root_dir, log_dir, path_dataset):
    utils.check_dir(os.path.join(root_dir,'runs'))
    utils.check_dir(log_dir)
    utils.check_dir(path_dataset)
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(_config, f)

@ex.capture
def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth.tar'):
    utils.check_dir(log_dir)
    filename = os.path.join(log_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(log_dir, 'model_best.pth.tar'))

    fn = os.path.join(log_dir, 'checkpoint_epoch{}.pth.tar')
    torch.save(state, fn.format(state['epoch']))

    if (state['epoch'] - 1 ) % 5 != 0:
      #remove intermediate saved models, e.g. non-modulo 5 ones
      if os.path.exists(fn.format(state['epoch'] - 1 )):
          os.remove(fn.format(state['epoch'] - 1 ))

    state['exp_logger'].to_json(log_dir=log_dir,filename='logger.json')


@ex.automain
def main(cpu, data, train, arch):
    """ Main func.
    """
    global best_score, best_epoch
    best_score, best_epoch = -1, -1
    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    # init random seeds 
    setup_env()

    init_output_env()
    exp_logger = init_logger()
    
    #dataset_train = TSP('dataset_tsp',split='train')
    dataset_train = TSP('TSP',file_name='tsp50-500_' ,split='train')
    train_loader = siamese_loader(dataset_train,train['batch_size'],constant_n_vertices=True)
    #dataset_val = TSP('dataset_tsp',split='val')
    dataset_val = TSP('TSP',file_name='tsp50-500_' ,split='val')
    val_loader = siamese_loader(dataset_val,train['batch_size'],constant_n_vertices=True)
    
    model = get_model(arch)
    model_path = './runs/TSP-50-new/TSP_Simple_Edge_Embedding_50_4_64_1_3'
    model_file = os.path.join(model_path,'model_best.pth.tar')
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer, scheduler = get_optimizer(train,model)
    criterion = tsp_loss()

    # exp_logger = init_logger(args)

    model.to(device)

    is_best = True
    #log_dir_ckpt = get_log_dir()
    #print(log_dir_ckpt)
    for epoch in range(train['epoch']):
        print('Current epoch: ', epoch)
        trainer.train_tsp(train_loader,model,criterion,optimizer,
        exp_logger,device,epoch,eval_score=metrics.compute_f1,
        print_freq=train['print_freq'])
        
        f1, loss = trainer.val_tsp(val_loader,model,criterion,
        exp_logger,device,epoch,eval_score=metrics.compute_f1)
        #print_freq=train['print_freq'])
        scheduler.step(loss)
        # remember best acc and save checkpoint
        is_best = (f1 > best_score)
        best_score = max(f1, best_score)
        if True == is_best:
            best_epoch = epoch
            #acc_test = trainer.val_triplet(val_loader,model,criterion,exp_logger,device,epoch,eval_score=metrics.accuracy_linear_assignment,val_test='test')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'best_epoch': best_epoch,
            'exp_logger': exp_logger,
            }, is_best)
