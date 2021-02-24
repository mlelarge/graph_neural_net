import os
import shutil
import json
from sacred import Experiment

import torch
import torch.backends.cudnn as cudnn
from toolbox import logger, metrics
from models import get_model
from loaders.siamese_loaders import siamese_loader
from loaders.data_generator import MCP_Generator,MCP_True_Generator
from toolbox.optimizer import get_optimizer
from toolbox.losses import loss_mcp
from toolbox import utils
import trainer as trainer

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

### BEGIN Sacred setup
ex = Experiment()
ex.add_config('default_mcp.yaml')

@ex.config_hook
def set_experiment_name(config, command_name, logger):
    ex.path = config['name']
    return config

@ex.config_hook
def update_config(config, command_name, logger):
    config.update(log_dir='{}/runs/{}/MCP_{}_{}_{}/'.format(
        config['root_dir'], config['name'],config['data']['n_vertices'],
        config['data']['clique_size'],config['data']['edge_density']),
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
    exp_logger.add_meters('train', metrics.make_meter_matching())
    exp_logger.add_meters('val', metrics.make_meter_matching())
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
        print("Best model yet saved")
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
    #print(data)
    gene_train = MCP_Generator('train', data)
    gene_train.load_dataset()
    train_loader = siamese_loader(gene_train, train['batch_size'],
                                  gene_train.constant_n_vertices)
    gene_val = MCP_Generator('val', data)
    gene_val.load_dataset()
    val_loader = siamese_loader(gene_val, train['batch_size'],
                                gene_val.constant_n_vertices)
    
    model = get_model(arch)
    optimizer, scheduler = get_optimizer(train,model)
    n_vertices, clique_size = data['n_vertices'],data['clique_size']
    edge_density = data['edge_density']
    weight = torch.ones((n_vertices,n_vertices))
    scaling = n_vertices**2*edge_density/clique_size**2
    weight[:clique_size,:clique_size] =  scaling* torch.ones((clique_size,clique_size))
    weight= weight.to(device)
    criterion = torch.nn.BCELoss(weight=weight,reduction='none')
    #criterion = torch.nn.BCELoss(reduction='mean')#loss_mcp(data['clique_size'], device, train['loss_reduction'])

    model.to(device)

    is_best = True
    #log_dir_ckpt = get_log_dir()
    #print(log_dir_ckpt)
    clique_size = data['clique_size']
    for epoch in range(train['epoch']):
        print('Current epoch: ', epoch)
        trainer.train_triplet(train_loader,model,criterion,optimizer,exp_logger,device,epoch,clique_size,eval_score=metrics.accuracy_mcp,print_freq=train['print_freq'])   

        acc, loss = trainer.val_triplet(val_loader,model,criterion,exp_logger,device,epoch,clique_size,eval_score=metrics.accuracy_mcp)
        scheduler.step(loss)
        # remember best acc and save checkpoint
        is_best = (acc > best_score)
        best_score = max(acc, best_score)
        if True == is_best:
            best_epoch = epoch
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'best_epoch': best_epoch,
            'exp_logger': exp_logger,
            }, is_best)
