import os
import shutil
import json
from sacred import Experiment
from sacred.config import config_scope

import torch
import torch.backends.cudnn as cudnn
from toolbox import logger, metrics
from models import get_model
from loaders.siamese_loaders import siamese_loader
from loaders.data_generator import QAP_Generator,TSP_Generator,MCP_Generator
from toolbox.optimizer import get_optimizer
import toolbox.losses
import toolbox.utils as utils
import trainer as trainer

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

### BEGIN Sacred setup
ex = Experiment()
ex.add_config('default_config.yaml')

def get_generator(pbm):
    """
    returns the problem associated generator
    """
    if pbm=="qap":
        generator = QAP_Generator
    elif pbm=="tsp":
        generator = TSP_Generator
    elif pbm=="mcp":
        generator = MCP_Generator
    return generator

def get_eval(pbm):
    """
    returns the problem associated eval function
    """
    if pbm=="qap":
        func = metrics.accuracy_max
    elif pbm=="tsp":
        func = metrics.compute_f1
    elif pbm=="mcp":
        func = metrics.accuracy_mcp
    return func

def get_criterion(pbm, device='cpu',reduction='mean'):
    """
    returns the problem associated eval function
    """
    if pbm=="qap":
        crit = toolbox.losses.triplet_loss(device=device,loss_reduction=reduction)
    elif pbm=="tsp":
        crit = toolbox.losses.tsp_loss(loss = torch.nn.BCELoss(reduction=reduction))
    elif pbm=="mcp":
        crit = torch.nn.BCELoss(reduction=reduction)
    return crit

@ex.config_hook
def create_data_dict(config, command_name, logger):
    """
    Creates the parameter dictionaries for the data generation
    """
    problem_key = "_" + config['problem']
    train_config = config['data']['train']
    test_config = config['data']['test']
    custom_test = config['test_enabled'] and config['data']['test']
    assert problem_key in list(train_config.keys())
    train_data_dict = dict()
    test_data_dict = dict()
    for key,value in train_config.items():
        if key[0]!='_': #Don't add the problem data yet
            train_data_dict[key] = value
            if not custom_test:
                test_data_dict[key] = value
            else:
                test_data_dict[key] = test_config[key]
        elif key==problem_key:#Add problem data
            for pb_key,pb_value in train_config[key].items():
                train_data_dict[pb_key] = pb_value
                if not custom_test:
                    test_data_dict[pb_key] = pb_value
                else:
                    test_data_dict[pb_key] = test_config[key][pb_key]
    
    test_data_dict['num_examples_test'] = test_config['num_examples_test'] #Special case of the numer of examples

    config.update(
        test_data_dict=test_data_dict,
        train_data_dict=train_data_dict
    )
    return config

@ex.config_hook
def set_experiment_name(config, command_name, logger):
    ex.path = config['name']
    return config

@ex.config_hook
def update_config(config, command_name, logger):
    pbm = config['problem']
    pbm_key = "_"+pbm
    l_params =[value for _,value in (config['data']['train'][pbm_key]).items()]
    config_str = "_".join([str(item) for item in l_params])
    config.update(log_dir='{}/runs/{}/{}-{}/'.format(
        config['root_dir'], config['name'],
        config['problem'].upper(),
        config_str), #End log_dir
        path_dataset=config['data']['train'][pbm_key]['path_dataset']
    )
    """
    config.update(log_dir='{}/runs/{}/{}_{}_{}_{}_{}_{}_{}/'.format(
        config['root_dir'], config['name'],
        config['problem'].upper(),
        config['data']['generative_model'],
        config['data']['noise_model'], config['data']['n_vertices'],
        config['data']['vertex_proba'],config['data']['noise'],config['data']['edge_density']),
                   # Some funcs need path_dataset while not requiring the whole data dict
                   path_dataset=config['data']['path_dataset'])
                   #res_dir='{}/runs/{}/res'.format(config['root_dir'], config['name'])
    """
    return config

@ex.config_hook
def init_observers(config, command_name, logger):
    neptune = config['observers']['neptune']
    problem = config['problem']
    if neptune['enable']:
        from neptunecontrib.monitoring.sacred import NeptuneObserver
        ex.observers.append(NeptuneObserver(project_name=neptune['project']+problem.upper()))
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
def init_logger(name, _config, _run, problem):
    # set loggers
    exp_logger = logger.Experiment_Helper(problem, name, _config, run=_run)
    #exp_logger.add_meters('train', metrics.make_meter_matching()) #Initialization is now taken care of into Experiment Helper
    #exp_logger.add_meters('val', metrics.make_meter_matching())
    #exp_logger.add_meters('test', metrics.make_meter_matching())
    #exp_logger.add_meters('hyperparams', {'learning_rate': metrics.ValueMeter()})
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
def main(cpu, train, problem, train_data_dict, test_data_dict, arch, test_enabled):
    """ Main func.
    """
    global best_score, best_epoch
    best_score, best_epoch = -1, -1
    print("Current problem : ", problem)

    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    # init random seeds 
    setup_env()

    init_output_env()
    exp_logger = init_logger()

    generator = get_generator(problem)
    
    gene_train = generator('train', train_data_dict)
    gene_train.load_dataset()
    train_loader = siamese_loader(gene_train, train['batch_size'],
                                  gene_train.constant_n_vertices)
    gene_val = generator('val', train_data_dict)
    gene_val.load_dataset()
    val_loader = siamese_loader(gene_val, train['batch_size'],
                                gene_val.constant_n_vertices)
    
    if test_enabled:
        gene_test = generator('test', test_data_dict)
        gene_test.load_dataset()
        test_loader = siamese_loader(gene_test, train['batch_size'],
                                    gene_test.constant_n_vertices)
    
    model = get_model(arch)
    optimizer, scheduler = get_optimizer(train,model)
    criterion = get_criterion(problem, device, train['loss_reduction'])

    eval_score = get_eval(problem)

    model.to(device)

    is_best = True
    #log_dir_ckpt = get_log_dir()
    #print(log_dir_ckpt)
    for epoch in range(train['epoch']):
        print('Current epoch: ', epoch)
        trainer.train_triplet(train_loader,model,criterion,optimizer,exp_logger,device,epoch,eval_score=True,print_freq=train['print_freq'])
        


        acc, loss = trainer.val_triplet(val_loader,model,criterion,exp_logger,device,epoch,eval_score=True)
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
            'exp_logger': exp_logger.get_logger(),
            }, is_best)
    
    if test_enabled:
        pass
